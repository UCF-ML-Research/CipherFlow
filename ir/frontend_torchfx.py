"""PyTorch -> OperatorGraph frontend via torch.fx (Section C).

Takes a `torch.nn.Module`, runs `torch.fx.symbolic_trace` (paper ref
[32]), propagates tensor shapes with `torch.fx.passes.shape_prop.ShapeProp`,
and collapses the fx graph into an `ir.types.OperatorGraph` whose nodes
are drawn from the compiler's canonical operator vocabulary:

    Attention_QK_MatMul, Softmax, Attention_V_MatMul, Out_Projection,
    Residual_Add, LayerNorm, FFN_Linear_1, GeLU, FFN_Linear_2

This is a *module-level* frontend, not a full aten lowering. It matches
torch submodules and a narrow set of `call_function` ops onto the
operator vocabulary by type. The heuristic is explicit and documented
below; unknown ops become `Unknown` nodes that the caller can detect
before handing the graph to the compiler.

Design notes
------------

- torch.fx is imported lazily so the rest of the framework never takes
  a hard dependency on torch.
- Shape propagation runs on a caller-provided example input so every
  emitted `OperatorNode` carries `input_shape` and `output_shape`.
- Linear layers are disambiguated into `FFN_Linear_1` / `FFN_Linear_2`
  / `Out_Projection` by examining the producer pattern *around* them
  (a linear feeding a GeLU is FFN_Linear_1; the linear consumed by a
  GeLU output is FFN_Linear_2; the linear immediately after an
  attention V matmul is Out_Projection). This is the same rule the
  hand-authored `ir/bert_block_builder.py` applies.
- `nn.MultiheadAttention` is expanded into the four-node attention
  sequence (`Attention_QK_MatMul -> Softmax -> Attention_V_MatMul ->
  Out_Projection`). Raw matmuls (`torch.matmul`) inside a manually
  written attention block are also recognised and routed to the
  `Attention_QK_MatMul` / `Attention_V_MatMul` types based on whether
  a softmax sits on the producer path.

The function raises `TorchFxFrontendUnavailable` if `torch` or
`torch.fx` cannot be imported — the test suite uses
`pytest.importorskip("torch")` to skip gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode


Shape = Tuple[int, ...]


class TorchFxFrontendUnavailable(RuntimeError):
    """Raised when torch / torch.fx cannot be imported."""


# Canonical op vocabulary — must match runtime.operator_specs.
_OP_TYPES = {
    "Attention_QK_MatMul",
    "Softmax",
    "Attention_V_MatMul",
    "Out_Projection",
    "Residual_Add",
    "LayerNorm",
    "FFN_Linear_1",
    "GeLU",
    "FFN_Linear_2",
    "Unknown",
}


@dataclass(frozen=True)
class TorchFxTraceConfig:
    """Frontend configuration.

    - ``graph_id``: graph_id assigned to the emitted OperatorGraph.
    - ``ffn_hidden_multiplier``: used to identify FFN_Linear_1 vs
      FFN_Linear_2 when the surrounding context is ambiguous; in a
      standard BERT block the first linear expands hidden_size → 4x
      and the second contracts 4x → hidden_size. Default = 4.
    - ``skip_output_node``: if True, drop the fx `output` sink node.
    - ``fail_on_unknown``: raise instead of emitting `Unknown` nodes.
    """

    graph_id: str = "fx_graph"
    ffn_hidden_multiplier: int = 4
    skip_output_node: bool = True
    fail_on_unknown: bool = False


def _import_torch():
    try:
        import torch
        import torch.fx as fx
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError as exc:
        raise TorchFxFrontendUnavailable(
            "torch / torch.fx not available; install torch to use the fx frontend"
        ) from exc
    return torch, fx, ShapeProp


def _shape_of(node: Any) -> Shape:
    meta = getattr(node, "meta", {}).get("tensor_meta")
    if meta is None:
        return ()
    shape = getattr(meta, "shape", None)
    if shape is None:
        return ()
    return tuple(int(d) for d in shape)


def _classify_call_module(
    submodule: Any,
    *,
    torch_mod: Any,
) -> str:
    """Return one of the canonical op types for a module instance."""
    nn = torch_mod.nn
    if isinstance(submodule, nn.LayerNorm):
        return "LayerNorm"
    if isinstance(submodule, nn.GELU):
        return "GeLU"
    if isinstance(submodule, nn.Softmax):
        return "Softmax"
    if isinstance(submodule, nn.Linear):
        # Ambiguous at this site — the caller disambiguates based on
        # surrounding pattern.
        return "Linear"
    if isinstance(submodule, nn.MultiheadAttention):
        return "MultiheadAttention"
    return "Unknown"


def _classify_call_function(target: Any) -> str:
    import operator as py_operator

    if target is py_operator.add:
        return "Residual_Add"
    name = getattr(target, "__name__", "") or str(target)
    # torch-namespaced functions
    lowered = name.lower()
    if "layer_norm" in lowered:
        return "LayerNorm"
    if "gelu" in lowered:
        return "GeLU"
    if "softmax" in lowered:
        return "Softmax"
    if "matmul" in lowered or "bmm" in lowered:
        return "MatMul"
    if lowered == "add":
        return "Residual_Add"
    return "Unknown"


def _classify_call_method(method_name: str) -> str:
    if method_name == "matmul":
        return "MatMul"
    if method_name == "softmax":
        return "Softmax"
    return "Unknown"


def _disambiguate_linear(
    fx_node: Any,
    op_type_by_fx_node: Dict[str, str],
    graph_module: Any,
    *,
    ffn_hidden_multiplier: int,
) -> str:
    """Decide whether a plain nn.Linear is FFN_Linear_1/2 or Out_Projection.

    Heuristic (in order):
      1. Inspect the user (successor) of this linear. If any user is a
         GeLU it is FFN_Linear_1. If any user is a Residual_Add and the
         producer path includes a GeLU it is FFN_Linear_2.
      2. Inspect the producer (predecessor) of this linear. If the
         producer is an Attention_V_MatMul it is Out_Projection.
      3. Fall back on shape ratios: if out_features == in_features *
         `ffn_hidden_multiplier`, it is FFN_Linear_1; if
         in_features == out_features * multiplier, it is FFN_Linear_2;
         if in_features == out_features, it is Out_Projection.
    """
    users = list(fx_node.users)
    for user in users:
        cls = op_type_by_fx_node.get(user.name)
        if cls == "GeLU":
            return "FFN_Linear_1"
        if cls == "Residual_Add":
            # Check the producer path for a GeLU — if found, this is ffn2.
            for prev in _walk_producers(fx_node, op_type_by_fx_node):
                if prev == "GeLU":
                    return "FFN_Linear_2"
            return "Out_Projection"

    for prev_type in _walk_producers(fx_node, op_type_by_fx_node):
        if prev_type == "Attention_V_MatMul":
            return "Out_Projection"

    # Shape-ratio fallback
    submodule = graph_module.get_submodule(fx_node.target)
    in_f = int(getattr(submodule, "in_features", 0))
    out_f = int(getattr(submodule, "out_features", 0))
    if in_f > 0 and out_f > 0:
        if out_f == in_f * ffn_hidden_multiplier:
            return "FFN_Linear_1"
        if in_f == out_f * ffn_hidden_multiplier:
            return "FFN_Linear_2"
        if in_f == out_f:
            return "Out_Projection"
    return "Unknown"


def _walk_producers(fx_node: Any, op_type_by_fx_node: Dict[str, str]) -> List[str]:
    """Walk predecessor chain up to 4 hops back, emitting op_types."""
    out: List[str] = []
    frontier = [a for a in fx_node.all_input_nodes]
    visited = set()
    hops = 0
    while frontier and hops < 4:
        next_frontier = []
        for n in frontier:
            if n.name in visited:
                continue
            visited.add(n.name)
            op = op_type_by_fx_node.get(n.name)
            if op:
                out.append(op)
            next_frontier.extend(a for a in n.all_input_nodes)
        frontier = next_frontier
        hops += 1
    return out


def _classify_matmul(
    fx_node: Any,
    op_type_by_fx_node: Dict[str, str],
) -> str:
    """Route a raw matmul onto QK or V MatMul based on downstream softmax."""
    # If any *user* of this matmul is a Softmax (before another matmul),
    # this is Attention_QK_MatMul. If a *producer* is a Softmax, it's V.
    for user in fx_node.users:
        if op_type_by_fx_node.get(user.name) == "Softmax":
            return "Attention_QK_MatMul"
    for prev_type in _walk_producers(fx_node, op_type_by_fx_node):
        if prev_type == "Softmax":
            return "Attention_V_MatMul"
    return "Attention_QK_MatMul"  # conservative default


def trace_module_to_graph(
    module: Any,
    example_input: Any,
    cfg: Optional[TorchFxTraceConfig] = None,
) -> OperatorGraph:
    """Symbolic-trace `module`, propagate shapes, and return OperatorGraph.

    `example_input` is either a single tensor or a tuple of tensors
    matching `module.forward`'s signature.
    """
    torch_mod, fx_mod, ShapeProp = _import_torch()
    cfg = cfg or TorchFxTraceConfig()

    graph_module = fx_mod.symbolic_trace(module)
    if isinstance(example_input, tuple):
        ShapeProp(graph_module).propagate(*example_input)
    else:
        ShapeProp(graph_module).propagate(example_input)

    op_type_by_fx_node: Dict[str, str] = {}
    # First pass: assign preliminary op types for every fx node.
    for node in graph_module.graph.nodes:
        if node.op == "placeholder" or node.op == "get_attr":
            continue
        if node.op == "output":
            continue
        if node.op == "call_module":
            submod = graph_module.get_submodule(node.target)
            op_type_by_fx_node[node.name] = _classify_call_module(submod, torch_mod=torch_mod)
            continue
        if node.op == "call_function":
            op_type_by_fx_node[node.name] = _classify_call_function(node.target)
            continue
        if node.op == "call_method":
            op_type_by_fx_node[node.name] = _classify_call_method(str(node.target))
            continue

    # Second pass: disambiguate Linear / MatMul using neighbours.
    resolved: Dict[str, str] = dict(op_type_by_fx_node)
    for node in graph_module.graph.nodes:
        name = node.name
        op = op_type_by_fx_node.get(name)
        if op == "Linear":
            resolved[name] = _disambiguate_linear(
                node,
                op_type_by_fx_node,
                graph_module,
                ffn_hidden_multiplier=cfg.ffn_hidden_multiplier,
            )
        elif op == "MatMul":
            resolved[name] = _classify_matmul(node, op_type_by_fx_node)
        elif op == "MultiheadAttention":
            # placeholder — the caller sees this and must expand it.
            resolved[name] = "MultiheadAttention"

    # Third pass: emit IR nodes and edges.
    ir_nodes: List[OperatorNode] = []
    ir_edges: List[DataEdge] = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder" or node.op == "get_attr":
            continue
        if node.op == "output" and cfg.skip_output_node:
            continue
        op_type = resolved.get(node.name, "Unknown")
        if op_type == "Unknown" and cfg.fail_on_unknown:
            raise ValueError(
                f"fx frontend could not classify node {node.name} "
                f"(op={node.op}, target={node.target!r})"
            )
        # MultiheadAttention collapses to 4 nodes — expand inline.
        if op_type == "MultiheadAttention":
            base = node.name
            in_shape = _shape_of(list(node.all_input_nodes)[0]) if node.all_input_nodes else ()
            out_shape = _shape_of(node)
            for suffix, sub_op in (
                ("_qk", "Attention_QK_MatMul"),
                ("_softmax", "Softmax"),
                ("_v", "Attention_V_MatMul"),
                ("_outproj", "Out_Projection"),
            ):
                ir_nodes.append(
                    OperatorNode(
                        node_id=f"{base}{suffix}",
                        op_type=sub_op,
                        input_shape=in_shape,
                        output_shape=out_shape,
                        attributes={"source_fx_node": base, "expanded_from": "MultiheadAttention"},
                    )
                )
            # Wire them into an internal chain.
            for a, b in zip(
                ["_qk", "_softmax", "_v"], ["_softmax", "_v", "_outproj"]
            ):
                ir_edges.append(
                    DataEdge(src=f"{base}{a}", dst=f"{base}{b}", tensor_shape=out_shape)
                )
            continue

        input_nodes = list(node.all_input_nodes)
        in_shape = _shape_of(input_nodes[0]) if input_nodes else ()
        out_shape = _shape_of(node)
        ir_nodes.append(
            OperatorNode(
                node_id=node.name,
                op_type=op_type,
                input_shape=in_shape,
                output_shape=out_shape,
                attributes={"source_fx_node": node.name, "fx_op": node.op},
            )
        )

    # Fourth pass: build edges between emitted nodes. Skip fx nodes we
    # dropped (placeholder / output / get_attr). For expanded
    # MultiheadAttention, the external boundary points at `${base}_qk`
    # on input and `${base}_outproj` on output.
    emitted_ids = {n.node_id for n in ir_nodes}
    mha_bases: Dict[str, Tuple[str, str]] = {}
    for ir_node in ir_nodes:
        if ir_node.attributes.get("expanded_from") == "MultiheadAttention":
            base = str(ir_node.attributes["source_fx_node"])
            mha_bases.setdefault(base, (f"{base}_qk", f"{base}_outproj"))

    def _external_id(fx_name: str, is_target: bool) -> Optional[str]:
        if fx_name in mha_bases:
            return mha_bases[fx_name][0 if is_target else 1]
        return fx_name if fx_name in emitted_ids else None

    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output" and cfg.skip_output_node:
            continue
        dst_id = _external_id(node.name, is_target=True)
        if dst_id is None:
            continue
        out_shape = _shape_of(node)
        seen = set()
        for pred in node.all_input_nodes:
            src_id = _external_id(pred.name, is_target=False)
            if src_id is None:
                continue
            pair = (src_id, dst_id)
            if pair in seen:
                continue
            seen.add(pair)
            ir_edges.append(
                DataEdge(
                    src=src_id,
                    dst=dst_id,
                    tensor_shape=_shape_of(pred) or out_shape,
                )
            )

    return OperatorGraph(
        graph_id=cfg.graph_id,
        nodes=ir_nodes,
        edges=ir_edges,
    )


def graph_executed_subset(graph: OperatorGraph) -> OperatorGraph:
    """Return a copy of `graph` with `Unknown` nodes dropped.

    Edges touching dropped nodes are rewritten so the surrounding graph
    stays connected. Useful when the fx frontend emits `Unknown` for
    framework glue (e.g. reshapes) that the compiler can safely ignore.
    """
    kept_ids = {n.node_id for n in graph.nodes if n.op_type != "Unknown"}
    predecessors: Dict[str, List[str]] = {}
    shape_by_src: Dict[str, Shape] = {}
    for edge in graph.edges:
        predecessors.setdefault(edge.dst, []).append(edge.src)
        shape_by_src[edge.src] = edge.tensor_shape

    def _resolve(src: str) -> List[str]:
        if src in kept_ids:
            return [src]
        out: List[str] = []
        for upstream in predecessors.get(src, []):
            out.extend(_resolve(upstream))
        return out

    new_edges: List[DataEdge] = []
    seen_pairs: set = set()
    for edge in graph.edges:
        if edge.dst not in kept_ids:
            continue
        for real_src in _resolve(edge.src):
            if real_src == edge.dst:
                continue
            pair = (real_src, edge.dst)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            new_edges.append(
                DataEdge(
                    src=real_src,
                    dst=edge.dst,
                    tensor_shape=shape_by_src.get(real_src, edge.tensor_shape),
                )
            )

    return OperatorGraph(
        graph_id=graph.graph_id,
        nodes=[n for n in graph.nodes if n.op_type != "Unknown"],
        edges=new_edges,
    )


__all__ = [
    "TorchFxFrontendUnavailable",
    "TorchFxTraceConfig",
    "trace_module_to_graph",
    "graph_executed_subset",
]

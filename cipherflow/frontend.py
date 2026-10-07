"""PyTorch front end: trace a Transformer encoder stack into CipherFlow's operator graph.

    graph = cf.models.from_torch(encoder, torch.randn(1, 128, 768))

The module's forward should take the hidden states [B, S, H] (embeddings and
task heads are outside the v0.1 operator set). The front end traces it with
torch.fx, propagates shapes, and maps it onto the canonical operators the same
way the built-in graphs are written:

* self-attention  Q/K/V linear projections of one tensor -> matmul -> softmax ->
                  matmul: Linear_QKV ([B,S,H] -> packed [3,B,S,H]),
                  Attention_QK_MatMul, Softmax, Attention_V_MatMul, as in
                  cf.models.bert/vit (qkv_projection=False leaves Linear_QKV out, as
                  the paper's graphs do); nn.MultiheadAttention (self-attention)
                  expands to the same nodes plus Out_Projection
* linear layers   after attention: Out_Projection; before / after GeLU:
                  FFN_Linear_1 / FFN_Linear_2; other square linears: Out_Projection
* LayerNorm, GeLU, residual Add of two equal-shape tensors
* shape-only ops (view, reshape, transpose, permute, split, getitem, dropout, ...)
  and products / sums with plaintext constants are dropped; data flows through them
* the model input becomes an `Input` node when it feeds several operators
  (so the first residual connection is kept)

Anything else raises UnsupportedModel naming every unsupported operation, so a
model is never planned with parts of it silently missing.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode

from .errors import UnsupportedModel

Shape = Tuple[int, ...]

_SHAPE_METHODS = {"view", "reshape", "transpose", "permute", "contiguous", "flatten", "unsqueeze",
                  "squeeze", "expand", "expand_as", "split", "chunk", "unbind", "type", "to", "float",
                  "half", "clone", "detach", "view_as", "reshape_as", "repeat", "__getitem__"}
_SHAPE_FUNCS = {"getitem", "reshape", "transpose", "permute", "flatten", "split", "chunk", "unbind",
                "squeeze", "unsqueeze", "cat", "stack", "dropout", "contiguous", "clone", "detach",
                "expand", "swapaxes", "movedim"}
_ARITH = {"add": "add", "iadd": "add", "__add__": "add", "__radd__": "add", "sub": "sub", "__sub__": "sub",
          "rsub": "sub", "__rsub__": "sub", "mul": "mul", "imul": "mul", "__mul__": "mul",
          "__rmul__": "mul", "truediv": "div", "div": "div", "__truediv__": "div", "true_divide": "div"}
_MATMUL = {"matmul", "bmm", "__matmul__"}


def _tensor_meta(node):
    return getattr(node, "meta", {}).get("tensor_meta")


def _is_tensor(node) -> bool:
    return hasattr(node, "op") and _tensor_meta(node) is not None


def _shape(node) -> Shape:
    tm = _tensor_meta(node)
    shape = getattr(tm, "shape", None)
    return tuple(int(d) for d in shape) if shape is not None else ()


def _fn_name(target) -> str:
    return getattr(target, "__name__", str(target))


def _qualname(node, gm) -> str:
    if node.op == "call_module":
        return type(gm.get_submodule(node.target)).__name__
    if node.op == "call_method":
        return f"Tensor.{node.target}"
    mod = getattr(node.target, "__module__", "") or ""
    return f"{mod}.{_fn_name(node.target)}".lstrip(".")


class _Traced:
    """Classification of every fx node: kind in {input, const, skip, op, unsupported}."""

    def __init__(self, gm, torch):
        self.gm, self.torch = gm, torch
        self.kind: Dict[str, str] = {}
        self.op: Dict[str, str] = {}              # canonical op (or Linear / MatMul / MHA) for kind == op
        self.why: Dict[str, str] = {}             # unsupported description
        self.nodes = list(gm.graph.nodes)
        self.by_name = {n.name: n for n in self.nodes}
        for n in self.nodes:
            self._classify(n)

    # ------------------------------------------------------------------ classify
    def _data_args(self, n):
        """Tensor-valued inputs that carry (encrypted) data, i.e. not constants."""
        return [a for a in n.all_input_nodes if _is_tensor(a) and self.kind.get(a.name) != "const"]

    def _set(self, n, kind, op=None, why=None):
        self.kind[n.name] = kind
        if op:
            self.op[n.name] = op
        if why:
            self.why[n.name] = why

    def _classify(self, n):
        nn = self.torch.nn
        if n.op == "placeholder":
            return self._set(n, "input" if _is_tensor(n) else "skip")
        if n.op == "get_attr":
            return self._set(n, "const")
        if n.op == "output":
            return self._set(n, "skip")
        if not _is_tensor(n) and not isinstance(_tensor_meta(n), (tuple, list)):
            return self._set(n, "skip")          # sizes, shapes, python scalars
        if n.op == "call_module":
            m = self.gm.get_submodule(n.target)
            cls = type(m).__name__
            if isinstance(m, nn.LayerNorm):
                return self._set(n, "op", "LayerNorm")
            if isinstance(m, nn.GELU) or "gelu" in cls.lower():
                return self._set(n, "op", "GeLU")
            if isinstance(m, nn.Softmax):
                return self._set(n, "op", "Softmax")
            if isinstance(m, nn.Linear):
                return self._set(n, "op", "Linear")
            if isinstance(m, nn.MultiheadAttention):
                return self._set(n, "op", "MHA")
            if isinstance(m, (nn.Dropout, nn.Identity)):
                return self._set(n, "skip")
            return self._set(n, "unsupported", why=cls)
        name = _fn_name(n.target) if n.op == "call_function" else str(n.target)
        low = name.lower()
        if "layer_norm" in low:
            return self._set(n, "op", "LayerNorm")
        if "gelu" in low:
            return self._set(n, "op", "GeLU")
        if "softmax" in low and "log" not in low:
            return self._set(n, "op", "Softmax")
        if low == "linear":
            return self._set(n, "op", "Linear")
        if low in _MATMUL:
            return self._set(n, "op", "MatMul")
        if low in _ARITH:
            data = self._data_args(n)
            if len(data) <= 1:
                return self._set(n, "skip")      # plaintext constant: scaling, bias, position embedding
            if _ARITH[low] == "add" and len(data) == 2 and _shape(data[0]) == _shape(data[1]) == _shape(n):
                return self._set(n, "op", "Residual_Add")
            if _ARITH[low] == "add":
                return self._set(n, "unsupported",
                                 why="broadcast add of two data tensors (e.g. attention mask)")
            return self._set(n, "unsupported",
                             why=f"element-wise {_ARITH[low]} of two data tensors (e.g. gating)")
        if low in _SHAPE_METHODS or low in _SHAPE_FUNCS or name == "getattr":
            return self._set(n, "skip")
        return self._set(n, "unsupported", why=_qualname(n, self.gm))

    # ------------------------------------------------------------------ data flow
    def producers(self, n) -> List[Tuple[Any, Any]]:
        """[(producer op/input node, node through which the data arrives)] for n's data inputs."""
        out, seen = [], set()
        for a in self._data_args(n):
            for p in self._resolve(a):
                if (p.name, a.name) not in seen:
                    seen.add((p.name, a.name))
                    out.append((p, a))
        return out

    def _resolve(self, a) -> List[Any]:
        k = self.kind.get(a.name)
        if k in ("op", "input", "unsupported"):
            return [a]
        if k == "const":
            return []
        res = []
        for b in a.all_input_nodes:
            if _is_tensor(b) or isinstance(_tensor_meta(b), (tuple, list)):
                for p in self._resolve(b):
                    if p not in res:
                        res.append(p)
        return res

    def consumers(self, n) -> List[Any]:
        out = []
        stack = list(n.users)
        seen = set()
        while stack:
            u = stack.pop()
            if u.name in seen:
                continue
            seen.add(u.name)
            if self.kind.get(u.name) in ("op", "unsupported"):
                out.append(u)
            elif self.kind.get(u.name) == "skip" and u.op != "output":
                stack.extend(u.users)
        return out


def _default_concrete_args(module, n_inputs: int) -> Dict[str, Any]:
    try:
        params = list(inspect.signature(module.forward).parameters.values())
    except (TypeError, ValueError):
        return {}
    out = {}
    for p in params[n_inputs:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty:
            out[p.name] = p.default
    if "return_dict" in out:
        out["return_dict"] = False
    return out


def _concrete_value(ph, ca: Dict[str, Any]):
    name = str(ph.target).lstrip("*")
    for key in (name, re.sub(r"_\d+$", "", name)):
        if key in ca:
            return ca[key]
    return ph.args[0] if ph.args else None


@dataclass
class TraceResult:
    """What the executor needs besides the graph: the traced fx module, how to run it, and which
    CipherFlow operator each fx node belongs to (Q/K/V projections belong to Linear_QKV)."""
    graph: OperatorGraph
    gm: Any
    run_args: List[Any]
    kind: Dict[str, str]                 # fx node -> input | const | skip | op | unsupported
    op: Dict[str, str]                   # fx op node -> Linear | MatMul | LayerNorm | ...
    owner: Dict[str, str]                # fx node -> CipherFlow node id ("MHA:<name>" for MultiheadAttention)


def from_torch(module, example_inputs, *, concrete_args: Optional[Dict[str, Any]] = None,
               graph_id: Optional[str] = None, qkv_projection: bool = True) -> OperatorGraph:
    """Trace `module` (an encoder stack taking hidden states) into an OperatorGraph."""
    return trace_torch(module, example_inputs, concrete_args=concrete_args, graph_id=graph_id,
                       qkv_projection=qkv_projection).graph


def trace_torch(module, example_inputs, *, concrete_args: Optional[Dict[str, Any]] = None,
                graph_id: Optional[str] = None, qkv_projection: bool = True) -> TraceResult:
    """from_torch, keeping the traced module and the fx node -> operator mapping."""
    try:
        import torch
        import torch.fx as fx
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedModel("from_torch needs PyTorch (pip install torch)") from exc

    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    ca: Dict[str, Any] = {}
    if isinstance(module, fx.GraphModule):
        gm = module
    else:
        ca = _default_concrete_args(module, len(inputs)) if concrete_args is None else concrete_args
        try:
            gm = fx.symbolic_trace(module, concrete_args=ca or None)
        except Exception as exc:
            raise UnsupportedModel(
                f"torch.fx could not trace {type(module).__name__}: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:200]}. Trace the encoder stack with a forward that takes "
                "the hidden states [B,S,H] and has no data-dependent control flow (for Hugging Face models, "
                "pass the encoder, e.g. model.encoder, with hidden states)") from exc
    # concrete args stay as guarded placeholders; feed them the values they were specialized to
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    args = list(inputs) + [_concrete_value(ph, ca) for ph in placeholders[len(inputs):]]
    try:
        ShapeProp(gm).propagate(*args)
    except Exception as exc:
        raise UnsupportedModel(f"shape propagation failed for {type(module).__name__}: "
                               f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}") from exc

    t = _Traced(gm, torch)
    bad: Dict[str, List[str]] = {}
    for n in t.nodes:
        if t.kind[n.name] == "unsupported":
            bad.setdefault(t.why[n.name], []).append(n.name)

    nodes: Dict[str, OperatorNode] = {}
    order: List[str] = []
    edges: List[DataEdge] = []
    folded = set()
    proj_of: Dict[str, str] = {}        # Q/K/V projection fx node -> its QK^T

    # ---------------------------------------------------------------- attention
    mats = [n for n in t.nodes if t.kind[n.name] == "op" and t.op[n.name] == "MatMul"]
    roles: Dict[str, str] = {}
    qk_inputs: Dict[str, Any] = {}      # qk name -> (X producer, arrival node)
    av_of: Dict[str, str] = {}          # qk name -> its Attn*V
    for qk in mats:
        sms = [c for c in t.consumers(qk) if t.op.get(c.name) == "Softmax"]
        if not sms:
            continue
        sm = sms[0]
        avs = [c for c in t.consumers(sm) if t.op.get(c.name) == "MatMul"]
        if len(avs) != 1:
            bad.setdefault("attention scores not followed by a single probabilities x V matmul", []).append(qk.name)
            continue
        av = avs[0]
        projs = [p for p, _ in t.producers(qk)] + [p for p, _ in t.producers(av) if p is not sm]
        uniq = list(dict.fromkeys(projs))
        srcs = {p.name: t.producers(p) for p in uniq}
        ok = (all(t.op.get(p.name) == "Linear" for p in uniq) and len(uniq) in (1, 3)
              and len({tuple(q.name for q, _ in s) for s in srcs.values()}) == 1
              and all(len(s) == 1 for s in srcs.values())
              and all({c.name for c in t.consumers(p)} <= {qk.name, av.name} for p in uniq))
        if not ok:
            bad.setdefault("attention whose Q/K/V are not linear projections of one tensor "
                           "(cross-attention, rotary embeddings, KV cache, ...)", []).append(qk.name)
            continue
        folded.update(p.name for p in uniq)
        proj_of.update({p.name: qk.name for p in uniq})
        x, arrival = srcs[uniq[0].name][0]
        qk_inputs[qk.name] = (x, arrival)
        roles[qk.name], roles[sm.name], roles[av.name] = "Attention_QK_MatMul", "Softmax", "Attention_V_MatMul"
        av_of[qk.name] = av.name
    for m in mats:
        if m.name not in roles:
            bad.setdefault("matmul outside self-attention", []).append(m.name)

    if bad:
        parts = [f"{what} x{len(ns)} (e.g. {ns[0]})" for what, ns in bad.items()]
        raise UnsupportedModel(
            f"{type(module).__name__} uses {len(bad)} operation kind(s) outside CipherFlow's v0.1 operator "
            f"set: " + "; ".join(parts) + ". Supported: self-attention with linear Q/K/V, linear layers, "
            "LayerNorm, GeLU, Softmax, residual Add, and shape-only ops")

    # ---------------------------------------------------------------- nodes
    def arrival_shape(p, a):
        s = _shape(a)
        return s if s else _shape(p)

    def emit(name, op_type, in_shape, out_shape, attrs=None):
        nodes[name] = OperatorNode(name, op_type, tuple(in_shape), tuple(out_shape),
                                   attributes={"source_fx_node": name, **(attrs or {})})
        order.append(name)

    inputs_used = [n for n in t.nodes if t.kind[n.name] == "input"]
    for n in t.nodes:
        if t.kind[n.name] != "op" or n.name in folded:
            continue
        op = t.op[n.name]
        prods = t.producers(n)
        in_shape = arrival_shape(*prods[0]) if prods else _shape(n)
        if op == "MHA":
            m = gm.get_submodule(n.target)
            data = t._data_args(n)
            if len({a.name for a in data}) != 1:
                raise UnsupportedModel(f"nn.MultiheadAttention {n.name}: only self-attention (query = key = "
                                       "value) is supported")
            s = in_shape if m.batch_first else (in_shape[1], in_shape[0], in_shape[2])
            B, S, E = s
            h = m.num_heads
            att = {"heads": h, "head_dim": E // h}
            if qkv_projection:
                emit(f"{n.name}_qkv", "Linear_QKV", (B, S, E), (3, B, S, E))
                att = dict(att, in_edge_shape=(2, B, S, E))
            emit(f"{n.name}_qk", "Attention_QK_MatMul", (3, B, S, E), (B, h, S, S), att)
            emit(f"{n.name}_sm", "Softmax", (B, h, S, S), (B, h, S, S), {"axis": -1})
            emit(f"{n.name}_av", "Attention_V_MatMul", (B, h, S, S), (B, S, E), att)
            emit(f"{n.name}_op", "Out_Projection", (B, S, E), (B, S, E))
            continue
        if n.name in roles:
            r = roles[n.name]
            if r == "Attention_QK_MatMul":
                x, a = qk_inputs[n.name]
                scores = _shape(n)
                attrs = {"heads": scores[1], "head_dim": arrival_shape(x, a)[-1] // scores[1]}
                if qkv_projection:
                    emit(f"{n.name}_qkv", "Linear_QKV", arrival_shape(x, a), (3,) + arrival_shape(x, a))
                    attrs["in_edge_shape"] = (2,) + arrival_shape(x, a)
                emit(n.name, r, (3,) + arrival_shape(x, a), scores, attrs)
            elif r == "Softmax":
                emit(n.name, r, in_shape, _shape(n), {"axis": -1})
            else:
                raw = _shape(n)
                emit(n.name, r, prods_shape(t, n, roles), raw,
                     {"heads": raw[1], "head_dim": raw[-1]} if len(raw) == 4 else {})
            continue
        if op == "Linear":
            cons = [t.op.get(c.name) for c in t.consumers(n)]
            pops = [t.op.get(p.name) or roles.get(p.name) for p, _ in prods]
            in_f, out_f = in_shape[-1], _shape(n)[-1]
            if "GeLU" in cons:
                op = "FFN_Linear_1"
            elif "GeLU" in pops:
                op = "FFN_Linear_2"
            elif "Attention_V_MatMul" in [roles.get(p.name) for p, _ in prods] or in_f == out_f:
                op = "Out_Projection"
            else:
                raise UnsupportedModel(
                    f"Linear {in_f}->{out_f} at {n.name} is neither an attention projection, an FFN layer "
                    "around GeLU, nor a square linear (task heads are outside the v0.1 operator set)")
        emit(n.name, op, in_shape, _shape(n))

    # ---------------------------------------------------------------- edges
    fan = {}
    for x in inputs_used:
        fan[x.name] = len({c for c in _input_consumers(t, x, folded, qk_inputs, nodes)})
    if len([x for x in inputs_used if fan[x.name] > 0]) > 1:
        raise UnsupportedModel(f"{type(module).__name__} has {len(inputs_used)} data inputs; only a single "
                               "input (the hidden states) is supported")
    input_ids = {}
    for x in inputs_used:
        if fan[x.name] > 1:
            emit("input", "Input", _shape(x), _shape(x))
            order.insert(0, order.pop())
            input_ids[x.name] = "input"

    def src_of(p):
        if t.kind[p.name] == "input":
            return input_ids.get(p.name)
        if p.name in nodes:
            return p.name
        if f"{p.name}_op" in nodes:          # expanded MultiheadAttention
            return f"{p.name}_op"
        return None

    def add_edge(src, dst, shape):
        if src and src != dst and not any(e.src == src and e.dst == dst for e in edges):
            edges.append(DataEdge(src, dst, tuple(shape)))

    for n in t.nodes:
        if t.kind[n.name] != "op" or n.name in folded:
            continue
        if t.op[n.name] == "MHA":
            qk_in = f"{n.name}_qkv" if qkv_projection else f"{n.name}_qk"
            for p, a in t.producers(n):
                add_edge(src_of(p), qk_in, arrival_shape(p, a))
            if qkv_projection:
                B, S, E = nodes[qk_in].input_shape
                add_edge(qk_in, f"{n.name}_qk", (2, B, S, E))
                add_edge(qk_in, f"{n.name}_av", (B, S, E))
            B, h, S, _ = nodes[f"{n.name}_qk"].output_shape
            add_edge(f"{n.name}_qk", f"{n.name}_sm", (B, h, S, S))
            add_edge(f"{n.name}_sm", f"{n.name}_av", (B, h, S, S))
            add_edge(f"{n.name}_av", f"{n.name}_op", nodes[f"{n.name}_av"].output_shape)
            continue
        if roles.get(n.name) == "Attention_QK_MatMul":
            x, a = qk_inputs[n.name]
            if qkv_projection:
                add_edge(src_of(x), f"{n.name}_qkv", arrival_shape(x, a))
                add_edge(f"{n.name}_qkv", n.name, (2,) + arrival_shape(x, a))
                add_edge(f"{n.name}_qkv", av_of[n.name], arrival_shape(x, a))      # V
            else:
                add_edge(src_of(x), n.name, arrival_shape(x, a))
            continue
        for p, a in t.producers(n):
            if p.name in folded:
                continue
            add_edge(src_of(p), n.name, arrival_shape(p, a))

    # a node's logical output is the shape its consumers receive (reshape/transpose are free)
    for name, node in list(nodes.items()):
        outs = {e.tensor_shape for e in edges if e.src == name}
        if len(outs) == 1:
            s = next(iter(outs))
            if s != node.output_shape and _numel(s) == _numel(node.output_shape):
                nodes[name] = OperatorNode(name, node.op_type, node.input_shape, s, attributes=node.attributes)

    gid = graph_id or f"torch_{type(module).__name__}"
    owner: Dict[str, str] = {}
    for nid in order:
        src = nodes[nid].attributes.get("source_fx_node")
        if src in t.by_name and t.op.get(src) == "MHA":
            owner[src] = f"MHA:{src}"
        elif src in t.by_name:
            owner[src] = nid
    for ph, nid in input_ids.items():
        owner[ph] = nid
    for proj, qk in proj_of.items():
        owner[proj] = f"{qk}_qkv" if qkv_projection else qk
    return TraceResult(OperatorGraph(gid, [nodes[i] for i in order], edges), gm, args, dict(t.kind),
                       dict(t.op), owner)


def prods_shape(t, n, roles):
    for p, a in t.producers(n):
        if roles.get(p.name) == "Softmax":
            return _shape(a) or _shape(p)
    return _shape(n)


def _input_consumers(t, x, folded, qk_inputs, nodes):
    out = set()
    for c in t.consumers(x):
        if c.name in folded:
            for qk, (src, _a) in qk_inputs.items():
                if src.name == x.name:
                    out.add(qk)
        else:
            out.add(c.name)
    return out


def _numel(s) -> int:
    k = 1
    for d in s:
        k *= int(d)
    return k

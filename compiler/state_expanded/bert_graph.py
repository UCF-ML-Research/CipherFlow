"""BERT-base operator DAG (G = (V, E)) used by the compiler.

Paper §4.1 illustrates the Transformer encoder block in Fig.1(1). We
materialise that block as an operator graph with real tensor shapes that
respect the backend shape contracts documented in CLAUDE.md:

- LayerNorm@HE is restricted to  1 <= B*S <= 16 on the current NEXUS path.
- FFN_Linear_1@HE expects input [B,S,768] and output [B,S,64]
  (hidden slice per call); inner hidden dim 3072 is a chain of these.
- Attention matmuls@HE are restricted to packed [3,B,S,768] / 12 heads.

We therefore profile and optimise with (B=1, S=16) which is the largest
shape that passes every restricted HE path. A separate call site can
override these via the ``bert_block_graph(batch, seq)`` helper; shapes
that break a contract make the HE vertex infeasible and the compiler
will correctly fall back to MPC for that vertex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode


# Hidden size of BERT-base.
HIDDEN: int = 768
# Inner FFN hidden size of BERT-base.
FFN_HIDDEN: int = 3072
# Attention heads of BERT-base.
HEADS: int = 12
HEAD_DIM: int = HIDDEN // HEADS  # 64


def _shape(*dims: int) -> Tuple[int, ...]:
    return tuple(int(d) for d in dims)


@dataclass(frozen=True)
class BertShapeManifest:
    """Concrete shapes fed to the profiler for one encoder block."""

    batch: int
    seq: int
    hidden: int = HIDDEN
    ffn_hidden: int = FFN_HIDDEN
    heads: int = HEADS

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "seq": self.seq,
            "hidden": self.hidden,
            "ffn_hidden": self.ffn_hidden,
            "heads": self.heads,
            "head_dim": self.head_dim,
        }


def default_manifest() -> BertShapeManifest:
    return BertShapeManifest(batch=1, seq=16)


def full_model_manifest() -> BertShapeManifest:
    """Full BERT-base scope as reported by the measured baselines.

    12 encoder blocks at seq_len=128. Used to place the compiler on the
    same footing as BumbleBee / BOLT / SHAFT / NEXUS, whose published
    end-to-end numbers in `baseline/` are all at this scope.
    """
    return BertShapeManifest(batch=1, seq=128)


def bert_block_graph(manifest: BertShapeManifest | None = None) -> OperatorGraph:
    """Return the operator DAG of one BERT-base encoder block.

    Edges correspond to real data dependencies, including the two
    residual bypass paths ``block_in -> Residual_Add_1`` and
    ``post_attn_ln -> Residual_Add_2`` which create SESE regions the
    §4.2.3 decomposition will exploit.
    """

    m = manifest or default_manifest()
    B, S, H, F, Hd = m.batch, m.seq, m.hidden, m.ffn_hidden, m.head_dim
    Hds = m.heads

    tok = _shape(B, S, H)              # [B, S, 768]
    qkv = _shape(3, B, S, H)           # packed qkv
    scores = _shape(B, Hds, S, S)      # [B, 12, S, S]
    ctx = _shape(B, S, H)              # attention output
    ffn1_out = _shape(B, S, F)         # [B, S, 3072]
    # NOTE: FFN_Linear_2 output shape on the NEXUS slice is [B,S,H] again.

    nodes: List[OperatorNode] = [
        OperatorNode("qk", "Attention_QK_MatMul", qkv, scores,
                     attributes={"heads": Hds, "head_dim": Hd}),
        OperatorNode("sm", "Softmax", scores, scores,
                     attributes={"axis": -1}),
        OperatorNode("av", "Attention_V_MatMul", scores, ctx,
                     attributes={"heads": Hds, "head_dim": Hd}),
        OperatorNode("op", "Out_Projection", ctx, tok),
        OperatorNode("add1", "Residual_Add", tok, tok),
        OperatorNode("ln1", "LayerNorm", tok, tok),
        OperatorNode("ffn1", "FFN_Linear_1", tok, ffn1_out),
        OperatorNode("gelu", "GeLU", ffn1_out, ffn1_out),
        OperatorNode("ffn2", "FFN_Linear_2", ffn1_out, tok),
        OperatorNode("add2", "Residual_Add", tok, tok),
        OperatorNode("ln2", "LayerNorm", tok, tok),
    ]

    edges: List[DataEdge] = [
        DataEdge("qk", "sm", scores),
        DataEdge("sm", "av", scores),
        DataEdge("av", "op", ctx),
        DataEdge("op", "add1", tok),
        # Residual bypass 1: block input (qk) flows as the skip connection.
        DataEdge("qk", "add1", tok),
        DataEdge("add1", "ln1", tok),
        DataEdge("ln1", "ffn1", tok),
        DataEdge("ffn1", "gelu", ffn1_out),
        DataEdge("gelu", "ffn2", ffn1_out),
        DataEdge("ffn2", "add2", tok),
        # Residual bypass 2: post-attention norm flows as the skip connection.
        DataEdge("ln1", "add2", tok),
        DataEdge("add2", "ln2", tok),
    ]

    return OperatorGraph(graph_id=f"bert_block_B{B}_S{S}", nodes=nodes, edges=edges)


def bert_multi_block_graph(
    num_blocks: int, manifest: BertShapeManifest | None = None
) -> OperatorGraph:
    """Chain ``num_blocks`` encoder blocks sharing manifest shapes.

    Node ids are prefixed with ``L<i>_`` for block index i in
    ``[0, num_blocks)``. Inter-block wiring feeds ``L<i>_ln2`` output
    into ``L<i+1>_qk`` input (same shape [B,S,H]).
    """
    m = manifest or default_manifest()
    tok = _shape(m.batch, m.seq, m.hidden)
    qkv = _shape(3, m.batch, m.seq, m.hidden)
    scores = _shape(m.batch, m.heads, m.seq, m.seq)
    ctx = _shape(m.batch, m.seq, m.hidden)
    ffn1_out = _shape(m.batch, m.seq, m.ffn_hidden)

    nodes: List[OperatorNode] = []
    edges: List[DataEdge] = []

    def prefix(i: int, name: str) -> str:
        return f"L{i}_{name}"

    for i in range(num_blocks):
        p = lambda n, _i=i: prefix(_i, n)
        nodes.extend([
            OperatorNode(p("qk"),   "Attention_QK_MatMul", qkv, scores,
                         attributes={"heads": m.heads, "head_dim": m.head_dim}),
            OperatorNode(p("sm"),   "Softmax",             scores, scores,
                         attributes={"axis": -1}),
            OperatorNode(p("av"),   "Attention_V_MatMul",  scores, ctx,
                         attributes={"heads": m.heads, "head_dim": m.head_dim}),
            OperatorNode(p("op"),   "Out_Projection",      ctx, tok),
            OperatorNode(p("add1"), "Residual_Add",        tok, tok),
            OperatorNode(p("ln1"),  "LayerNorm",           tok, tok),
            OperatorNode(p("ffn1"), "FFN_Linear_1",        tok, ffn1_out),
            OperatorNode(p("gelu"), "GeLU",                ffn1_out, ffn1_out),
            OperatorNode(p("ffn2"), "FFN_Linear_2",        ffn1_out, tok),
            OperatorNode(p("add2"), "Residual_Add",        tok, tok),
            OperatorNode(p("ln2"),  "LayerNorm",           tok, tok),
        ])
        edges.extend([
            DataEdge(p("qk"),   p("sm"),   scores),
            DataEdge(p("sm"),   p("av"),   scores),
            DataEdge(p("av"),   p("op"),   ctx),
            DataEdge(p("op"),   p("add1"), tok),
            # Residual 1: attention input bypass.
            DataEdge(p("qk"),   p("add1"), tok),
            DataEdge(p("add1"), p("ln1"),  tok),
            DataEdge(p("ln1"),  p("ffn1"), tok),
            DataEdge(p("ffn1"), p("gelu"), ffn1_out),
            DataEdge(p("gelu"), p("ffn2"), ffn1_out),
            DataEdge(p("ffn2"), p("add2"), tok),
            # Residual 2: post-attention LayerNorm bypass.
            DataEdge(p("ln1"),  p("add2"), tok),
            DataEdge(p("add2"), p("ln2"),  tok),
        ])
        # Inter-block wiring: previous ln2 feeds this qk.
        if i > 0:
            edges.append(DataEdge(prefix(i - 1, "ln2"), p("qk"), tok))

    return OperatorGraph(
        graph_id=f"bert_{num_blocks}block_B{m.batch}_S{m.seq}",
        nodes=nodes,
        edges=edges,
    )


def enumerate_profile_shapes(
    manifest: BertShapeManifest | None = None,
) -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    """Unique (op_type, input_shape, output_shape) tuples in the DAG."""

    g = bert_block_graph(manifest)
    seen = {}
    for node in g.nodes:
        key = (node.op_type, node.input_shape, node.output_shape)
        seen.setdefault(key, key)
    return list(seen.values())


def enumerate_edge_shapes(
    manifest: BertShapeManifest | None = None,
) -> List[Tuple[int, ...]]:
    """Unique tensor shapes that cross edges (needed for conversion profiling)."""
    g = bert_block_graph(manifest)
    return sorted({edge.tensor_shape for edge in g.edges})

"""Built-in reference graphs and the PyTorch front end."""
from __future__ import annotations

from compiler.state_expanded.bert_graph import BertShapeManifest, bert_multi_block_graph


def bert(num_blocks: int = 12, seq: int = 128, batch: int = 1, qkv_projection: bool = True):
    """BERT-base encoder: 12 operators per block (Q/K/V projections, attention, output
    projection, FFN, two residual Adds and LayerNorms), two residual regions per block.

    qkv_projection=False leaves out the Q/K/V projections, as the graph used in the
    NeurIPS 2026 evaluation does (11 operators per block); use it with the paper's
    profiles (paper-cpu / paper-gpu), which have no cost for them."""
    return bert_multi_block_graph(num_blocks, BertShapeManifest(batch=batch, seq=seq), qkv_projection)


def bert_paper(num_blocks: int = 12, seq: int = 128, batch: int = 1):
    """BERT-base as modeled in the NeurIPS 2026 evaluation (no Q/K/V projections)."""
    return bert(num_blocks, seq, batch, qkv_projection=False)


def vit(num_blocks: int = 12, tokens: int = 197, batch: int = 1, hidden: int = 768,
        mlp_hidden: int = 3072, heads: int = 12, qkv_projection: bool = True):
    """ViT-B/16 as the architecture defines it: pre-LN encoder blocks.

    patch embedding (a 768 -> 768 linear over 16x16x3 patches) -> 12 x
    [LN -> Q/K/V projections -> attention -> Add -> LN -> MLP -> Add] -> final LN
    (qkv_projection=False leaves out the projections). The residual of each
    sub-layer starts at the block input, before the LayerNorm. The CLS token,
    position embedding and classification head are not modeled (as for BERT-base).
    """
    from ir.types import DataEdge, OperatorGraph, OperatorNode
    B, S, H, F = batch, tokens, hidden, mlp_hidden
    tok, qkv, sc, f1 = (B, S, H), (3, B, S, H), (B, heads, S, S), (B, S, F)
    att = {"heads": heads, "head_dim": H // heads}
    nodes = [OperatorNode("patch_embed", "Out_Projection", tok, tok)]
    edges, prev = [], "patch_embed"
    for i in range(num_blocks):
        p = lambda n, _i=i: f"L{_i}_{n}"
        att_in = p("qkv") if qkv_projection else p("qk")
        nodes.append(OperatorNode(p("ln1"), "LayerNorm", tok, tok))
        if qkv_projection:
            nodes.append(OperatorNode(p("qkv"), "Linear_QKV", tok, qkv))
            edges.append(DataEdge(p("qkv"), p("qk"), qkv))
        nodes += [OperatorNode(p("qk"), "Attention_QK_MatMul", qkv, sc, attributes=att),
                  OperatorNode(p("sm"), "Softmax", sc, sc, attributes={"axis": -1}),
                  OperatorNode(p("av"), "Attention_V_MatMul", sc, tok, attributes=att),
                  OperatorNode(p("op"), "Out_Projection", tok, tok),
                  OperatorNode(p("add1"), "Residual_Add", tok, tok),
                  OperatorNode(p("ln2"), "LayerNorm", tok, tok),
                  OperatorNode(p("ffn1"), "FFN_Linear_1", tok, f1),
                  OperatorNode(p("gelu"), "GeLU", f1, f1),
                  OperatorNode(p("ffn2"), "FFN_Linear_2", f1, tok),
                  OperatorNode(p("add2"), "Residual_Add", tok, tok)]
        edges += [DataEdge(prev, p("ln1"), tok), DataEdge(p("ln1"), att_in, tok),
                  DataEdge(p("qk"), p("sm"), sc), DataEdge(p("sm"), p("av"), sc),
                  DataEdge(p("av"), p("op"), tok), DataEdge(p("op"), p("add1"), tok),
                  DataEdge(prev, p("add1"), tok),
                  DataEdge(p("add1"), p("ln2"), tok), DataEdge(p("ln2"), p("ffn1"), tok),
                  DataEdge(p("ffn1"), p("gelu"), f1), DataEdge(p("gelu"), p("ffn2"), f1),
                  DataEdge(p("ffn2"), p("add2"), tok), DataEdge(p("add1"), p("add2"), tok)]
        prev = p("add2")
    nodes.append(OperatorNode("final_ln", "LayerNorm", tok, tok))
    edges.append(DataEdge(prev, "final_ln", tok))
    return OperatorGraph(f"vit_b16_preln_{num_blocks}block_B{B}_S{S}" + ("_qkv" if qkv_projection else ""),
                         nodes, edges)


def vit_paper(num_blocks: int = 12, tokens: int = 197, batch: int = 1):
    """ViT-B/16 as modeled in the NeurIPS 2026 evaluation: BERT-style (post-LN)
    encoder blocks at 197 tokens. Use this to reproduce the paper's ViT numbers."""
    return bert_multi_block_graph(num_blocks, BertShapeManifest(batch=batch, seq=tokens))


def from_torch(module, example_inputs, **kwargs):
    """Trace a Transformer encoder stack (forward takes hidden states [B,S,H]) into the
    operator graph. Raises UnsupportedModel naming any operation outside the operator
    set; see cipherflow.frontend for the mapping rules."""
    from .frontend import from_torch as _from_torch
    return _from_torch(module, example_inputs, **kwargs)

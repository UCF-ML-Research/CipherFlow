"""v0.1.1: limits fail explicitly instead of producing a plan for a different model.

1. cost entries scaled from another shape are reported (warning, plan.warnings,
   per-node cost_source) and refused with strict=True
2. method contracts (NEXUS: hidden 768, 12 heads) keep nodes out of HE
3. the PyTorch front end maps models onto the operator set or names what it cannot
4. graph structures the planner does not handle raise UnsupportedGraph
"""
import math
import warnings

import pytest

import cipherflow as cf
from cipherflow import backends
from compiler.state_expanded.bert_graph import BertShapeManifest, bert_multi_block_graph
from compiler.state_expanded.sese import find_sese_regions
from ir.types import DataEdge, OperatorGraph, OperatorNode

DEP = cf.Deployment.of("gpu", "100Mbps/20ms")
PAPER = dict(profile="paper-gpu", security="2pc-dealer")   # the paper's GPU profile (SHAFT: trusted dealer)
TOK = (1, 128, 768)


def _compile(graph, **kw):
    kw = kw if "methods" in kw else {**PAPER, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(graph, DEP, verbose=False, **kw)


# ------------------------------------------------------------------ 1. cost sources

def test_profiled_shapes_are_exact():
    with warnings.catch_warnings():
        warnings.simplefilter("error", cf.CostExtrapolationWarning)
        plan = cf.compile(cf.models.bert_paper(2), DEP, verbose=False, **PAPER)
    assert not any("scaled from the nearest" in w for w in plan.warnings)
    assert {n.cost_source for n in plan.nodes} == {"exact"}


def test_scaled_entries_warn_and_are_marked():
    with pytest.warns(cf.CostExtrapolationWarning, match="scaled from the nearest profiled shape"):
        plan = cf.compile(cf.models.vit(2, qkv_projection=False), DEP, verbose=False, **PAPER)
    assert plan.warnings and "strict=True" in plan.warnings[0]
    assert {n.cost_source for n in plan.nodes} == {"scaled"}
    assert cf.Plan.from_json(plan.to_json()).warnings == plan.warnings


def test_strict_refuses_scaled_entries():
    with pytest.raises(cf.CostExtrapolationError):
        cf.compile(cf.models.vit(2, qkv_projection=False), DEP, strict=True, verbose=False, **PAPER)
    cf.compile(cf.models.bert_paper(2), DEP, strict=True, verbose=False, **PAPER)


# ------------------------------------------------------------------ 2. contracts

def _bert_large(blocks=1):
    return bert_multi_block_graph(blocks, BertShapeManifest(batch=1, seq=128, hidden=1024,
                                                           ffn_hidden=4096, heads=16))


def test_contracts_keep_wide_models_out_of_he():
    plan = _compile(_bert_large(2))
    restricted = {"LayerNorm", "Attention_QK_MatMul", "Attention_V_MatMul", "FFN_Linear_1"}
    assert all(n.domain == "MPC" for n in plan.nodes if n.op_type in restricted)
    assert any("kept out of HE" in w for w in plan.warnings)


def test_contracts_hold_for_bert_base():
    assert backends.violations(cf.models.bert(12), backends.builtin_methods()) == {}
    assert backends.violations(cf.models.vit(12), backends.builtin_methods()) == {}


def test_node_with_no_feasible_domain_is_refused():
    he_only = [m for m in backends.builtin_methods() if m.domain == "HE"]
    with pytest.raises(cf.UnsupportedOperator, match="no domain can run"):
        _compile(_bert_large(1), methods=he_only)


# ------------------------------------------------------------------ 4. structure

def _n(i, op="LayerNorm"):
    return OperatorNode(i, op, TOK, TOK)


def _g(nodes, edges):
    return OperatorGraph("t", [_n(*x) if isinstance(x, tuple) else _n(x) for x in nodes],
                         [DataEdge(a, b, TOK) for a, b in edges])


@pytest.mark.parametrize("graph", [cf.models.bert(12), cf.models.vit(12), cf.models.vit_paper(2)])
def test_builtin_graphs_pass_structure_check(graph):
    cf.check_graph(graph)


def test_nested_regions_are_planned():
    g = _g(["s", "a", "b", ("j1", "Residual_Add"), ("j2", "Residual_Add")],
           [("s", "a"), ("a", "b"), ("b", "j1"), ("a", "j1"), ("j1", "j2"), ("s", "j2")])
    cf.check_graph(g)
    plan = cf.compile(g, DEP, verbose=False, **PAPER)
    assert len(plan.nodes) == 5 and plan.cost["total_ms"] > 0


@pytest.mark.parametrize("nodes,edges,match", [
    (["a", "b", ("c", "Residual_Add")], [("a", "c"), ("b", "c")], "2 inputs"),
    (["a", "b", "c"], [("a", "b"), ("a", "c")], "2 outputs"),
    (["s", "a", "b", "c", ("j", "Residual_Add")], [("s", "a"), ("s", "b"), ("s", "c"), ("a", "j"), ("b", "j"), ("c", "j")],
     "feeds 3 operators"),
    (["s", "a", "b", "x", ("j1", "Residual_Add"), ("j2", "Residual_Add")],      # overlapping, not nested
     [("s", "a"), ("s", "j1"), ("a", "b"), ("a", "x"), ("b", "j1"), ("j1", "j2"), ("x", "j2")],
     "not a chain|overlap"),
    (["a", "b"], [("a", "b"), ("b", "a")], "cycle"),
])
def test_unsupported_structures_raise(nodes, edges, match):
    with pytest.raises(cf.UnsupportedGraph, match=match):
        cf.check_graph(_g(nodes, edges))
    with pytest.raises(cf.UnsupportedGraph):
        cf.compile(_g(nodes, edges), DEP, verbose=False)


# ------------------------------------------------------------------ 3. PyTorch front end

torch = pytest.importorskip("torch")
nn = torch.nn


class Attn(nn.Module):
    def __init__(s, h=768, nh=12):
        super().__init__()
        s.nh = nh
        s.q, s.k, s.v, s.o = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        s.drop = nn.Dropout(0.1)

    def forward(s, x):
        B, S, H = x.shape
        d = H // s.nh

        def heads(t):
            return t.view(B, S, s.nh, d).transpose(1, 2)
        a = s.drop(torch.softmax(torch.matmul(heads(s.q(x)), heads(s.k(x)).transpose(-1, -2)) / math.sqrt(d), dim=-1))
        return s.o(torch.matmul(a, heads(s.v(x))).transpose(1, 2).reshape(B, S, H))


class Mlp(nn.Module):
    def __init__(s, h=768, f=3072):
        super().__init__()
        s.f1, s.act, s.f2 = nn.Linear(h, f), nn.GELU(), nn.Linear(f, h)

    def forward(s, x):
        return s.f2(s.act(s.f1(x)))


class PostLN(nn.Module):
    def __init__(s):
        super().__init__()
        s.attn, s.ln1, s.mlp, s.ln2 = Attn(), nn.LayerNorm(768), Mlp(), nn.LayerNorm(768)

    def forward(s, x):
        x = s.ln1(x + s.attn(x))
        return s.ln2(x + s.mlp(x))


class PreLN(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.attn, s.ln2, s.mlp = nn.LayerNorm(768), Attn(), nn.LayerNorm(768), Mlp()

    def forward(s, x):
        x = x + s.attn(s.ln1(x))
        return x + s.mlp(s.ln2(x))


class Bert(nn.Module):
    def __init__(s, n=2):
        super().__init__()
        s.blocks = nn.ModuleList(PostLN() for _ in range(n))

    def forward(s, x):
        for b in s.blocks:
            x = b(x)
        return x


class ViT(nn.Module):
    def __init__(s, n=2):
        super().__init__()
        s.patch, s.blocks, s.norm = nn.Linear(768, 768), nn.ModuleList(PreLN() for _ in range(n)), nn.LayerNorm(768)

    def forward(s, x):
        x = s.patch(x)
        for b in s.blocks:
            x = b(x)
        return s.norm(x)


class MHABlock(nn.Module):
    def __init__(s):
        super().__init__()
        s.mha, s.ln1, s.mlp, s.ln2 = nn.MultiheadAttention(768, 12, batch_first=True), nn.LayerNorm(768), Mlp(), nn.LayerNorm(768)

    def forward(s, x):
        x = s.ln1(x + s.mha(x, x, x, need_weights=False)[0])
        return s.ln2(x + s.mlp(x))


class Masked(PostLN):
    def forward(s, x, mask):
        return super().forward(x) + mask


def _sig(g):
    idx = {n.node_id: i for i, n in enumerate(g.nodes)}
    return ([(n.op_type, tuple(n.input_shape), tuple(n.output_shape)) for n in g.nodes],
            sorted((idx[e.src], idx[e.dst], tuple(e.tensor_shape)) for e in g.edges))


def test_traced_vit_equals_builtin_vit():
    x = torch.randn(1, 197, 768)
    assert _sig(cf.models.from_torch(ViT(2).eval(), x)) == _sig(cf.models.vit(2, tokens=197))
    g = cf.models.from_torch(ViT(2).eval(), x, qkv_projection=False)        # the paper's profiles
    ref = cf.models.vit(2, tokens=197, qkv_projection=False)
    assert _sig(g) == _sig(ref)
    assert _compile(g).cost["total_ms"] == pytest.approx(_compile(ref).cost["total_ms"])


def test_traced_bert_keeps_both_residuals():
    x = torch.randn(1, 128, 768)
    full = cf.models.from_torch(Bert(2).eval(), x)
    assert sorted(n.op_type for n in full.nodes if n.op_type != "Input") == sorted(n.op_type for n in cf.models.bert(2).nodes if n.op_type != "Input")
    g = cf.models.from_torch(Bert(2).eval(), x, qkv_projection=False)
    assert g.nodes[0].op_type == "Input"
    assert len(find_sese_regions(g)) == 4 and len(find_sese_regions(full)) == 6     # + V's region per block
    ops = sorted(n.op_type for n in g.nodes if n.op_type != "Input")
    assert ops == sorted(n.op_type for n in cf.models.bert_paper(2).nodes)
    plan = _compile(g)
    assert not any("scaled from the nearest" in w for w in plan.warnings) and plan.input_domain() in ("HE", "MPC")
    assert plan.cost["total_ms"] == pytest.approx(_compile(cf.models.bert_paper(2)).cost["total_ms"])


def test_multihead_attention_expands():
    x = torch.randn(1, 128, 768)
    g = cf.models.from_torch(MHABlock().eval(), x)
    assert [n.op_type for n in g.nodes][:6] == ["Input", "Linear_QKV", "Attention_QK_MatMul", "Softmax",
                                                "Attention_V_MatMul", "Out_Projection"]
    _compile(cf.models.from_torch(MHABlock().eval(), x, qkv_projection=False))


@pytest.mark.parametrize("model,inputs,match", [
    (lambda: nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU()), (torch.randn(1, 3, 8, 8),), "Conv2d.*ReLU"),
    (lambda: Masked(), (torch.randn(*TOK), torch.randn(1, 1, 768)), "broadcast add"),
    (lambda: Masked(), (torch.randn(*TOK), torch.randn(*TOK)), "2 data inputs"),
    (lambda: nn.Sequential(nn.Linear(768, 768), nn.SiLU()), (torch.randn(*TOK),), "SiLU"),
    (lambda: nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 2)), (torch.randn(*TOK),), "Linear 768->2"),
])
def test_unsupported_models_name_the_cause(model, inputs, match):
    with pytest.raises(cf.UnsupportedModel, match=match):
        cf.models.from_torch(model().eval(), inputs)


def test_huggingface_bert_encoder():
    transformers = pytest.importorskip("transformers")
    from transformers.models.bert.modeling_bert import BertEncoder
    enc = BertEncoder(transformers.BertConfig(num_hidden_layers=2)).eval()
    x = torch.randn(1, 128, 768)
    full = cf.models.from_torch(enc, x)
    assert sorted(n.op_type for n in full.nodes if n.op_type != "Input") == sorted(n.op_type for n in cf.models.bert(2).nodes if n.op_type != "Input")
    g = cf.models.from_torch(enc, x, qkv_projection=False)
    assert _compile(g).cost["total_ms"] == pytest.approx(_compile(cf.models.bert_paper(2)).cost["total_ms"])

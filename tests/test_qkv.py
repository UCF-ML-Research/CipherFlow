"""Q/K/V projections as an operator (Linear_QKV); the paper's graphs and profiles leave them out."""
import warnings

import pytest

import cipherflow as cf
from cipherflow import backends
from compiler.state_expanded.bert_graph import bert_multi_block_graph, default_manifest
from compiler.state_expanded.sese import find_sese_regions

MEASURED = [n for n, v in cf.profiles.BUILTIN.items() if v["role"] == "measured"]


def _ops(g):
    return [n.op_type for n in g.nodes]


def test_graphs_have_the_projections_and_paper_graphs_do_not():
    g, paper = cf.models.bert(2), cf.models.bert_paper(2)
    assert _ops(g).count("Linear_QKV") == 2 and "Linear_QKV" not in _ops(paper)
    assert _ops(g)[:3] == ["Input", "Linear_QKV", "Attention_QK_MatMul"] and len(g.nodes) == len(paper.nodes) + 3
    qkv = g.nodes[1]
    assert (tuple(qkv.input_shape), tuple(qkv.output_shape)) == ((1, 128, 768), (3, 1, 128, 768))
    assert _ops(paper) == _ops(bert_multi_block_graph(2, default_manifest()))   # the paper's graph is unchanged
    cf.check_graph(g)
    assert len(find_sese_regions(paper)) == 4 and len(find_sese_regions(g)) == 6     # + V's region per block
    vit = cf.models.vit(2)
    assert _ops(vit)[:4] == ["Out_Projection", "LayerNorm", "Linear_QKV", "Attention_QK_MatMul"]
    assert "Linear_QKV" not in _ops(cf.models.vit_paper(2)) + _ops(cf.models.vit(2, qkv_projection=False))


def test_paper_profiles_refuse_the_projections_with_a_hint():
    dep = cf.Deployment.of("gpu", "100Mbps/20ms")
    with pytest.raises(cf.UnsupportedOperator, match="qkv_projection=False"):
        cf.compile(cf.models.bert(2), dep, profile="paper-gpu", security="2pc-dealer", verbose=False)
    cf.compile(cf.models.bert_paper(2), dep, profile="paper-gpu", security="2pc-dealer", verbose=False)


def test_contracts():
    g = cf.models.bert(1)
    assert backends.violations(g, backends.builtin_methods("sci", he="nexus")) == {}
    ofhe = backends.builtin_methods("sci", he="openfhe-fast", level_deltas={"Linear_QKV": 1})
    assert not [k for k in backends.violations(g, ofhe) if k[0].endswith("qkv")]
    short = backends.violations(cf.models.bert(1, seq=64), ofhe)
    assert ("L0_qkv", "HE") in short and "128" in short[("L0_qkv", "HE")]


@pytest.mark.parametrize("name", MEASURED)
def test_measured_profiles_price_the_projections(name):
    prof = cf.load_profile(name)
    at128 = lambda op: {r.domain: r for r in prof.operators if r.op_type == op and r.input_shape[1] == 128}
    qkv, op, ffn1 = at128("Linear_QKV"), at128("Out_Projection"), at128("FFN_Linear_1")
    assert set(qkv) == {"HE", "MPC"}
    for d in ("HE", "MPC"):          # output width 3H: between the output projection (H) and FFN-1 (4H)
        assert op[d].local_compute_ms < qkv[d].local_compute_ms < ffn1[d].local_compute_ms, (d, name)
    assert op["MPC"].comm_bytes < qkv["MPC"].comm_bytes < ffn1["MPC"].comm_bytes
    assert any(tuple(c.tensor_shape) == (3, 1, 128, 768) for c in prof.conversions)


def test_default_plan_includes_the_projections():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        plan = cf.compile(cf.models.bert(2), cf.Deployment.of("cpu", "100Mbps/20ms"), security="2pc-dealer",
                          verbose=False)
        paper_shape = cf.compile(cf.models.bert_paper(2), cf.Deployment.of("cpu", "100Mbps/20ms"),
                                 security="2pc-dealer", verbose=False)
    assert sum(n.op_type == "Linear_QKV" for n in plan.nodes) == 2
    assert plan.cost["total_ms"] > paper_shape.cost["total_ms"]

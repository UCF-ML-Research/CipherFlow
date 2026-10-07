"""V's edge from the Q/K/V projections to Attn*V: nested regions, priced per edge."""
import itertools
import warnings

import pytest

import cipherflow as cf
from cipherflow import backends
from compiler.state_expanded.cost_model import StateExpandedCostModel
from compiler.state_expanded.sese import find_sese_regions

PROFILE = "cpu-openfhe-fast-shaft"
NO_LEVELS = {op: 0 for op in backends.DEFAULT_LEVEL_DELTAS}      # levels never bind: domains alone


def _compile(g, net, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(g, cf.Deployment.of("cpu", net), profile=PROFILE, he="openfhe-fast", mpc="shaft",
                          security="2pc-dealer", verbose=False, **kw)


def brute_force(g, net):
    """Every domain assignment: operator costs plus a conversion on every edge whose ends differ."""
    cm = StateExpandedCostModel(cf.load_profile(PROFILE))
    ns = cf.Deployment.of("cpu", net).network.to_setting()
    ops = [n for n in g.nodes]
    cost = {(n.node_id, d): (0.0 if n.op_type == "Input" else
                             cm.estimate_operator(n.op_type, d, n.input_shape, n.output_shape, ns).latency_ms)
            for n in ops for d in ("HE", "MPC")}
    conv = {(e.src, e.dst, a): cm.estimate_conversion(a, "MPC" if a == "HE" else "HE", e.tensor_shape, ns).latency_ms
            for e in g.edges for a in ("HE", "MPC")}
    best = float("inf")
    for doms in itertools.product(("HE", "MPC"), repeat=len(ops)):
        d = {n.node_id: x for n, x in zip(ops, doms)}
        c = sum(cost[(nid, x)] for nid, x in d.items())
        c += sum(conv[(e.src, e.dst, d[e.src])] for e in g.edges if d[e.src] != d[e.dst])
        best = min(best, c)
    return best


def test_graph_structure():
    g = cf.models.bert(2)
    cf.check_graph(g)
    edges = {(e.src, e.dst): tuple(e.tensor_shape) for e in g.edges}
    assert edges[("L0_qkv", "L0_qk")] == (2, 1, 128, 768) and edges[("L0_qkv", "L0_av")] == (1, 128, 768)
    assert edges[("input", "L0_add1")] == (1, 128, 768) and g.nodes[0].op_type == "Input"
    regions = {(r.fork, r.join): r for r in find_sese_regions(g)}
    assert regions[("L0_qkv", "L0_av")].main_path == ["L0_qk", "L0_sm"]
    assert regions[("input", "L0_add1")].main_path == ["L0_qkv", "L0_qk", "L0_sm", "L0_av", "L0_op"]
    assert ("L0_ln2", "L1_add1") in regions
    cf.check_graph(cf.models.vit(2))
    assert not any(e.dst.endswith("_av") and e.src.endswith("_qkv") for e in cf.models.bert_paper(2).edges)


@pytest.mark.parametrize("net", ["10Mbps/20ms", "100Mbps/1ms", "1Gbps/40ms"])
def test_planner_matches_brute_force_with_v_edge(net):
    g = cf.models.bert(1)
    meths = backends.builtin_methods("shaft", he="openfhe-fast", level_deltas=NO_LEVELS)
    plan = _compile(g, net, methods=meths)
    assert plan.cost["total_ms"] == pytest.approx(brute_force(g, net), rel=1e-9)


def test_v_is_converted_when_attention_is_split():
    """Force QK^T (and its inputs) into MPC and Attn*V into HE: V must be converted to HE."""
    g = cf.models.bert(1)
    split = [m for m in backends.builtin_methods("shaft", he="openfhe-fast", level_deltas=NO_LEVELS)
             if not (m.op_type in ("Linear_QKV", "Attention_QK_MatMul", "Softmax") and m.domain == "HE")
             and not (m.op_type == "Attention_V_MatMul" and m.domain == "MPC")]
    plan = _compile(g, "10Mbps/20ms", methods=split)
    dom = plan.domains()
    assert dom["L0_qkv"] == dom["L0_qk"] == "MPC" and dom["L0_av"] == "HE"
    into_av = [a for a in plan.actions if a.kind == "convert" and a.before == "L0_av"]
    assert sorted(tuple(a.tensor_shape) for a in into_av) == [(1, 12, 128, 128), (1, 128, 768)]   # P and V
    assert plan.cost["total_ms"] == pytest.approx(brute_force_restricted(g, "10Mbps/20ms", dom), rel=1e-9)


def brute_force_restricted(g, net, dom):
    cm = StateExpandedCostModel(cf.load_profile(PROFILE))
    ns = cf.Deployment.of("cpu", net).network.to_setting()
    c = sum(0.0 if n.op_type == "Input" else
            cm.estimate_operator(n.op_type, dom[n.node_id], n.input_shape, n.output_shape, ns).latency_ms
            for n in g.nodes)
    return c + sum(cm.estimate_conversion(dom[e.src], dom[e.dst], e.tensor_shape, ns).latency_ms
                   for e in g.edges if dom[e.src] != dom[e.dst])

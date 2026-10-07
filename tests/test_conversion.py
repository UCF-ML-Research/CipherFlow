"""Hardened HE<->MPC conversion: the planner leaves an HE -> MPC conversion enough levels."""
import math
import warnings
from dataclasses import replace

import pytest

import cipherflow as cf
from cipherflow import backends
from compiler.state_expanded.cost_model import StateExpandedCostModel

GRAPH = cf.models.bert(2)


def _profile(min_levels, softmax_out=None):
    """The SHAFT profile with conversions 20x cheaper than measured (so that plans convert), the
    given minimum level for HE -> MPC and, optionally, the level HE Softmax leaves (measured: 0)."""
    prof = cf.load_profile("cpu-openfhe-fast-shaft")
    if softmax_out is not None:
        prof = replace(prof, operators=[
            replace(r, metadata=dict(r.metadata, level_out=softmax_out)) if r.op_type == "Softmax" and r.domain == "HE"
            else r for r in prof.operators])
    convs = [replace(c, local_compute_ms=c.local_compute_ms / 20, comm_bytes=c.comm_bytes // 20,
                     metadata=dict(c.metadata or {}, protocol="hardened",
                                   **({"min_levels_left": min_levels} if c.from_domain == "HE" else {})))
             for c in prof.conversions]
    return replace(prof, conversions=convs)


def _compile(prof, net, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(GRAPH, cf.Deployment.of("cpu", net), profile=prof, he="openfhe-fast", mpc="shaft",
                          security="2pc-dealer", verbose=False, **kw)


def _levels_at_he_to_mpc(plan):
    """Level of the value at every HE -> MPC conversion, following each edge's actions."""
    lv = {n.id: n.level_out for n in plan.nodes}
    L = plan.profile["he_level_budget"]
    out, cur = [], {}
    for a in plan.actions:
        key = tuple(a.edge)
        lvl = cur.get(key, lv.get(a.edge[0]))
        if a.kind == "bootstrap":
            cur[key] = L
        elif a.src == "HE":
            out.append(lvl)
            cur[key] = None
        else:
            cur[key] = L
    return out


def test_cost_model_reads_the_minimum_level():
    assert StateExpandedCostModel(_profile(2)).he_to_mpc_min_level == 2
    assert StateExpandedCostModel(cf.load_profile("paper-gpu")).he_to_mpc_min_level == 0


@pytest.mark.parametrize("net", ["10Mbps/20ms", "100Mbps/20ms"])
def test_he_to_mpc_conversions_have_the_levels(net):
    plan = _compile(_profile(2), net)
    assert plan.profile["he_to_mpc_min_level"] == 2
    levels = _levels_at_he_to_mpc(plan)
    assert levels and all(lvl is None or lvl >= 2 for lvl in levels)
    from cipherflow.runtime import check_plan
    assert check_plan(plan, GRAPH)["problems"] == []
    assert not any("unhardened" in w for w in plan.warnings)


def _split_attention(prof):
    """QK^T and Softmax only in HE, Attn*V only in MPC."""
    deltas, outs = backends.profile_level_rules(prof)
    return [m for m in backends.builtin_methods("shaft", he="openfhe-fast", level_deltas=deltas, level_out=outs)
            if not (m.op_type in ("Attention_QK_MatMul", "Softmax") and m.domain == "MPC")
            and not (m.op_type == "Attention_V_MatMul" and m.domain == "HE")]


def test_a_conversion_from_a_spent_ciphertext_bootstraps_first():
    """A Softmax that leaves level 1: the scores are bootstrapped (which needs that level) before
    they are converted (which needs 2)."""
    prof = _profile(2, softmax_out=1)
    plan = _compile(prof, "10Mbps/20ms", methods=_split_attention(prof))
    acts = [a for a in plan.actions if a.edge == ["L0_sm", "L0_av"]]
    assert [a.kind for a in acts] == ["bootstrap", "convert"]
    from cipherflow.runtime import check_plan
    assert check_plan(plan, GRAPH)["problems"] == []
    bad = cf.Plan.from_json(plan.to_json())
    bad.actions = [a for a in bad.actions if not (a.kind == "bootstrap" and a.edge == ["L0_sm", "L0_av"])]
    assert any("needs 2" in p for p in check_plan(bad, GRAPH)["problems"])
    bad = cf.Plan.from_json(plan.to_json())
    for n in bad.nodes:
        if n.id == "L0_sm":
            n.level_out = 0
    assert any("bootstrapping needs 1" in p for p in check_plan(bad, GRAPH)["problems"])


def test_a_spent_ciphertext_is_a_dead_end():
    """The measured fast Softmax leaves level 0: its output can be neither bootstrapped (needs 1)
    nor converted (needs 2), so no plan runs Softmax in HE and Attn*V in MPC."""
    prof = _profile(2)
    assert StateExpandedCostModel(prof).bootstrap_min_level == 1
    with pytest.raises(RuntimeError, match="No feasible plan"):
        _compile(prof, "10Mbps/20ms", methods=_split_attention(prof))


def test_unhardened_conversions_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        plan = cf.compile(GRAPH, cf.Deployment.of("cpu", "10Mbps/20ms"), profile="cpu-nexus-shaft",
                          he="nexus", mpc="shaft", security="2pc-dealer", verbose=False)
    assert plan.counts()["conversions"] > 0
    assert any("unhardened" in w for w in plan.warnings)


def test_emulated_conversion_errors():
    torch = pytest.importorskip("torch")
    from cipherflow.runtime import numerics as nm
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(100000, dtype=torch.float64) * 200 - 100
    e = (nm.to_mpc(x, 12, gen) - x).abs().max()
    assert 0.5 * 2 ** -12 < e <= 2 ** -12 + 1e-12                      # within one step, not half
    assert (nm.to_he(x, 2.0e-5, gen) - x).std() == pytest.approx(2.0e-5, rel=0.05)


def test_calibrated_values_above_the_bound_warn():
    ok = {"in_min": -1.0, "in_max": 1.0, "var_min": 1.0, "var_max": 2.0, "lme_min": 5.5, "lme_max": 5.5}
    small = cf.Calibration("m", 1, {n.node_id: dict(ok, out_absmax=3.0) for n in GRAPH.nodes})
    big = cf.Calibration("m", 1, {n.node_id: dict(ok, out_absmax=5000.0) for n in GRAPH.nodes})
    prof = _profile(2)
    prof = replace(prof, conversions=[replace(c, metadata=dict(c.metadata, bound_bits=12)) for c in prof.conversions])
    plans = [_compile(prof, "10Mbps/20ms", calibration=c) for c in (small, big)]
    assert all(p.counts()["conversions"] > 0 for p in plans)
    assert not any("conversion protocol assumes" in w for w in plans[0].warnings)
    assert any("up to 5000" in w and "|x| < 4096" in w for w in plans[1].warnings)


def test_emulation_reports_values_above_the_bound(monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import torch
    from transformers.models.bert.modeling_bert import BertEncoder
    from cipherflow.runtime import emulate
    from cipherflow.runtime import numerics as nm
    torch.manual_seed(0)
    enc = BertEncoder(transformers.BertConfig(num_hidden_layers=1)).eval()
    x = torch.randn(1, 128, 768)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        plan = cf.compile(cf.models.from_torch(enc, x), cf.Deployment.of("cpu", "10Mbps/20ms"),
                          security="2pc-dealer", verbose=False)
    assert plan.counts()["conversions"] > 0
    assert not any("conversion_in_bound" in v for v in emulate(plan, enc, x).out_of_range().values())
    monkeypatch.setattr(nm, "CONVERSION_BOUND_BITS", -30)
    flagged = [k for k, v in emulate(plan, enc, x).out_of_range().items() if v.get("conversion_in_bound") is False]
    assert flagged


def test_static_baselines_name_their_edges_and_keep_the_levels():
    """A static placement (non-linear operators in HE, linear in MPC) bootstraps HE outputs that
    have fewer levels left than the conversion needs, and every action names its edge."""
    from compiler.state_expanded.planner import POLICY_LINEAR_MPC_NONLINEAR_HE, _materialise_static_plan
    from cipherflow.plan import Plan
    from cipherflow.planning import _prepare_profile
    from cipherflow.runtime import check_plan
    deltas, outs = backends.profile_level_rules(_profile(2, softmax_out=1))
    meths = backends.builtin_methods("shaft", he="openfhe-fast", level_deltas=deltas, level_out=outs)
    prof, _ = _prepare_profile(GRAPH, meths, _profile(2, softmax_out=1))
    dep = cf.Deployment.of("cpu", "10Mbps/20ms")
    cp = _materialise_static_plan(GRAPH, prof, dep.network.to_setting(), POLICY_LINEAR_MPC_NONLINEAR_HE,
                                  "linMPC_nlHE", backends.level_deltas(meths))
    plan = Plan.from_compiled(cp, GRAPH, prof, dep)
    assert plan.actions and all(a.edge for a in plan.actions)
    assert {n.domain for n in plan.nodes if n.op_type == "Input"} == {"MPC"}      # its consumers' domain
    pre = [a for a in plan.actions if a.kind == "bootstrap" and a.edge == ["L0_sm", "L0_av"]]
    assert len(pre) == 1                                                          # Softmax leaves level 1
    assert check_plan(plan, GRAPH)["problems"] == []
    deltas, outs = backends.profile_level_rules(_profile(2))                      # measured: level 0
    meths = backends.builtin_methods("shaft", he="openfhe-fast", level_deltas=deltas, level_out=outs)
    prof, _ = _prepare_profile(GRAPH, meths, _profile(2))
    cp = _materialise_static_plan(GRAPH, prof, dep.network.to_setting(), POLICY_LINEAR_MPC_NONLINEAR_HE,
                                  "linMPC_nlHE", backends.level_deltas(meths))
    assert math.isinf(cp.total_cost_ms)

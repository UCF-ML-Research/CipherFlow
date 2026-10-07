"""HE backends: registry, precision filter, per-ciphertext bootstrapping, methods that bootstrap internally."""
import warnings
from dataclasses import replace

import pytest

import cipherflow as cf
from cipherflow import backends
from compiler.state_expanded.cost_model import NetworkSetting, StateExpandedCostModel
from compiler.state_expanded.state_graph import ChainOp, State, outgoing

GRAPH = cf.models.bert(num_blocks=2, seq=128)
CPU = cf.Deployment.of("cpu", "100Mbps/20ms")
GPU = cf.Deployment.of("gpu", "10Mbps/20ms")
NET = NetworkSetting(1e8, 20.0)


def _compile(graph, dep, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(graph, dep, verbose=False, **kw)


def test_registry_states_status_and_precision():
    he = backends.HE_BACKENDS
    assert he["nexus"].status == "reference-scaled" and he["nexus"].precision_bits is None
    assert he["openfhe-fast"].status == he["openfhe-precise"].status == "measured"
    assert he["openfhe-fast"].precision_bits == 10.5 and he["openfhe-precise"].precision_bits == 23.5
    assert (he["openfhe-fast"].levels, he["openfhe-precise"].levels) == (8, 7)
    assert {m.backend for m in backends.builtin_methods("sci") if m.domain == "HE"} == {"nexus"}
    with pytest.raises(ValueError, match="measured profile"):
        backends.builtin_methods("sci", he="openfhe-fast")       # levels come from its profile


def test_bootstrap_is_priced_per_ciphertext_when_the_record_says_so():
    measured = StateExpandedCostModel(cf.load_profile("cpu-nexus-sci"))
    paper = StateExpandedCostModel(cf.load_profile("paper-cpu"))
    tok, ffn = (1, 128, 768), (1, 128, 3072)
    assert measured.bootstrap_count(tok) == 12 and measured.bootstrap_count(ffn) == 48   # 8192 values each
    assert paper.bootstrap_count(tok) == 1                                               # v0.1: per call
    one = measured.estimate_bootstrap(NET).latency_ms
    assert measured.estimate_bootstrap(NET, tok).latency_ms == pytest.approx(12 * one)
    assert paper.estimate_bootstrap(NET, tok).latency_ms == paper.estimate_bootstrap(NET).latency_ms


def test_method_that_bootstraps_internally_sets_the_output_level():
    cm = StateExpandedCostModel(cf.load_profile("paper-gpu"))
    op = ChainOp("ln", "LayerNorm", (1, 128, 768), (1, 128, 768), he_level_delta=3, he_level_out=6)
    he = [e for e in outgoing(State(0, "HE", 4), [op], cm, NET, 20) if e.action == "he_exec"]
    assert he and he[0].dst.l == 6                         # not 4 - 3
    assert not [e for e in outgoing(State(0, "HE", 2), [op], cm, NET, 20) if e.action == "he_exec"]


def test_level_out_reaches_the_plan():
    meths = backends.builtin_methods("shaft", level_out={"LayerNorm": 5},
                                     level_deltas=backends.HE_BACKENDS["nexus"].level_deltas)
    plan = _compile(cf.models.bert_paper(2), GPU, profile="paper-gpu", security="2pc-dealer", methods=meths)
    ln = [n for n in plan.nodes if n.op_type == "LayerNorm" and n.domain == "HE"]
    assert ln and all(n.level_out == 5 for n in ln)


def test_profile_level_rules():
    prof = cf.load_profile("paper-gpu")
    ops = [replace(r, he_level_delta=7, metadata=dict(r.metadata, level_out=9))
           if r.domain == "HE" and r.op_type == "Softmax" else r for r in prof.operators]
    deltas, outs = backends.profile_level_rules(replace(prof, operators=ops))
    assert deltas["Softmax"] == 7 and outs == {"Softmax": 9}


def test_min_precision_selects_the_he_backend():
    plan = _compile(GRAPH, CPU, min_precision=12)
    assert plan.he_backend == "openfhe-precise" and plan.precision_bits == 23.5
    ex = plan.alternatives
    assert "not established" in ex["nexus+sci"]["excluded"] and "10.5 bits" in ex["openfhe-fast+sci"]["excluded"]
    with pytest.raises(cf.UnsupportedBackend, match="precision"):
        _compile(GRAPH, CPU, min_precision=30)
    assert _compile(GRAPH, CPU, he="openfhe-fast").he_backend == "openfhe-fast"
    with pytest.raises(ValueError):
        _compile(GRAPH, CPU, he="nope")


def test_openfhe_profiles_are_measured_with_level_rules():
    prof = cf.load_profile("cpu-openfhe-fast-shaft")
    assert prof.he_level_budget == 8 and prof.bootstrap.metadata["slots"] == 32768
    deltas, outs = backends.profile_level_rules(prof)
    assert deltas["Softmax"] == 8 and outs["Softmax"] == 0 and outs["LayerNorm"] == 1
    plan = _compile(GRAPH, cf.Deployment.of("cpu", "10Mbps/20ms"), he="openfhe-fast", mpc="shaft",
                    security="2pc-dealer")
    assert plan.he_backend == "openfhe-fast" and not any("reference-scaled" in w for w in plan.warnings)
    for n in plan.nodes:
        if n.domain == "HE" and n.op_type == "LayerNorm":
            assert n.level_out == 1
    with pytest.raises(cf.UnsupportedOperator):     # 64 tokens: outside the OpenFHE kernels, and HE-only
        _compile(cf.models.bert(1, seq=64), CPU, he="openfhe-fast", methods=[
            m for m in backends.builtin_methods(None, he="openfhe-fast", level_deltas={})])


def test_plan_reports_backend_pair_and_he_status():
    plan = _compile(GRAPH, CPU, he="nexus", security="2pc-dealer")
    assert plan.he_backend == "nexus" and plan.precision_bits is None
    assert any("reference-scaled" in w for w in plan.warnings)
    assert set(plan.alternatives) == {"nexus+sci" if plan.mpc_backend == "shaft" else "nexus+shaft"}
    assert set(_compile(GRAPH, CPU, security="2pc-dealer").alternatives) >= {"openfhe-fast+sci", "nexus+shaft"}
    back = cf.Plan.from_json(plan.to_json())
    assert (back.he_backend, back.precision_bits) == (plan.he_backend, plan.precision_bits)


def test_unhardened_conversions_rank_last():
    """NEXUS profiles price the slot-domain mask: with 'auto' backends, a NEXUS plan that converts
    is not chosen over the OpenFHE pairs (hardened conversions), even when it is cheaper."""
    best = _compile(GRAPH, cf.Deployment.of("cpu", "10Mbps/20ms"), security="2pc-dealer")
    assert best.he_backend != "nexus" or best.counts()["conversions"] == 0
    flagged = {n: a for n, a in best.alternatives.items() if a.get("unhardened_conversions")}
    assert flagged and all(n.startswith("nexus+") for n in flagged)
    assert any(a["total_ms"] < best.cost["total_ms"] for a in flagged.values())   # cheaper, yet not chosen


def test_match_lists_he_backend_pairs():
    rep = cf.match(GRAPH, CPU, security="2pc-dealer")
    assert len(rep.compatible()) == 6
    rows = {r.backend: r for r in rep.rows}
    assert "HE costs reference-scaled" in rows["nexus+sci"].detail
    assert "reference-scaled" not in rows["openfhe-precise+sci"].detail
    assert rows["openfhe-precise+sci"].precision_bits == 23.5

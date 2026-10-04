"""Several MPC backends: registry, profile attribution, security filter, selection, match report."""
import warnings
from dataclasses import replace

import pytest

import cipherflow as cf
from cipherflow import backends

GRAPH = cf.models.bert(num_blocks=2, seq=128)
CPU = cf.Deployment.of("cpu", "100Mbps/20ms")
GPU = cf.Deployment.of("gpu", "100Mbps/20ms")


def _compile(graph, dep, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(graph, dep, verbose=False, **kw)


def test_registry():
    assert backends.MPC_BACKENDS["sci"].security == "2pc"
    assert backends.MPC_BACKENDS["shaft"].security == "2pc-dealer"
    assert {m.backend for m in backends.builtin_methods("shaft") if m.domain == "MPC"} == {"shaft"}
    with pytest.raises(ValueError):
        backends.builtin_methods("nope")


@pytest.mark.parametrize("name,backend", [("cpu-nexus-sci", "sci"), ("cpu-nexus-shaft", "shaft"),
                                          ("gpu-nexus-shaft", "shaft"), ("paper-cpu", "shaft"),
                                          ("paper-gpu", "shaft"), ("cpu", "shaft"), ("gpu", "shaft")])
def test_profiles_are_attributed_to_their_backend(name, backend):
    prof = cf.load_profile(name)
    assert backends.infer_backend(prof) == backend and backends.infer_he_backend(prof) == "nexus"
    assert backends.infer_backend(cf.profiles.scale_compute(prof, 2.0)) == backend


def test_default_security_uses_dealer_free_backend():
    plan = _compile(GRAPH, CPU)
    assert plan.mpc_backend == "sci" and plan.security == "2pc"
    assert "security" in plan.alternatives["nexus+shaft"]["excluded"]
    assert {n.backend for n in plan.nodes if n.domain == "MPC"} <= {"sci"}
    assert {n.backend for n in plan.nodes if n.domain == "HE"} <= {plan.he_backend}


def test_auto_picks_the_fastest_allowed_backend():
    plan = _compile(GRAPH, CPU, security="2pc-dealer")
    other = "nexus+sci" if plan.mpc_backend == "shaft" else "nexus+shaft"
    assert plan.cost["total_ms"] <= plan.alternatives[other]["total_ms"]
    for name in ("sci", "shaft"):
        single = _compile(GRAPH, CPU, mpc=name, security="2pc-dealer")
        assert single.mpc_backend == name
        assert single.cost["total_ms"] >= plan.cost["total_ms"]


def test_no_allowed_backend_is_explicit():
    with pytest.raises(cf.UnsupportedBackend, match="security") as e:
        _compile(GRAPH, GPU)
    assert "no built-in 'gpu' profile for nexus+sci" in str(e.value)
    with pytest.raises(cf.UnsupportedBackend, match="security"):
        _compile(GRAPH, CPU, mpc="shaft")
    with pytest.raises(ValueError):
        _compile(GRAPH, CPU, mpc="nope")
    with pytest.raises(ValueError):
        _compile(GRAPH, CPU, security="nope")


def test_profile_and_mpc_must_agree():
    with pytest.raises(cf.UnsupportedBackend, match="holds shaft costs"):
        _compile(GRAPH, CPU, profile="cpu-nexus-shaft", mpc="sci")
    plan = _compile(GRAPH, CPU, profile="cpu-nexus-sci")
    assert plan.mpc_backend == "sci" and plan.profile["name"] == "cpu-nexus-sci"


def test_unknown_backend_needs_security_any():
    prof = cf.load_profile("cpu-nexus-shaft")
    custom = replace(prof, operators=[replace(r, method="mylib_" + r.method) if r.domain == "MPC" else r
                                      for r in prof.operators],
                     metadata={k: v for k, v in prof.metadata.items() if k != "mpc_backend"})
    with pytest.raises(cf.UnsupportedBackend, match="security model is unknown"):
        _compile(GRAPH, CPU, profile=custom)
    plan = _compile(GRAPH, CPU, profile=custom, security="any")
    assert plan.mpc_backend == "custom" and plan.security == "unknown"


def test_strict_skips_backends_that_need_scaled_costs():
    plan = _compile(GRAPH, CPU, security="2pc-dealer", strict=True)    # both measured at 128 tokens
    assert "total_ms" in plan.alternatives["nexus+sci" if plan.mpc_backend == "shaft" else "nexus+shaft"]
    with pytest.raises(cf.UnsupportedBackend, match="CostExtrapolationError"):
        _compile(cf.models.vit(2), CPU, strict=True)                     # 197 tokens: every pair would scale
    with pytest.raises(cf.UnsupportedBackend, match="CostExtrapolationError"):
        _compile(cf.models.vit(2), CPU, security="2pc-dealer", strict=True)


def test_match_report():
    rep = cf.match(GRAPH, CPU, security="2pc-dealer")
    assert {"nexus+sci", "nexus+shaft", "openfhe-fast+sci", "openfhe-precise+shaft"} <= set(rep.compatible())
    rows = {r.backend: r for r in rep.rows}
    assert rows["nexus+sci"].detail.startswith("28/28") and rows["nexus+shaft"].detail.startswith("28/28")
    rep = cf.match(cf.models.vit(2), GPU)
    assert rep.compatible() == [] and "pass security='2pc-dealer'" in str(rep)


def test_plan_json_keeps_backend_fields():
    plan = _compile(GRAPH, CPU, security="2pc-dealer")
    back = cf.Plan.from_json(plan.to_json())
    assert (back.mpc_backend, back.security, back.alternatives) == (plan.mpc_backend, plan.security,
                                                                    plan.alternatives)
    assert [n.backend for n in back.nodes] == [n.backend for n in plan.nodes]

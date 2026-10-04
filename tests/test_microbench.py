"""profile='auto': requirement analysis, cache reuse, incremental measurement, guards.

Real backends are replaced by fast fake measurers; the end-to-end run against
NEXUS / SCI binaries is exercised separately (it takes minutes).
"""
from collections import Counter

import pytest

import cipherflow as cf
from cipherflow import measure as mb
from compiler.state_expanded.profile_schema import BootstrapRecord, ConversionRecord, OperatorRecord


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("CIPHERFLOW_CACHE", str(tmp_path))
    calls = Counter()

    def ops(work, cache, verbose):
        calls[work[0][1]] += len(work)
        return [OperatorRecord(op, d, f"fake_{d}", i, o, local_compute_ms=10.0 if d == "HE" else 1.0,
                               comm_bytes=0 if d == "HE" else 1 << 20, comm_rounds=0 if d == "HE" else 4,
                               he_level_delta=1 if d == "HE" else 0) for (op, d, i, o) in work]

    def conv(shapes, cache, verbose):
        calls["conversion"] += len(shapes)
        return [ConversionRecord(a, b, "fake", s, 0.5, 1 << 20, 1) for s in shapes for a, b in (("HE", "MPC"), ("MPC", "HE"))]

    def boot(cache, verbose):
        calls["bootstrap"] += 1
        return BootstrapRecord("fake", 100.0, 0, 0)

    monkeypatch.setitem(mb.MEASURERS, "cpu", {"nexus+sci": {"HE": ops, "MPC": ops, "conversion": conv,
                                                             "bootstrap": boot}})
    return calls


def test_measures_once_then_hits_cache(fake):
    g = cf.models.bert(num_blocks=2, seq=16)
    p1 = cf.microbench(g, device="cpu", verbose=False)
    first = dict(fake)
    assert first == {"HE": 10, "MPC": 10, "conversion": 4, "bootstrap": 1}  # 10 op shapes x 2 domains
    cf.microbench(g, device="cpu", verbose=False)
    assert dict(fake) == first                                           # nothing re-measured
    assert len(p1.operators) == 20


def test_new_graph_measures_only_new_shapes(fake):
    cf.microbench(cf.models.bert(num_blocks=1, seq=16), device="cpu", verbose=False)
    before = dict(fake)
    cf.microbench(cf.models.bert(num_blocks=1, seq=32), device="cpu", verbose=False)
    assert fake["bootstrap"] == before["bootstrap"]                      # shape-independent: reused
    assert fake["HE"] - before["HE"] == 10 and fake["conversion"] - before["conversion"] == 4
    todo = cf.microbench(cf.models.bert(num_blocks=4, seq=32), device="cpu", dry_run=True, verbose=False)
    assert not todo.ops and not todo.conversion_shapes and not todo.bootstrap   # deeper model: all cached


def test_compile_auto_on_local_deployment(fake):
    g = cf.models.bert(num_blocks=2, seq=16)
    plan = cf.compile(g, cf.Deployment.local("100Mbps/20ms", device="cpu"), verbose=False)
    assert len(plan.nodes) == len(g.nodes) and plan.cost["total_ms"] > 0
    assert {n.cost_source for n in plan.nodes} == {"exact"}                            # measured, not scaled
    assert not any("scaled from the nearest" in w for w in plan.warnings)


def test_auto_refuses_a_remote_target(fake):
    with pytest.raises(ValueError, match="measures this machine"):
        cf.compile(cf.models.bert(1, seq=16), cf.Deployment.of("gpu", "1Gbps/1ms"), profile="auto")


def test_unwired_device_is_explicit(fake):
    with pytest.raises(cf.BackendUnavailable):
        cf.microbench(cf.models.bert(1, seq=16), device="cuda", verbose=False)


def test_cache_is_per_backend(fake):
    g = cf.models.bert(num_blocks=1, seq=16)
    p = cf.microbench(g, device="cpu", verbose=False)
    assert (mb.cache_dir("cpu", "sci") / "profile.json").exists()
    assert p.metadata["mpc_backend"] == "sci" and cf.backends.infer_backend(p) == "sci"


def test_v01_cache_layout_is_still_read(fake):
    g = cf.models.bert(num_blocks=1, seq=16)
    cf.microbench(g, device="cpu", verbose=False)
    new = mb.cache_dir("cpu", "sci") / "profile.json"
    new.rename(mb.machine_dir("cpu") / "profile.json")                 # v0.1 location
    assert mb.load_cached("cpu", "sci") is not None
    assert not cf.microbench(g, device="cpu", dry_run=True, verbose=False).ops


def test_auto_with_an_unwired_backend_is_explicit(fake):
    dep = cf.Deployment.local("100Mbps/20ms", device="cpu")
    with pytest.raises(cf.UnsupportedBackend, match=r"not wired for nexus\+shaft"):
        cf.compile(cf.models.bert(1, seq=16), dep, mpc="shaft", security="2pc-dealer", verbose=False)
    plan = cf.compile(cf.models.bert(1, seq=16), dep, security="2pc-dealer", verbose=False)
    assert plan.mpc_backend == "sci" and "not wired" in plan.alternatives["nexus+shaft"]["excluded"]

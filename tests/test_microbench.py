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

    monkeypatch.setitem(mb.MEASURERS, "cpu", {"HE": ops, "MPC": ops, "conversion": conv, "bootstrap": boot})
    return calls


def test_measures_once_then_hits_cache(fake):
    g = cf.models.bert(num_blocks=2, seq=16)
    p1 = cf.microbench(g, device="cpu", verbose=False)
    first = dict(fake)
    assert first == {"HE": 9, "MPC": 9, "conversion": 3, "bootstrap": 1}   # 9 op shapes x 2 domains
    cf.microbench(g, device="cpu", verbose=False)
    assert dict(fake) == first                                           # nothing re-measured
    assert len(p1.operators) == 18


def test_new_graph_measures_only_new_shapes(fake):
    cf.microbench(cf.models.bert(num_blocks=1, seq=16), device="cpu", verbose=False)
    before = dict(fake)
    cf.microbench(cf.models.bert(num_blocks=1, seq=32), device="cpu", verbose=False)
    assert fake["bootstrap"] == before["bootstrap"]                      # shape-independent: reused
    assert fake["HE"] - before["HE"] == 9 and fake["conversion"] - before["conversion"] == 3
    todo = cf.microbench(cf.models.bert(num_blocks=4, seq=32), device="cpu", dry_run=True, verbose=False)
    assert not todo.ops and not todo.conversion_shapes and not todo.bootstrap   # deeper model: all cached


def test_compile_auto_on_local_deployment(fake):
    g = cf.models.bert(num_blocks=2, seq=16)
    plan = cf.compile(g, cf.Deployment.local("100Mbps/20ms", device="cpu"), verbose=False)
    assert len(plan.nodes) == len(g.nodes) and plan.cost["total_ms"] > 0
    assert plan.warnings == [] and {n.cost_source for n in plan.nodes} == {"exact"}   # measured, not scaled


def test_auto_refuses_a_remote_target(fake):
    with pytest.raises(ValueError, match="measures this machine"):
        cf.compile(cf.models.bert(1, seq=16), cf.Deployment.of("gpu", "1Gbps/1ms"), profile="auto")


def test_unwired_device_is_explicit(fake):
    with pytest.raises(cf.BackendUnavailable):
        cf.microbench(cf.models.bert(1, seq=16), device="cuda", verbose=False)

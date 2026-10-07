"""Kernel variants (wide-range GeLU) chosen per node from a calibration, and bootstrapping's
level and range requirements (native/openfhe/NONLINEAR.md)."""
import json
import re
import warnings
from pathlib import Path

import pytest

import cipherflow as cf
from cipherflow import backends
from cipherflow.kernel_params import bootstrap_scale, gelu_wide_bound, gelu_wide_coefficients

ROOT = Path(__file__).resolve().parents[1]
GRAPH = cf.models.bert(2)
GELU = [n.node_id for n in GRAPH.nodes if n.op_type == "GeLU"]          # L0_gelu, L1_gelu
SLOW = cf.Deployment.of("cpu", "10Mbps/20ms")


def _calibration(gelu_ranges, absmax=3.0):
    ok = {"in_min": -1.0, "in_max": 1.0, "var_min": 1.0, "var_max": 2.0, "lme_min": 5.5, "lme_max": 5.5,
          "out_absmax": absmax}
    nodes = {n.node_id: dict(ok) for n in GRAPH.nodes}
    for nid, (lo, hi) in zip(GELU, gelu_ranges):
        nodes[nid].update(in_min=lo, in_max=hi)
    return cf.Calibration("m", 1, nodes)


def _compile(cal=None, profile="cpu-openfhe-fast-sci", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(GRAPH, SLOW, profile=profile, he="openfhe-fast", mpc="sci", verbose=False,
                          calibration=cal, **kw)


def test_bounds_and_scales():
    assert gelu_wide_bound({"in_min": -20.0, "in_max": 30.0}) == 64          # 1.25 * 30 = 37.5
    assert gelu_wide_bound({"in_min": -61.0, "in_max": 135.0}) == 256
    assert gelu_wide_bound({"in_min": -300.0, "in_max": 1.0}) is None
    assert bootstrap_scale(15.0) == 1 and bootstrap_scale(30.0) == 2 and bootstrap_scale(1060.0) == 128


def test_kernels_are_chosen_per_node():
    deltas, outs = backends.profile_level_rules(cf.load_profile("cpu-openfhe-fast-sci"))
    meths = backends.builtin_methods("sci", he="openfhe-fast", level_deltas=deltas, level_out=outs)
    g, kernels = backends.assign_kernels(GRAPH, meths, _calibration([(-4.0, 5.0), (-20.0, 30.0)]))
    assert kernels == {"L1_gelu": {"kernel": "wide", "bound": 64}}            # L0 fits the base kernel
    assert {n.node_id: n.op_type for n in g.nodes}["L1_gelu"] == "GeLU.wide"
    assert backends.assign_kernels(GRAPH, meths, None) == (GRAPH, {})        # variants need a calibration


def test_plans_record_the_kernel():
    plan = _compile(_calibration([(-20.0, 30.0), (-61.0, 135.0)]))
    gelu = {n.id: n for n in plan.nodes if n.op_type == "GeLU"}
    assert set(gelu) == set(GELU)                                            # op types as in the model
    he = [n for n in gelu.values() if n.domain == "HE"]
    assert he and all(n.kernel["kernel"] == "wide" for n in he)
    assert all(n.kernel["bound"] == {"L0_gelu": 64, "L1_gelu": 256}[n.id] for n in he)
    assert any("wide-range HE kernel" in w for w in plan.warnings)
    back = cf.Plan.from_json(plan.to_json())
    assert [n.kernel for n in back.nodes] == [n.kernel for n in plan.nodes]
    out = _compile(_calibration([(-300.0, 1.0), (-300.0, 1.0)]))            # no kernel covers it
    assert all(n.domain == "MPC" for n in out.nodes if n.op_type == "GeLU")
    assert any("kept out of HE" in w and "wide HE GeLU" in w for w in out.warnings)


def test_header_matches_the_coefficients():
    text = (ROOT / "native" / "openfhe" / "gelu_wide_coeffs.h").read_text()
    for b in (32, 64, 128, 256):
        m = re.search(r"\{" + str(b) + r", \{\{([^}]*)\}, \{([^}]*)\}", text)
        f1, h = ([float(t) for t in grp.split(",")] for grp in m.groups())
        c = gelu_wide_coefficients(b)
        assert f1[0] == pytest.approx(2 * c["f1"][0], abs=1e-15) and f1[1:] == pytest.approx(c["f1"][1:], abs=1e-15)
        assert h[0] == pytest.approx(2 * c["h"][0]) and h[1:] == pytest.approx(c["h"][1:])


def test_emulated_wide_gelu():
    torch = pytest.importorskip("torch")
    from cipherflow.runtime import HE_NUMERICS
    from cipherflow.runtime import numerics as nm
    gen = torch.Generator().manual_seed(0)
    for b in (32, 64, 128, 256):
        x = torch.linspace(-b, b, 200001, dtype=torch.float64)
        st = {}
        precise = (nm.he_gelu_wide(x, b, HE_NUMERICS["openfhe-precise"], gen, st) - nm.gelu_exact(x)).abs().max()
        assert float(precise) <= gelu_wide_coefficients(b)["max_err"] * 1.01 + 1e-6
        fast = (nm.he_gelu_wide(x, b, HE_NUMERICS["openfhe-fast"], gen, {}) - nm.gelu_exact(x)).abs().max()
        assert float(fast) < 2e-3                    # he_ops measured 2.9e-4 - 4.9e-4 (fast)
        assert st["outside"] == 0 and st["bootstrap_in_range"]


def test_bootstrap_noise_model():
    torch = pytest.importorskip("torch")
    from cipherflow.runtime import HE_NUMERICS
    from cipherflow.runtime import numerics as nm
    fast = HE_NUMERICS["openfhe-fast"]
    gen = torch.Generator().manual_seed(0)

    def err(amp, s=1.0, mean=0.0):
        x = (torch.rand(100000, generator=gen, dtype=torch.float64) * 2 - 1) * amp + mean
        return float((nm.bootstrap(x, fast, gen, s)[0] - x).std())
    assert err(1.0) == pytest.approx(err(30.0), rel=0.05)                    # flat up to 32
    assert err(100.0) > 20 * err(1.0)                                        # ~cubic beyond
    assert err(100.0, 4.0) == pytest.approx(4 * err(1.0), rel=0.05)         # scaled: s times the base noise
    assert err(1.0, mean=3.0) > 10 * err(1.0)                                # a mean of 3 counts like ~90
    assert not nm.bootstrap(torch.tensor([100.0, -100.0]), fast, gen)[1]
    assert nm.bootstrap(torch.tensor([100.0, -100.0]), fast, gen, 4.0)[1]


def test_scaled_bootstraps_need_a_level():
    plan = _compile(_calibration([(-4.0, 5.0), (-4.0, 5.0)], absmax=300.0))
    assert plan.profile["bootstrap_min_level"] == 1
    assert set(plan.profile["scaled_bootstrap"].values()) == {32}            # 2 * 300 / 32 -> 32
    from cipherflow.runtime import check_plan
    assert check_plan(plan, GRAPH)["problems"] == []
    boots = [a for a in plan.actions if a.kind == "bootstrap" and a.edge and a.level_in is not None]
    assert all(a.level_in >= 2 for a in boots if a.edge[0] in plan.profile["scaled_bootstrap"])


# ------------------------------------------------------------ wide-range Softmax

def test_softmax_wide_design():
    from cipherflow.kernel_params import softmax_wide_contract, softmax_wide_design
    d = softmax_wide_design({"in_min": -10.1, "in_max": 11.2, "rowmax_min": 0.0})
    assert d["k"] == 3 and d["c"] == pytest.approx(11.2 + 0.2 * 21.3) and softmax_wide_contract(None, {
        "in_min": -10.1, "in_max": 11.2, "rowmax_min": 0.0}) is None
    assert softmax_wide_design({"in_min": -28.2, "in_max": 28.6, "rowmax_min": 0.0})["k"] == 4
    assert softmax_wide_design({"in_min": -10.1, "in_max": 11.2, "lme_min": -2.0})["k"] == 3    # older calibrations
    assert "rounds" in softmax_wide_contract(None, {"in_min": -200.0, "in_max": 200.0, "rowmax_min": 0.0})
    assert softmax_wide_design({"in_min": -1.0}) is None


def test_softmax_header_matches_the_coefficients():
    from cipherflow.kernel_params import softmax_wide_constants
    k = softmax_wide_constants()
    text = (ROOT / "native" / "openfhe" / "softmax_wide_coeffs.h").read_text()
    for name, part in (("SMW_P1", "first"), ("SMW_P2", "rounds")):
        m = re.search(name + r" = \{([^}]*)\}", text)
        c = [float(t) for t in m.group(1).split(",")]
        assert c[0] == pytest.approx(2 * k[part]["coef"][0]) and c[1:] == pytest.approx(k[part]["coef"][1:])
    assert f"SMW_ZETA = {k['rounds']['zeta']!r}" in text


def test_emulated_wide_softmax():
    torch = pytest.importorskip("torch")
    from cipherflow.kernel_params import softmax_wide_design
    from cipherflow.runtime import HE_NUMERICS
    from cipherflow.runtime import numerics as nm
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(12, 128, 128, generator=gen, dtype=torch.float64) * 4 + \
        torch.rand(12, 128, 1, generator=gen, dtype=torch.float64) * 8          # rows with different maxima
    d = softmax_wide_design({"in_min": float(x.min()), "in_max": float(x.max()),
                             "rowmax_min": float(x.max(-1).values.min())})
    ref = x.softmax(-1)
    st = {}
    precise = (nm.he_softmax_wide(x, d, HE_NUMERICS["openfhe-precise"], gen, st) - ref).abs().max()
    fast = (nm.he_softmax_wide(x, d, HE_NUMERICS["openfhe-fast"], gen, {}) - ref).abs().max()
    assert float(precise) < 1e-4 and float(fast) < 1e-2
    assert st["sums_outside"] == 0 and st["outside"] == 0 and st["rounds"] == d["k"]


def test_softmax_kernel_is_chosen_per_node():
    deltas, outs = backends.profile_level_rules(cf.load_profile("cpu-openfhe-fast-sci"))
    meths = backends.builtin_methods("sci", he="openfhe-fast", level_deltas=deltas, level_out=outs)
    cal = _calibration([(-4.0, 5.0), (-4.0, 5.0)])
    sm = [n.node_id for n in GRAPH.nodes if n.op_type == "Softmax"]
    cal.nodes[sm[0]].update(in_min=-10.1, in_max=11.2, rowmax_min=0.0, lme_min=-1.0)        # beyond [-6, 6]
    cal.nodes[sm[1]].update(in_min=-300.0, in_max=300.0, rowmax_min=0.0, lme_min=-1.0)      # beyond 4 rounds
    g, kernels = backends.assign_kernels(GRAPH, meths, cal)
    assert kernels[sm[0]]["kernel"] == "wide" and kernels[sm[0]]["k"] == 3
    assert sm[1] not in kernels
    assert {n.node_id: n.op_type for n in g.nodes}[sm[0]] == "Softmax.wide"


# ------------------------------------------------------------ wide-range LayerNorm

def test_layernorm_wide_design():
    from cipherflow.kernel_params import layernorm_wide_design
    d = layernorm_wide_design({"var_min": 0.024, "var_max": 2.1}, 1e-3)             # attention LN
    assert d["e0"] == 3e-3 and d["steps"] == 2 and d["vhi"] == pytest.approx(2.1 * 2.5)
    assert layernorm_wide_design({"var_min": 0.7, "var_max": 1559.0}, 1e-3) is None   # layers 9-10 after the FFN
    assert layernorm_wide_design({"var_min": 0.7, "var_max": 1559.0}, 1e-4)["steps"] == 6
    assert layernorm_wide_design({"var_min": 1e-3, "var_max": 1e3}, 1e-4) is None
    assert layernorm_wide_design({}, 1e-4) is None


def test_layernorm_header_matches_the_coefficients():
    from cipherflow.kernel_params import layernorm_wide_constants
    fits = layernorm_wide_constants()["fits"]
    text = (ROOT / "native" / "openfhe" / "layernorm_wide_coeffs.h").read_text()
    rows = re.findall(r"\{([0-9.e-]+), ([0-9.e-]+), (\d+), \{([^}]*)\}, ([0-9.e-]+)\}", text)
    assert len(rows) == len(fits)
    for e0, s, steps, coef, _ in rows:
        f = fits[repr(float(e0))]
        c = [float(t) for t in coef.split(",")]
        assert float(s) == pytest.approx(f["s"]) and int(steps) == f["steps"]
        assert c[0] == pytest.approx(2 * f["coef"][0]) and c[1:] == pytest.approx(f["coef"][1:])


def test_emulated_wide_layernorm():
    torch = pytest.importorskip("torch")
    from cipherflow.kernel_params import layernorm_wide_design
    from cipherflow.runtime import HE_NUMERICS
    from cipherflow.runtime import numerics as nm
    gen = torch.Generator().manual_seed(0)
    for (lo, hi), min_e0 in (((0.024, 2.1), 1e-3), ((0.7, 1559.0), 1e-4)):
        sd = torch.sqrt(lo * (hi / lo) ** torch.rand(128, 1, generator=gen, dtype=torch.float64))
        x = torch.randn(128, 768, generator=gen, dtype=torch.float64) * sd + 3
        ref = torch.nn.functional.layer_norm(x, (768,), eps=1e-12)
        d = layernorm_wide_design({"var_min": lo, "var_max": hi}, min_e0)
        st = {}
        precise = (nm.he_layernorm_wide(x, None, None, d, HE_NUMERICS["openfhe-precise"], gen, st) - ref).abs().max()
        fast = (nm.he_layernorm_wide(x, None, None, d, HE_NUMERICS["openfhe-fast"], gen, {}) - ref).abs().max()
        assert float(precise) < 1e-5 and float(fast) < 1e-2 and st["outside"] == 0


def test_layernorm_kernels_are_chosen_per_node():
    deltas, outs = backends.profile_level_rules(cf.load_profile("cpu-openfhe-fast-sci"))
    meths = backends.builtin_methods("sci", he="openfhe-fast", level_deltas=deltas, level_out=outs)
    cal = _calibration([(-4.0, 5.0), (-4.0, 5.0)])
    ln = [n.node_id for n in GRAPH.nodes if n.op_type == "LayerNorm"]
    cal.nodes[ln[0]].update(var_min=1.0, var_max=2.0)            # the base kernel's [0.25, 25]
    cal.nodes[ln[1]].update(var_min=0.024, var_max=2.1)          # wide
    cal.nodes[ln[2]].update(var_min=0.7, var_max=1559.0)         # wider
    cal.nodes[ln[3]].update(var_min=1e-4, var_max=1e3)           # none
    _, kernels = backends.assign_kernels(GRAPH, meths, cal)
    assert ln[0] not in kernels and ln[3] not in kernels
    assert kernels[ln[1]]["kernel"] == "wide" and kernels[ln[2]]["kernel"] == "wider"

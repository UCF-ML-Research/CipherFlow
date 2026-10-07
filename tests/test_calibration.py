"""Calibration and the HE methods' value contracts."""
import math
import warnings

import pytest

import cipherflow as cf
from cipherflow import backends
from cipherflow.kernel_params import HE_NUMERICS, value_contracts

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers.models.bert.modeling_bert import BertEncoder   # noqa: E402

from cipherflow import runtime                                   # noqa: E402

CPU = cf.Deployment.of("cpu", "1Gbps/40ms")
SLOW = cf.Deployment.of("cpu", "10Mbps/20ms")      # plans with HE non-linear operators


def _compile(g, dep=CPU, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(g, dep, security="2pc", verbose=False, **kw)


def test_value_contracts():
    c = value_contracts(HE_NUMERICS["openfhe-fast"])
    assert c["GeLU"](None, {"in_min": -5.0, "in_max": 5.0}) is None
    assert c["GeLU"](None, {"in_min": -7.9, "in_max": 7.9}) is not None          # no room for the margin
    assert "[-8, 8]" in c["GeLU"](None, {"in_min": -61.0, "in_max": 129.0})
    assert c["LayerNorm"](None, {"var_min": 0.4, "var_max": 9.0}) is None
    assert c["LayerNorm"](None, {"var_min": 0.3, "var_max": 20.0}) is not None
    assert "variances" in c["LayerNorm"](None, {"var_min": 0.03, "var_max": 1442.0})
    ok = {"in_min": -2.0, "in_max": 2.0, "lme_min": math.log(0.5) + 6}        # u = 0.5
    assert c["Softmax"](None, ok) is None
    assert "row means" in c["Softmax"](None, dict(ok, lme_min=0.0))            # u = exp(-6) < 0.02
    assert "scores" in c["Softmax"](None, dict(ok, in_max=30.0))
    assert "not in the calibration" in c["GeLU"](None, {})


def test_calibration_round_trip_and_merge(tmp_path):
    a = cf.Calibration("m", 2, {"n": {"in_min": -1.0, "in_max": 2.0}})
    b = cf.Calibration("m", 3, {"n": {"in_min": -3.0, "in_max": 1.0}, "k": {"var_min": 0.5}})
    m = a.merge(b)
    assert m.samples == 5 and m.nodes["n"] == {"in_min": -3.0, "in_max": 2.0} and "k" in m.nodes
    a.to_json(tmp_path / "c.json")
    assert cf.Calibration.load(tmp_path / "c.json") == a


def test_calibrated_plans_keep_out_of_range_operators_out_of_he():
    torch.manual_seed(0)
    enc = BertEncoder(transformers.BertConfig(num_hidden_layers=1)).eval()
    g = cf.models.from_torch(enc, torch.randn(1, 128, 768))
    cal = runtime.calibrate(enc, [torch.randn(1, 32, 768) for _ in range(3)])     # any length: same node ids
    assert cal.samples == 3 and {"var_min", "var_max"} <= set(cal.nodes["layer_0_attention_output_layer_norm"])
    plain = _compile(g, SLOW)
    assert any("calibration=" in w for w in plain.warnings)
    plan = _compile(g, SLOW, calibration=cal)
    # random BERT: the base Softmax kernel's row means are too small (u < umin); only the wide one fits
    sm = [n for n in plan.nodes if n.op_type == "Softmax"]
    assert all(n.domain == "MPC" or (n.kernel or {}).get("kernel") == "wide" for n in sm)
    assert any(("kept out of HE" in w or "wide-range HE kernel" in w) and "Softmax" in w for w in plan.warnings) \
        or all(n.domain == "MPC" for n in sm)
    res = runtime.emulate(plan, enc, torch.randn(1, 128, 768))
    assert res.checks["problems"] == [] and not res.out_of_range()


def test_unknown_ranges_are_not_trusted():
    meths = backends.builtin_methods("shaft", he="nexus")
    gelu = next(m for m in meths if m.op_type == "GeLU" and m.domain == "HE")
    node = cf.models.bert(1).nodes[9]
    cal = cf.Calibration("m", 1, {node.node_id: {"in_min": 0.0, "in_max": 1.0}})
    assert "not known" in gelu.violation(node, cal)
    assert gelu.violation(node) is None                              # without a calibration: as before


def test_glue_scores():
    from cipherflow.runtime.glue import _scores
    assert _scores("acc", [1, 0, 1, 1], [1, 0, 0, 1]) == {"accuracy": 0.75}
    s = _scores("acc_f1", [1, 0, 1, 1], [1, 0, 0, 1])
    assert s["accuracy"] == 0.75 and s["f1"] == pytest.approx(0.8)
    assert _scores("corr", [1.0, 2.0, 3.0], [1.0, 2.5, 3.2])["pearson"] > 0.9

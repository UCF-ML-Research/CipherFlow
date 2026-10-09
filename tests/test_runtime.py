"""cipherflow.runtime: plan checks and emulation on real data."""
import math
import warnings
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers.models.bert.modeling_bert import BertEncoder   # noqa: E402

import cipherflow as cf                                          # noqa: E402
from cipherflow import backends                                  # noqa: E402
from cipherflow.runtime import HE_NUMERICS, check_plan, emulate  # noqa: E402
from cipherflow.runtime import numerics as nm                    # noqa: E402

torch.manual_seed(0)
ENC = BertEncoder(transformers.BertConfig(num_hidden_layers=1)).eval()
X = torch.randn(1, 128, 768)
GRAPH = cf.models.from_torch(ENC, X)


def _plan(net="10Mbps/20ms", security="2pc-dealer", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(GRAPH, cf.Deployment.of("cpu", net), security=security, verbose=False, **kw)


# ------------------------------------------------------------ numerical models vs the measured kernels

def test_he_operators_reproduce_the_measured_kernel_errors():
    """he_ops's inputs; measured max |error| (one CPU, OpenFHE 1.4.2): GeLU 5.6e-4, LayerNorm 1.8-2.2e-2
    (fast) and 5.0-5.3e-3 (precise), Softmax 5.7e-4-9.2e-4 (fast)."""
    torch.manual_seed(1234)
    gen = torch.Generator().manual_seed(1)
    fast, precise = HE_NUMERICS["openfhe-fast"], HE_NUMERICS["openfhe-precise"]
    x = torch.rand(128, 3072, dtype=torch.float64) * 12 - 6
    assert float((nm.he_gelu(x, fast, {}) - nm.gelu_exact(x)).abs().max()) == pytest.approx(5.6e-4, rel=0.02)
    mu = torch.rand(128, 1, dtype=torch.float64) * 4 - 2
    sd = 0.65 + torch.rand(128, 1, dtype=torch.float64) * (3.5 - 0.65)
    xl = mu + sd * torch.randn(128, 768, dtype=torch.float64)
    ref = (xl - xl.mean(-1, keepdim=True)) / torch.sqrt(xl.var(-1, unbiased=False, keepdim=True) + 1e-5)
    assert 1e-2 < float((nm.he_layernorm(xl, None, None, fast, gen, {}) - ref).abs().max()) < 3e-2
    assert 3e-3 < float((nm.he_layernorm(xl, None, None, precise, gen, {}) - ref).abs().max()) < 8e-3
    s = torch.rand(12, 128, 128, dtype=torch.float64) * 12 - 6
    assert 3e-4 < float((nm.he_softmax(s, fast, gen, {}) - s.softmax(-1)).abs().max()) < 2e-3


# ------------------------------------------------------------ plan checks

@pytest.mark.parametrize("net,security", [("10Mbps/20ms", "2pc-dealer"), ("1Gbps/40ms", "2pc"),
                                          ("300Mbps/40ms", "2pc")])
def test_compiled_plans_check_and_run(net, security):
    plan = _plan(net, security)
    res = emulate(plan, ENC, X)
    assert res.checks["problems"] == []
    assert res.applied == {"conversions": plan.counts()["conversions"], "bootstraps": plan.counts()["bootstraps"]}


def test_broken_plans_are_caught():
    plan = _plan("1Gbps/40ms", "2pc")
    plan.actions = [a for a in plan.actions if a.kind != "convert" or a.edge != ["L0_qkv", "L0_av"]] \
        if any(a.edge == ["L0_qkv", "L0_av"] for a in plan.actions) else plan.actions[1:]
    assert check_plan(plan, GRAPH)["problems"]


# ------------------------------------------------------------ emulation

def test_all_mpc_is_fixed_point_only():
    plan = _plan(profile="cpu-openfhe-fast-shaft", he="openfhe-fast", mpc="shaft",
                 methods=[m for m in backends.builtin_methods("shaft", he="openfhe-fast", level_deltas={})
                          if m.domain == "MPC"])
    assert {n.domain for n in plan.operators()} == {"MPC"}
    assert emulate(plan, ENC, X).rel_err < 1e-3


def test_split_attention_converts_v():
    meths = [m for m in backends.builtin_methods("shaft", he="openfhe-fast",
                                                 level_deltas={op: 0 for op in backends.DEFAULT_LEVEL_DELTAS})
             if not (m.op_type in ("Linear_QKV", "Attention_QK_MatMul", "Softmax") and m.domain == "HE")
             and not (m.op_type == "Attention_V_MatMul" and m.domain == "MPC")]
    plan = _plan(profile="cpu-openfhe-fast-shaft", he="openfhe-fast", mpc="shaft", methods=meths)
    v = [a for a in plan.actions if a.kind == "convert" and a.edge and a.edge[1] == "matmul_1"
         and a.edge[0].endswith("_qkv")]
    assert len(v) == 1 and v[0].tensor_shape == [1, 128, 768]
    res = emulate(plan, ENC, X)
    assert res.checks["problems"] == [] and res.applied["conversions"] == plan.counts()["conversions"]


def test_softmax_range_is_reported():
    plan = _plan("10Mbps/20ms", "2pc", he="openfhe-precise")       # precise Softmax leaves a level to go on with
    res = emulate(plan, ENC, X)
    sm = [k for k, v in res.nodes.items() if "u_below_umin" in v]
    assert sm and all(res.nodes[k]["u_below_umin"] > 0.5 for k in sm)      # scores near 0: exp(x - 6) is tiny
    tuned = replace(HE_NUMERICS["openfhe-fast"], sm_bound=2.0)
    assert emulate(plan, ENC, X, he_numerics=tuned).rel_err < 0.5 * res.rel_err

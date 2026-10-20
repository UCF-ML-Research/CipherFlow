"""Backends from their primitives: operators lowered to primitive calls, priced and planned."""
import warnings

import pytest

import cipherflow as cf

torch = pytest.importorskip("torch")

from demo import transformer  # noqa: E402

HE_PRIMS = {"add": cf.Cost(0.3), "mul_plain": cf.Cost(3), "mul": cf.Cost(40), "rotate": cf.Cost(35)}


@pytest.fixture(scope="module")
def block():
    return cf.models.from_torch(transformer.bert(1), torch.zeros(1, 128, 768))


def _compile(graph, exts, net="100Mbps/20ms"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(graph, cf.Deployment.of("cpu", net), he="openfhe-fast", mpc="sci", extensions=exts)


def test_operators_are_lowered_to_primitive_calls(block):
    he = cf.PrimitiveBackend.he("my-ckks", slots=32768, levels=10, bootstrap=cf.Cost(9_000), primitives=HE_PRIMS)
    low = {x.source: x for x in he.lower(block).ops}
    out = low["blocks.0.attn.o"]       # 768 x 768 on 128 tokens, 42 whole tokens per ciphertext: 4 ciphertexts, BSGS
    assert out.kernel.counts == {"mul_plain": 4 * 768, "add": 4 * 768, "rotate": 4 * 2 * 28}
    assert out.kernel.levels == 1 and out.bootstraps == 0
    assert out.cost.local_ms == pytest.approx(4 * 768 * 3 + 4 * 768 * 0.3 + 4 * 56 * 35)
    soft = low["blocks.0.attn: softmax"]               # more levels than the budget: bootstraps inside
    assert soft.kernel.levels > 10 and soft.bootstraps > 0 and soft.level_out is not None
    assert "BSGS" in str(he.lower(block))


def test_a_primitive_backend_plans_in_place_of_the_builtin_one(block):
    he = cf.PrimitiveBackend.he("my-ckks", slots=32768, levels=10, bootstrap=cf.Cost(9_000), primitives=HE_PRIMS)
    plan = _compile(block, [he.extension()])
    assert plan.profile["he_level_budget"] == 10
    assert {n.backend for n in plan.nodes if n.domain == "HE" and n.op_type != "Input"} <= {"my-ckks"}
    assert any("extension my-ckks prices" in w for w in plan.warnings)


def test_a_missing_primitive_keeps_operators_out_of_the_domain(block):
    no_rot = cf.PrimitiveBackend.he("no-rotation", slots=32768, levels=10, bootstrap=cf.Cost(9_000),
                                    primitives={k: v for k, v in HE_PRIMS.items() if k != "rotate"})
    low = {x.op_type: x for x in no_rot.lower(block).ops}
    assert low["Out_Projection"].kernel is None and low["Out_Projection"].missing == ("rotate",)
    assert low["GeLU"].kernel is not None                # a polynomial needs no rotation
    plan = _compile(block, [no_rot.extension()])
    assert all(n.domain == "MPC" for n in plan.nodes if n.op_type in ("Out_Projection", "Softmax", "LayerNorm"))


def test_mpc_primitives_price_bytes_per_element_and_rounds_per_call(block):
    mpc = cf.PrimitiveBackend.mpc("my-2pc", primitives={"mul": cf.Cost(1e-4, 32, 1), "trunc": cf.Cost(0, 8, 1),
                                                         "cmp": cf.Cost(1e-3, 400, 8)})
    low = {x.source: x for x in mpc.lower(block).ops}
    fc1 = low["blocks.0.mlp.fc1"]                       # no matmul primitive: one mul per multiply-accumulate
    assert fc1.kernel.counts["mul"] == 128 * 768 * 3072
    assert fc1.cost.comm_bytes == 128 * 768 * 3072 * 32 + 128 * 3072 * 8 and fc1.cost.comm_rounds == 2
    assert low["blocks.0: add"].cost.comm_bytes == 0
    plan = _compile(block, [mpc.extension()], net="1Gbps/1ms")
    assert {n.backend for n in plan.nodes if n.domain == "MPC" and n.op_type != "Input"} <= {"my-2pc"}


def test_unknown_primitives_are_refused():
    with pytest.raises(ValueError, match="unknown HE primitives"):
        cf.PrimitiveBackend.he("x", slots=8, levels=2, primitives={"ntt": cf.Cost(1)})
    with pytest.raises(ValueError, match="unknown MPC primitives"):
        cf.PrimitiveBackend.mpc("x", primitives={"rotate": cf.Cost(1)})

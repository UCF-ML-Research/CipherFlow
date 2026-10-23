"""Extensions: new primitives for the backends and new operators for the planner."""
import json
import warnings

import pytest

import cipherflow as cf
from cipherflow.extensions import like

torch = pytest.importorskip("torch")

from demo import transformer  # noqa: E402
from demo.extensions import private_rag  # noqa: E402

DEP = cf.Deployment.of("cpu", "10Mbps/1ms")


def _compile(graph, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(graph, kw.pop("dep", DEP), he="openfhe-fast", mpc="sci", **kw)


@pytest.fixture(scope="module")
def bert2():
    return cf.models.from_torch(transformer.bert(2), torch.zeros(1, 128, 768))


def test_a_new_he_primitive_replaces_the_builtin_one(bert2):
    before = _compile(bert2)
    assert {n.domain for n in before.nodes if n.op_type == "Softmax"} == {"MPC"}
    ext = cf.Extension("fast-softmax").primitive("Softmax", "HE", levels=4,
                                                  cost=like("Softmax", "HE", lambda n, r: 0.4))
    after = _compile(bert2, extensions=[ext])
    soft = [n for n in after.nodes if n.op_type == "Softmax"]
    assert {(n.domain, n.backend, n.cost_source) for n in soft} == {("HE", "fast-softmax", "extension")}
    assert after.cost["total_ms"] < before.cost["total_ms"]
    assert any("extension fast-softmax prices Softmax in HE" in w for w in after.warnings)
    assert "Softmax [fast-softmax]" in after.explain_steps()
    back = cf.Plan.from_json(after.to_json())
    assert [n.backend for n in back.nodes] == [n.backend for n in after.nodes]


def test_a_primitive_that_accepts_nothing_keeps_the_operator_out_of_its_domain(bert2):
    ext = cf.Extension("picky").primitive("Softmax", "HE", levels=4, cost=cf.Cost(local_ms=1.0),
                                           accepts=lambda n: "this kernel takes no input")
    plan = _compile(bert2, extensions=[ext])
    assert {n.domain for n in plan.nodes if n.op_type == "Softmax"} == {"MPC"}


def test_new_operators_from_pytorch_modules():
    rag = private_rag.rag_extension()
    model = private_rag.Retriever(n_docs=1024, k=4, blocks=1).eval()
    graph = cf.models.from_torch(model, torch.zeros(1, 128, 768), extensions=[rag])
    tail = [(n.op_type, tuple(n.input_shape), tuple(n.output_shape), n.attributes.get("source")) for n in graph.nodes[-3:]]
    assert tail == [("MeanPool", (1, 128, 768), (1, 768), "pool"), ("DocScores", (1, 768), (1, 1024), "scores"),
                    ("TopK", (1, 1024), (1, 4), "topk")]
    plan = _compile(graph, extensions=[rag])
    dom = {n.op_type: n.domain for n in plan.nodes}
    assert dom["TopK"] == "MPC"                                  # no HE primitive: comparisons
    assert {n.backend for n in plan.nodes if n.op_type in ("MeanPool", "DocScores", "TopK")} == {"private-rag"}
    assert json.loads(plan.to_json())["nodes"][-1]["source"] == "topk"


def test_without_the_extension_new_operators_fail_explicitly():
    model = private_rag.Retriever(n_docs=64, k=2, blocks=1).eval()
    with pytest.raises(cf.UnsupportedModel, match=r"Tensor\.mean.*torch\.topk.*cipherflow\.Extension"):
        cf.models.from_torch(model, torch.zeros(1, 128, 768))
    rag = private_rag.rag_extension()
    graph = cf.models.from_torch(model, torch.zeros(1, 128, 768), extensions=[rag])
    with pytest.raises(cf.UnsupportedOperator, match="TopK"):
        _compile(graph)


def test_extension_declarations_are_checked(bert2):
    with pytest.raises(ValueError, match="'HE' or 'MPC'"):
        cf.Extension("x").primitive("Softmax", "GPU", cost=cf.Cost(1.0))
    with pytest.raises(ValueError, match="no CKKS levels"):
        cf.Extension("x").primitive("Softmax", "MPC", levels=2, cost=cf.Cost(1.0))
    a = cf.Extension("a").primitive("GeLU", "HE", cost=cf.Cost(1.0))
    b = cf.Extension("b").primitive("GeLU", "HE", cost=cf.Cost(2.0))
    with pytest.raises(ValueError, match="both implement GeLU in HE"):
        _compile(bert2, extensions=[a, b])
    bad = cf.Extension("bad").primitive("GeLU", "HE", cost=like("NoSuchOp", "HE", lambda n, r: 1.0))
    with pytest.raises(cf.UnsupportedOperator, match="no HE record of NoSuchOp"):
        _compile(bert2, extensions=[bad])


def test_extension_operators_are_calibrated_and_emulated():
    from cipherflow import runtime
    torch.manual_seed(0)
    rag = private_rag.rag_extension()
    model = private_rag.Retriever(n_docs=256, k=4, blocks=1).eval()
    x = torch.randn(1, 128, 768)
    with pytest.raises(cf.UnsupportedModel, match="cipherflow.Extension"):
        runtime.calibrate(model, [x])                            # without the extension: unknown operations
    cal = runtime.calibrate(model, [x], extensions=[rag])
    assert {"pool", "scores", "topk"} <= set(cal.nodes)
    graph = cf.models.from_torch(model, x, extensions=[rag])
    plan = _compile(graph, dep=cf.Deployment.of("cpu", "100Mbps/20ms"), extensions=[rag], calibration=cal)
    dom = {n.op_type: n.domain for n in plan.nodes}
    assert dom["TopK"] == "MPC" and dom["DocScores"] == "HE"   # a conversion to MPC before the top-k
    res = runtime.emulate(plan, model, x, extensions=[rag])
    assert res.checks["problems"] == []
    assert res.applied == {"conversions": res.checks["conversions"], "bootstraps": res.checks["bootstraps"]}
    assert res.output.dtype == torch.int64 and res.output.shape == (1, 4)    # indices stay integers in MPC
    assert res.exact and "equals the plaintext model's" in res.summary()

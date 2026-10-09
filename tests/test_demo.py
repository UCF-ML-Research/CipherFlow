"""The demo's input/output: a PyTorch Transformer in, the plan operator by operator out."""
import json
import warnings

import pytest

import cipherflow as cf

torch = pytest.importorskip("torch")

from cipherflow.cli import main  # noqa: E402
from demo import transformer  # noqa: E402


def _sig(g):
    idx = {n.node_id: i for i, n in enumerate(g.nodes)}
    return ([(n.op_type, tuple(n.input_shape), tuple(n.output_shape)) for n in g.nodes],
            sorted((idx[e.src], idx[e.dst], tuple(e.tensor_shape)) for e in g.edges))


def test_demo_models_trace_to_the_builtin_graphs():
    bert = cf.models.from_torch(transformer.bert(2), torch.zeros(1, 128, 768))
    vit = cf.models.from_torch(transformer.vit(2), torch.zeros(1, 197, 768))
    assert _sig(bert) == _sig(cf.models.bert(num_blocks=2, seq=128))
    assert _sig(vit) == _sig(cf.models.vit(num_blocks=2, tokens=197))


def test_operators_name_their_place_in_the_pytorch_code():
    g = cf.models.from_torch(transformer.bert(2), torch.zeros(1, 128, 768))
    src = [(n.op_type, n.attributes.get("source")) for n in g.nodes]
    assert src[:13] == [
        ("Input", "input x"), ("Linear_QKV", "blocks.0.attn.{q,k,v}"),
        ("Attention_QK_MatMul", "blocks.0.attn: matmul"), ("Softmax", "blocks.0.attn: softmax"),
        ("Attention_V_MatMul", "blocks.0.attn: matmul"), ("Out_Projection", "blocks.0.attn.o"),
        ("Residual_Add", "blocks.0: add"), ("LayerNorm", "blocks.0.ln1"), ("FFN_Linear_1", "blocks.0.mlp.fc1"),
        ("GeLU", "blocks.0.mlp.act"), ("FFN_Linear_2", "blocks.0.mlp.fc2"), ("Residual_Add", "blocks.0: add"),
        ("LayerNorm", "blocks.0.ln2")]
    assert src[13] == ("Linear_QKV", "blocks.1.attn.{q,k,v}")

    class MHA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mha, self.ln = torch.nn.MultiheadAttention(768, 12, batch_first=True), torch.nn.LayerNorm(768)

        def forward(self, x):
            return self.ln(x + self.mha(x, x, x, need_weights=False)[0])
    g = cf.models.from_torch(MHA(), torch.zeros(1, 128, 768))
    assert [n.attributes.get("source") for n in g.nodes] == [
        "input x", "mha: q/k/v in_proj", "mha: q @ k^T", "mha: softmax", "mha: attn @ v", "mha.out_proj", "add", "ln"]


@pytest.fixture(scope="module")
def plan():
    g = cf.models.from_torch(transformer.bert(1), torch.zeros(1, 128, 768))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        return cf.compile(g, cf.Deployment.of("cpu", "1Gbps/1ms"), he="openfhe-fast", mpc="sci")


def test_steps_list_every_operator_with_the_actions_before_it(plan):
    steps = plan.steps()
    ops = [s for s in steps if s["step"] == "operator"]
    assert [s["id"] for s in ops] == [n.id for n in plan.nodes]
    assert all(s["source"] for s in ops)
    acts = [s for s in steps if s["step"] != "operator"]
    assert len(acts) == len(plan.actions) and len(acts) == plan.counts()["conversions"] + plan.counts()["bootstraps"]
    seen = set()
    for s in steps:                          # an action converts the output of an operator that already ran
        if s["step"] == "operator":
            seen.add(s["id"])
        elif s["edge"]:
            assert s["edge"][0] in seen
    for s in ops:
        assert (s["level_in"] is not None) == (s["domain"] == "HE")
    text = plan.explain(steps=True)
    assert "blocks.0.attn.{q,k,v}" in text and "blocks.0.mlp.act" in text
    assert text.count("convert ") == plan.counts()["conversions"]


def test_plan_json_keeps_the_pytorch_locations(plan):
    back = cf.Plan.from_json(plan.to_json())
    assert [n.source for n in back.nodes] == [n.source for n in plan.nodes]
    assert back.steps() == plan.steps()
    assert json.loads(plan.to_json())["nodes"][1]["source"] == "blocks.0.attn.{q,k,v}"


def test_builtin_graphs_have_no_pytorch_locations():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        p = cf.compile(cf.models.bert(num_blocks=1), cf.Deployment.of("cpu", "1Gbps/1ms"), he="openfhe-fast", mpc="sci")
    assert all(n.source is None for n in p.nodes)
    assert "L0_qkv" in p.explain_steps()            # falls back to the node id


def test_cli_compiles_a_pytorch_file(tmp_path, capsys):
    out = tmp_path / "plan.json"
    assert main(["compile", "--torch", f"{transformer.__file__}:bert", "--input-shape", "1,128,768",
                 "--hardware", "cpu", "--he", "openfhe-fast", "--mpc", "sci", "--network", "10Mbps/1ms",
                 "--steps", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "graph=torch_BertEncoder" in text and "blocks.11.ln2" in text
    assert main(["explain", str(out), "--steps"]) == 0
    assert "blocks.11.ln2" in capsys.readouterr().out
    assert main(["compile", "--torch", "demo.transformer:vit", "--input-shape", "1,197,768", "--no-qkv",
                 "--hardware", "gpu", "--profile", "paper-gpu", "--security", "2pc-dealer"]) == 0
    assert "Linear_QKV" not in capsys.readouterr().out
    assert main(["compile", "--torch", "no_such_file.py:bert"]) == 2

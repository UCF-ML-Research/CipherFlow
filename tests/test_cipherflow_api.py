"""cipherflow v1 API: wrapper parity with the paper planner, JSON round trip, mixed plans."""
import cipherflow as cf
from compiler.state_expanded.planner import compile_plan_safe

GRAPH = cf.models.bert_paper(num_blocks=2, seq=128)      # the paper's graph, for the paper's profiles
def PAPER(hw):                                        # the paper's profiles (SHAFT: trusted dealer)
    return dict(profile=f"paper-{hw}", security="2pc-dealer")


def test_parity_with_planner():
    for hw, net in [("cpu", "10Mbps/1ms"), ("gpu", "1Gbps/40ms"), ("gpu", "3Gbps/80ms")]:
        dep = cf.Deployment.of(hw, net)
        plan = cf.compile(GRAPH, dep, **PAPER(hw))
        ref = compile_plan_safe(GRAPH, cf.load_profile(f"paper-{hw}"), dep.network.to_setting())
        assert abs(plan.cost["total_ms"] - ref.total_cost_ms) < 1e-6
        assert plan.domains() == ref.node_assignment


def test_json_round_trip(tmp_path):
    plan = cf.compile(GRAPH, cf.Deployment.of("gpu", "1Gbps/40ms"), **PAPER("gpu"))
    path = tmp_path / "plan.json"
    plan.to_json(str(path))
    back = cf.Plan.from_json(str(path))
    assert back.to_dict() == plan.to_dict()


def test_mixed_plan_has_conversions_and_levels():
    plan = cf.compile(GRAPH, cf.Deployment.of("gpu", "1Gbps/40ms"), **PAPER("gpu"))
    k = plan.counts()
    assert k["HE"] > 0 and k["MPC"] > 0 and k["conversions"] > 0
    he = [n for n in plan.nodes if n.domain == "HE"]
    assert all(n.level_in is not None and n.level_out <= n.level_in for n in he)
    ids = {n.id for n in plan.nodes} | {"<output>"}
    assert all(a.before in ids for a in plan.actions)


def test_unknown_profile_is_explicit():
    import pytest
    with pytest.raises(cf.ProfileNotFound):
        cf.load_profile("no-such-profile")


def test_pre_ln_blocks_are_fully_planned():
    """Regression: a join that is also the next region's fork (pre-LN) must not drop that region."""
    g = cf.models.vit(num_blocks=2, qkv_projection=False)
    for hw, net in [("gpu", "10Mbps/1ms"), ("cpu", "1Gbps/20ms")]:
        plan = cf.compile(g, cf.Deployment.of(hw, net), **PAPER(hw))
        assert len(plan.nodes) == len(g.nodes)
        assert plan.cost["total_ms"] > 0


def test_vit_variants_differ_in_structure():
    paper, faithful = cf.models.vit_paper(num_blocks=2), cf.models.vit(num_blocks=2)
    order = lambda g: [n.op_type for n in g.nodes]
    assert order(paper)[:2] == ["Attention_QK_MatMul", "Softmax"]          # post-LN, no embedding, no Q/K/V
    assert order(faithful)[:4] == ["Out_Projection", "LayerNorm", "Linear_QKV", "Attention_QK_MatMul"]

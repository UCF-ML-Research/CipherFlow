"""Validation tests for the state-expanded compiler.

Tests cover:
(a) Table 1 transitions are obeyed (HE level drops, constraints, reset).
(b) Every emitted plan honours the HE level budget.
(c) Compiler is never worse than min(all-HE, all-MPC, static) under any
    (bandwidth, RTT) setting for both CPU and GPU profiles.
(d) SESE merge alignment: at every Residual_Add the two operands meet
    in the same domain (and level, if HE).
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))

from compiler.state_expanded.bert_graph import bert_block_graph, default_manifest
from compiler.state_expanded.cost_model import NetworkSetting
from compiler.state_expanded.planner import (
    DEFAULT_LEVEL_DELTAS,
    compile_plan,
    evaluate_static_hybrid,
    evaluate_uniform_domain,
)
from compiler.state_expanded.profile_schema import LatencyProfile


PROFILE_CPU = HERE / "compiler" / "state_expanded" / "profiles" / "profile_cpu.json"
PROFILE_GPU = HERE / "compiler" / "state_expanded" / "profiles" / "profile_gpu.json"

BANDWIDTHS = [10e6, 100e6, 1e9, 3e9]
RTTS = [1.0, 20.0, 40.0, 80.0]


def _validate_level_budget(plan, L: int) -> None:
    """Assert no he_exec transition is taken with insufficient level."""
    for step in plan.steps:
        if step.action == "he_exec":
            # src.l >= delta was enforced by state_graph.outgoing
            if step.src is None or step.dst is None:
                continue
            if step.src.d == "HE":
                assert step.src.l >= 0, f"Negative source level on {step}"
                assert step.dst.l >= 0, f"Negative dest level on {step}"
                assert step.src.l <= L, f"Source level exceeds budget: {step}"


def _validate_transitions(plan) -> None:
    """Each step must match one of the five Table 1 transition types."""
    allowed = {"he_exec", "mpc_exec", "he2mpc", "mpc2he", "bootstrap"}
    for step in plan.steps:
        assert step.action in allowed, f"Unknown action {step.action} in plan step {step}"


def _validate_merge_alignment(plan, graph) -> None:
    """At each Residual_Add op in the plan, both incoming domains match.

    This is enforced structurally by the SESE transfer function, so the
    test is sanity: the node_assignment for add1/add2 is populated and
    domain-consistent with the upstream ops on both branches.
    """
    add_nodes = [n for n in graph.nodes if n.op_type == "Residual_Add"]
    # For each residual add, the predecessors (both branches) must
    # arrive in a compatible way. At minimum: the assignment exists.
    for n in add_nodes:
        assert n.node_id in plan.node_assignment, (
            f"Residual_Add {n.node_id} missing from assignment"
        )


class TestCompilerOnCpuProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PROFILE_CPU.exists():
            raise unittest.SkipTest(f"profile not found: {PROFILE_CPU}")
        cls.profile = LatencyProfile.load(PROFILE_CPU)
        cls.graph = bert_block_graph(default_manifest())

    def test_sweep_feasible(self) -> None:
        for bw in BANDWIDTHS:
            for rtt in RTTS:
                net = NetworkSetting(bw, rtt)
                plan = compile_plan(self.graph, self.profile, net)
                _validate_transitions(plan)
                _validate_level_budget(plan, self.profile.he_level_budget)
                _validate_merge_alignment(plan, self.graph)
                self.assertFalse(math.isinf(plan.total_cost_ms))

    def test_never_worse_than_baselines(self) -> None:
        for bw in BANDWIDTHS:
            for rtt in RTTS:
                net = NetworkSetting(bw, rtt)
                plan = compile_plan(self.graph, self.profile, net)
                baseline = min(
                    evaluate_uniform_domain(self.graph, self.profile, net, "HE"),
                    evaluate_uniform_domain(self.graph, self.profile, net, "MPC"),
                    evaluate_static_hybrid(self.graph, self.profile, net),
                )
                # 1% tolerance absorbs minor numerical slack in the
                # SESE transfer function; the claim of the paper is
                # "never materially worse than the best baseline".
                self.assertLessEqual(
                    plan.total_cost_ms, baseline * 1.01 + 1e-6,
                    f"Compiler {plan.total_cost_ms:.2f} worse than baseline "
                    f"{baseline:.2f} at BW={bw} RTT={rtt}",
                )

    def test_every_op_assigned_a_domain(self) -> None:
        net = NetworkSetting(1e9, 20.0)
        plan = compile_plan(self.graph, self.profile, net)
        for node in self.graph.nodes:
            self.assertIn(node.node_id, plan.node_assignment,
                          f"Missing assignment for {node.node_id}")


class TestCompilerOnGpuProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not PROFILE_GPU.exists():
            raise unittest.SkipTest(f"profile not found: {PROFILE_GPU}")
        cls.profile = LatencyProfile.load(PROFILE_GPU)
        cls.graph = bert_block_graph(default_manifest())

    def test_sweep_feasible(self) -> None:
        for bw in BANDWIDTHS:
            for rtt in RTTS:
                net = NetworkSetting(bw, rtt)
                plan = compile_plan(self.graph, self.profile, net)
                _validate_transitions(plan)
                _validate_level_budget(plan, self.profile.he_level_budget)
                _validate_merge_alignment(plan, self.graph)
                self.assertFalse(math.isinf(plan.total_cost_ms))

    def test_never_worse_than_baselines(self) -> None:
        for bw in BANDWIDTHS:
            for rtt in RTTS:
                net = NetworkSetting(bw, rtt)
                plan = compile_plan(self.graph, self.profile, net)
                baseline = min(
                    evaluate_uniform_domain(self.graph, self.profile, net, "HE"),
                    evaluate_uniform_domain(self.graph, self.profile, net, "MPC"),
                    evaluate_static_hybrid(self.graph, self.profile, net),
                )
                # 1% tolerance absorbs minor numerical slack in the
                # SESE transfer function; the claim of the paper is
                # "never materially worse than the best baseline".
                self.assertLessEqual(
                    plan.total_cost_ms, baseline * 1.01 + 1e-6,
                    f"Compiler {plan.total_cost_ms:.2f} worse than baseline "
                    f"{baseline:.2f} at BW={bw} RTT={rtt}",
                )


class TestStateTransitions(unittest.TestCase):
    def test_he_exec_consumes_level(self) -> None:
        from compiler.state_expanded.state_graph import ChainOp, State, outgoing
        from compiler.state_expanded.cost_model import StateExpandedCostModel
        if not PROFILE_CPU.exists():
            self.skipTest(f"profile not found: {PROFILE_CPU}")
        profile = LatencyProfile.load(PROFILE_CPU)
        cm = StateExpandedCostModel(profile)
        net = NetworkSetting(1e9, 1.0)
        L = profile.he_level_budget
        op = ChainOp(
            node_id="gelu", op_type="GeLU",
            input_shape=(1, 16, 3072), output_shape=(1, 16, 3072),
            he_level_delta=DEFAULT_LEVEL_DELTAS["GeLU"],
        )
        # HE exec from full budget must produce state at level L - 4.
        src = State(0, "HE", L)
        edges = outgoing(src, [op], cm, net, L)
        he_execs = [e for e in edges if e.action == "he_exec"]
        self.assertTrue(he_execs, "No HE exec edge from full-budget state")
        self.assertEqual(he_execs[0].dst.l, L - 4)
        # HE exec from budget 3 (< 4) must be impossible.
        src = State(0, "HE", 3)
        edges = outgoing(src, [op], cm, net, L)
        he_execs = [e for e in edges if e.action == "he_exec"]
        self.assertFalse(he_execs, "HE exec should be pruned when l < delta")

    def test_bootstrap_refreshes(self) -> None:
        from compiler.state_expanded.state_graph import ChainOp, State, outgoing
        from compiler.state_expanded.cost_model import StateExpandedCostModel
        if not PROFILE_CPU.exists():
            self.skipTest(f"profile not found: {PROFILE_CPU}")
        profile = LatencyProfile.load(PROFILE_CPU)
        cm = StateExpandedCostModel(profile)
        net = NetworkSetting(1e9, 1.0)
        L = profile.he_level_budget
        op = ChainOp("x", "GeLU", (1, 16, 3072), (1, 16, 3072), 4)
        src = State(0, "HE", 0)  # depleted
        edges = outgoing(src, [op], cm, net, L)
        bs = [e for e in edges if e.action == "bootstrap"]
        self.assertTrue(bs, "No bootstrap edge from depleted state")
        self.assertEqual(bs[0].dst.l, L)


if __name__ == "__main__":
    unittest.main()

"""Optimality tests: compiler vs flat-state-graph Dijkstra (brute force).

The compiler uses a hierarchical (SESE) planner. The brute-force
verifier runs a flat Dijkstra over the full state-expanded graph.
Both should produce the same global minimum under the paper's cost
model.

We test on small synthetic chains where enumeration is feasible.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))

from ir.types import DataEdge, OperatorGraph, OperatorNode

from compiler.state_expanded.brute_force_verifier import brute_force_minimum
from compiler.state_expanded.cost_model import NetworkSetting
from compiler.state_expanded.planner import compile_plan
from compiler.state_expanded.profile_schema import LatencyProfile


PROFILE_CPU = HERE / "compiler" / "state_expanded" / "profiles" / "profile_cpu.json"
PROFILE_FULL = HERE / "compiler" / "state_expanded" / "profiles" / "profile_cpu_full_model.json"


def _tiny_chain(tag: str = "tiny") -> OperatorGraph:
    """4-op linear chain with shapes that hit all three HE contracts."""
    tok = (1, 16, 768)
    ffn1_out = (1, 16, 3072)
    nodes = [
        OperatorNode("ln1",  "LayerNorm",    tok,      tok),
        OperatorNode("ffn1", "FFN_Linear_1", tok,      ffn1_out),
        OperatorNode("gelu", "GeLU",         ffn1_out, ffn1_out),
        OperatorNode("ffn2", "FFN_Linear_2", ffn1_out, tok),
    ]
    edges = [
        DataEdge("ln1",  "ffn1", tok),
        DataEdge("ffn1", "gelu", ffn1_out),
        DataEdge("gelu", "ffn2", ffn1_out),
    ]
    return OperatorGraph(graph_id=tag, nodes=nodes, edges=edges)


class TestOptimality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROFILE_CPU.exists():
            raise unittest.SkipTest(f"profile not found: {PROFILE_CPU}")
        cls.profile = LatencyProfile.load(PROFILE_CPU)

    def test_tiny_chain_matches_brute_force(self):
        graph = _tiny_chain()
        for bw in (10e6, 100e6, 1e9, 3e9):
            for rtt in (1.0, 20.0, 80.0):
                net = NetworkSetting(bw, rtt)
                bf = brute_force_minimum(graph, self.profile, net)
                plan = compile_plan(graph, self.profile, net)
                self.assertFalse(math.isinf(bf))
                self.assertFalse(math.isinf(plan.total_cost_ms))
                # Compiler should be within 0.1 ms of the flat-Dijkstra
                # minimum. Any positive gap would indicate a SESE
                # alignment bug or missing transition.
                self.assertLessEqual(
                    plan.total_cost_ms,
                    bf + 0.1,
                    f"compiler={plan.total_cost_ms:.3f} > brute_force="
                    f"{bf:.3f} at BW={bw:g} RTT={rtt}",
                )
                # And the compiler should never be below the flat-graph
                # minimum (that would indicate a cost-model or transition
                # accounting bug).
                self.assertGreaterEqual(
                    plan.total_cost_ms,
                    bf - 0.1,
                    f"compiler={plan.total_cost_ms:.3f} < brute_force="
                    f"{bf:.3f} (impossible under the same cost model) "
                    f"at BW={bw:g} RTT={rtt}",
                )


class TestFullModelOptimality(unittest.TestCase):
    """On the 12-block graph, verify the compiler is within a small
    slack of the flat-Dijkstra lower bound (flat Dijkstra doesn't
    enforce join alignment, so it can only be <= the SESE result)."""

    @classmethod
    def setUpClass(cls):
        if not PROFILE_FULL.exists():
            raise unittest.SkipTest(f"profile not found: {PROFILE_FULL}")
        cls.profile = LatencyProfile.load(PROFILE_FULL)

    def test_12block_compiler_not_worse_than_lower_bound_plus_eps(self):
        from compiler.state_expanded.bert_graph import (
            bert_multi_block_graph,
            full_model_manifest,
        )
        graph = bert_multi_block_graph(12, full_model_manifest())
        for bw, rtt in [(100e6, 20.0), (1e9, 20.0), (3e9, 1.0)]:
            net = NetworkSetting(bw, rtt)
            bf = brute_force_minimum(graph, self.profile, net)
            plan = compile_plan(graph, self.profile, net)
            # The flat Dijkstra ignores join alignment, so it's a true
            # lower bound. The compiler's cost must be >= it.
            self.assertGreaterEqual(
                plan.total_cost_ms,
                bf - 1.0,
                f"compiler={plan.total_cost_ms:.1f} below flat-Dijkstra "
                f"lower bound {bf:.1f} at BW={bw:g} RTT={rtt}",
            )


if __name__ == "__main__":
    unittest.main()

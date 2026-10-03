"""Guarantee tests: the compiler is never slower than any same-stack baseline.

This is the central practical claim the paper makes operationalised as
pass/fail: across every (bandwidth, RTT) setting and every profile we
maintain, ``compile_plan_safe`` must return a plan whose total cost is
≤ the minimum over {FHE, MPC, linHE_nlMPC, linMPC_nlHE, attnMPC_FFNHE}
evaluated under the same cost model.

If this test ever fails, it means either:
(a) the compiler's state-expanded search missed an optimal plan
    (optimality bug), or
(b) the cost model is inconsistent between the compiler and the
    baseline evaluator.

Both cases are errors that would require investigation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))

from compiler.state_expanded.bert_graph import (
    bert_block_graph,
    bert_multi_block_graph,
    default_manifest,
    full_model_manifest,
)
from compiler.state_expanded.cost_model import NetworkSetting
from compiler.state_expanded.planner import (
    compile_plan_safe,
    evaluate_named_static_hybrids,
    evaluate_uniform_domain,
)
from compiler.state_expanded.profile_schema import LatencyProfile

PROFILE_CPU = HERE / "compiler" / "state_expanded" / "profiles" / "profile_cpu.json"
PROFILE_GPU = HERE / "compiler" / "state_expanded" / "profiles" / "profile_gpu.json"
PROFILE_FULL = HERE / "compiler" / "state_expanded" / "profiles" / "profile_cpu_full_model.json"


BANDWIDTHS = [10e6, 100e6, 1e9, 3e9]
RTTS = [1.0, 20.0, 40.0, 80.0]


def _check_guarantee(profile_path: Path, graph_fn, test):
    if not profile_path.exists():
        test.skipTest(f"profile not found: {profile_path}")
    profile = LatencyProfile.load(profile_path)
    graph = graph_fn()
    for bw in BANDWIDTHS:
        for rtt in RTTS:
            net = NetworkSetting(bw, rtt)
            all_he  = evaluate_uniform_domain(graph, profile, net, "HE")
            all_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            hybrids = evaluate_named_static_hybrids(graph, profile, net)
            baselines = [
                all_he, all_mpc,
                hybrids["linHE_nlMPC"],
                hybrids["linMPC_nlHE"],
                hybrids["attnMPC_ffnHE"],
            ]
            best_base = min(baselines)
            plan = compile_plan_safe(graph, profile, net)
            test.assertLessEqual(
                plan.total_cost_ms,
                best_base + 1e-6,
                f"GUARANTEE VIOLATED at BW={bw:g} RTT={rtt}: "
                f"compiler={plan.total_cost_ms:.3f} "
                f"> min(baselines)={best_base:.3f} "
                f"[strategy={plan.strategy_used}]",
            )


class TestPerBlockCpuGuarantee(unittest.TestCase):
    def test_compiler_never_slower_than_any_baseline(self):
        _check_guarantee(
            PROFILE_CPU,
            lambda: bert_block_graph(default_manifest()),
            self,
        )


class TestPerBlockGpuGuarantee(unittest.TestCase):
    def test_compiler_never_slower_than_any_baseline(self):
        _check_guarantee(
            PROFILE_GPU,
            lambda: bert_block_graph(default_manifest()),
            self,
        )


class TestFullModelGuarantee(unittest.TestCase):
    def test_compiler_never_slower_than_any_baseline(self):
        _check_guarantee(
            PROFILE_FULL,
            lambda: bert_multi_block_graph(12, full_model_manifest()),
            self,
        )


if __name__ == "__main__":
    unittest.main()

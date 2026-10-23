"""The CostSphere and EncFormer re-implementations in demo/compare (see docs/comparison.md)."""
import itertools
import math

import pytest

from cipherflow import profiles
from compiler.state_expanded.bert_graph import bert_multi_block_graph, full_model_manifest
from compiler.state_expanded.cost_model import NetworkSetting, StateExpandedCostModel
from demo.compare.prior_planners import Chain, cipherflow, costsphere, costsphere_window, encformer, score


def _eq4(chain, assign, n):                     # Eq. 4 by hand
    total, count = 0.0, 0
    for i, d in enumerate(assign):
        total += chain.op[i][d]
        if i and assign[i - 1] != d:
            total += chain.conv[i][(assign[i - 1], d)]
        count = ((count if i and assign[i - 1] == "HE" else 0) + 1) if d == "HE" else 0
        if count > n:
            total, count = total + chain.boot, 0
    return total


def _eq13(chain, assign, n):                    # the sliding window by hand
    total = sum(chain.op[i][d] for i, d in enumerate(assign))
    total += sum(chain.conv[i][(assign[i - 1], d)] for i, d in enumerate(assign) if i and assign[i - 1] != d)
    return total + chain.boot * sum(all(x == "HE" for x in assign[i - n + 1:i + 1]) for i in range(n - 1, len(assign)))


@pytest.fixture(scope="module")
def block():
    prof = profiles.load("paper-cpu")
    return bert_multi_block_graph(1, full_model_manifest()), prof, StateExpandedCostModel(prof)


@pytest.mark.parametrize("bw,rtt", [(10e6, 1.0), (1e9, 20.0)])
@pytest.mark.parametrize("n", [1, 3, 9])
def test_costsphere_solvers_find_the_optimum(block, bw, rtt, n):
    graph, _, cm = block
    chain = Chain(graph, cm, NetworkSetting(bw, rtt))
    every = list(itertools.product(("HE", "MPC"), repeat=len(chain.order)))
    assert costsphere(chain, n)[1] == pytest.approx(min(_eq4(chain, list(a), n) for a in every))
    pytest.importorskip("scipy")
    assert costsphere_window(chain, n)[1] == pytest.approx(min(_eq13(chain, list(a), n) for a in every))


def test_encformer_split_and_cipherflow_is_never_slower(block):
    graph, prof, cm = block
    net = NetworkSetting(100e6, 20.0)
    ms, conversions, bootstraps, _ = encformer(graph, cm, net, prof.he_level_budget)
    assert bootstraps == 0 and conversions >= 8                 # four HE<->MPC round trips per block
    ours = cipherflow(graph, prof, cm, net, prof.he_level_budget, edge_shapes=True)[0]
    assert ours <= ms
    chain = Chain(graph, cm, net)
    assert ours <= min(score(graph, cm, net, costsphere(chain, n)[0], prof.he_level_budget)[0]
                       for n in range(1, 21)) + 1e-6
    assert math.isfinite(ours)

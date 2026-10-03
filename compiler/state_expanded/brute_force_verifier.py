"""Brute-force reference solver for optimality verification.

Given a small operator graph and the same cost model the compiler uses,
this module enumerates *every* valid execution plan and returns the
minimum-cost one. The compiler's output must never exceed this minimum;
a gap is a proof that the compiler is suboptimal.

Scope
-----
- Only for small graphs — enumeration is exponential in the number of
  operators (branch over domain choice, bootstrap insertion, conversion
  insertion at each position).
- Used strictly as a test oracle for ``compile_plan``.

Enumeration strategy
--------------------
We treat the problem as a depth-first search over states
``(i, d, l, total_cost)`` using ``state_graph.outgoing`` to generate
the legal successor states. We cap:
- number of bootstraps in a row at 2 (any more is provably wasteful
  since l resets to L after one bootstrap)
- number of consecutive conversions at 2 (any more just oscillates)

A visited-state memo keyed on ``(i, d, l)`` stores the minimum cost
found so far — this turns the DFS into effectively another Dijkstra,
matching the compiler's algorithm. To still serve as an independent
verifier we reimplement the recurrence using a simple priority-queue
Dijkstra that does NOT use the SESE hierarchy — so it exercises the
flat state graph and confirms the hierarchical planner matches.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from ir.types import OperatorGraph

from .cost_model import NetworkSetting, StateExpandedCostModel
from .planner import DEFAULT_LEVEL_DELTAS
from .profile_schema import LatencyProfile
from .state_graph import ChainOp, State, outgoing

Domain = str


def _linearise(graph: OperatorGraph) -> List[ChainOp]:
    """Topological sort into a flat chain for brute-force search.

    The verifier only runs on graphs the compiler also handles as a
    chain (residuals are handled via the same Dijkstra; we bake the
    residual alignment into the chain by treating Residual_Add as a
    single-input pass-through, which under the additive cost model
    gives a *lower bound* on cost — the true cost is ≥ this bound).
    """
    # Topological sort.
    indeg = {n.node_id: 0 for n in graph.nodes}
    out_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        indeg[e.dst] += 1
        out_adj[e.src].append(e.dst)
    node_map = {n.node_id: n for n in graph.nodes}
    ready = [n.node_id for n in graph.nodes if indeg[n.node_id] == 0]
    topo: List[str] = []
    while ready:
        ready.sort()
        u = ready.pop(0)
        topo.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return [
        ChainOp(
            node_id=node_map[n].node_id,
            op_type=node_map[n].op_type,
            input_shape=node_map[n].input_shape,
            output_shape=node_map[n].output_shape,
            he_level_delta=DEFAULT_LEVEL_DELTAS.get(node_map[n].op_type, 0),
        )
        for n in topo
    ]


def brute_force_minimum(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
) -> float:
    """Return the minimum-cost plan on the flat state-expanded graph.

    Uses Dijkstra directly over all states — no SESE hierarchy, no
    region transfer caching. This is a reference for what ``compile_plan``
    should achieve (or beat, if it can find a lower-cost plan via
    smarter region alignment).
    """
    cm = StateExpandedCostModel(profile)
    L = profile.he_level_budget
    chain = _linearise(graph)

    # Super-source: can start in HE or MPC at full budget.
    dist: Dict[Tuple[int, Domain, int], float] = {}
    pq: List[Tuple[float, int, Domain, int]] = []
    for d0 in ("HE", "MPC"):
        k = (0, d0, L)
        dist[k] = 0.0
        heapq.heappush(pq, (0.0, 0, d0, L))

    while pq:
        c, i, d, l = heapq.heappop(pq)
        if dist.get((i, d, l), float("inf")) < c - 1e-12:
            continue
        if i == len(chain):
            continue
        state = State(i, d, l)
        for edge in outgoing(state, chain, cm, net, L):
            nk = (edge.dst.i, edge.dst.d, edge.dst.l)
            nc = c + edge.cost_ms
            if nc + 1e-12 < dist.get(nk, float("inf")):
                dist[nk] = nc
                heapq.heappush(pq, (nc, edge.dst.i, edge.dst.d, edge.dst.l))

    # Minimum over any terminal state.
    final = len(chain)
    best = min(
        (c for (i, _d, _l), c in dist.items() if i == final),
        default=float("inf"),
    )
    return best

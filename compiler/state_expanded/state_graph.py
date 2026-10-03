"""State-expanded graph for a linear chain (paper §4.2.2, Table 1).

State
-----
    s = (i, d, l)

where ``i`` is an index along the chain, ``d ∈ {HE, MPC}`` is the
execution domain, and ``l ∈ {0..L}`` is the discretised remaining HE
noise budget. ``L`` is ``profile.he_level_budget``.

For MPC states we use ``l = L`` canonically: the HE budget is
irrelevant there, but fixing one representative value keeps the graph
a DAG with a well-defined "current" budget when we re-enter HE via a
conversion. This matches the paper's Table 1 entries
``(i,MPC,L) -> (i+1,MPC,L)`` and ``(i,MPC,L) -> (i,HE,L)``.

Transitions per Table 1 (operator v_i sits between index i and i+1):

1. HE execution              (i,HE,l)  -> (i+1,HE,l-δ_i)   if l >= δ_i
2. MPC execution             (i,MPC,L) -> (i+1,MPC,L)
3. HE→MPC conversion         (i,HE,l)  -> (i,MPC,L)        (edge at v_i)
4. MPC→HE conversion         (i,MPC,L) -> (i,HE,L)         (edge at v_i)
5. Bootstrapping             (i,HE,l)  -> (i,HE,L)         (same i)

We treat ``v_0 .. v_{N-1}`` as the operators; source is index 0, sink is
index N. Source and sink can begin/end in either domain — the caller
(SESE wrapper or planner) picks boundary states.

This module is pure graph+cost plumbing; it does not know about BERT
or the SESE hierarchy. Branching (residual) graphs are handled one
level up (``sese.py``).

Output is a list of ``(dst_state, cost_ms, action_label)`` edges given
a source state and an operator context. The planner uses Dijkstra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from .cost_model import CostEstimate, NetworkSetting, StateExpandedCostModel
from .profile_schema import LatencyProfile

Domain = str  # "HE" | "MPC"


@dataclass(frozen=True)
class State:
    i: int          # operator index along the linear chain
    d: Domain       # "HE" | "MPC"
    l: int          # HE level budget (0..L). For MPC, always L.

    def key(self) -> Tuple[int, Domain, int]:
        return (self.i, self.d, self.l)


@dataclass(frozen=True)
class TransitionEdge:
    src: State
    dst: State
    cost_ms: float
    action: str            # "he_exec" | "mpc_exec" | "he2mpc" | "mpc2he" | "bootstrap"
    detail: str = ""       # op_type / tensor_shape / ...


@dataclass(frozen=True)
class ChainOp:
    """One operator on the chain — what the planner needs to know."""
    node_id: str
    op_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    he_level_delta: int


def enumerate_states(chain_len: int, L: int) -> List[State]:
    """All (i, d, l) states used by Dijkstra."""
    states: List[State] = []
    for i in range(chain_len + 1):
        for l in range(L + 1):
            states.append(State(i, "HE", l))
        # single canonical MPC state per index
        states.append(State(i, "MPC", L))
    return states


def outgoing(
    src: State,
    chain: List[ChainOp],
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
    conversion_shape_at: Optional[Callable[[int], Tuple[int, ...]]] = None,
    allow_bootstrap: bool = True,
) -> List[TransitionEdge]:
    """Return all legal outgoing edges from ``src`` under the paper's Table 1.

    ``conversion_shape_at(i)`` gives the tensor shape of the active
    edge at index i — for linear chains this is the operator's input
    shape. It is a callable so SESE code can provide per-position
    shapes that may differ from chain[i].input_shape in branching
    regions (e.g. a residual tail).
    """

    edges: List[TransitionEdge] = []
    i = src.i
    if i < len(chain):
        op = chain[i]
        # -- 1. HE execution --
        if src.d == "HE" and src.l >= op.he_level_delta:
            est = cm.estimate_operator(
                op.op_type, "HE", op.input_shape, op.output_shape, net
            )
            if est.resolution != "infeasible":
                edges.append(
                    TransitionEdge(
                        src=src,
                        dst=State(i + 1, "HE", src.l - op.he_level_delta),
                        cost_ms=est.latency_ms,
                        action="he_exec",
                        detail=f"{op.node_id}:{op.op_type}",
                    )
                )
        # -- 2. MPC execution --
        if src.d == "MPC":
            est = cm.estimate_operator(
                op.op_type, "MPC", op.input_shape, op.output_shape, net
            )
            if est.resolution != "infeasible":
                edges.append(
                    TransitionEdge(
                        src=src,
                        dst=State(i + 1, "MPC", L),
                        cost_ms=est.latency_ms,
                        action="mpc_exec",
                        detail=f"{op.node_id}:{op.op_type}",
                    )
                )

    # -- 3. HE->MPC conversion (same i) --
    if src.d == "HE":
        shape = conversion_shape_at(i) if conversion_shape_at else (
            chain[i].input_shape if i < len(chain) else chain[-1].output_shape
        )
        est = cm.estimate_conversion("HE", "MPC", shape, net)
        edges.append(
            TransitionEdge(
                src=src,
                dst=State(i, "MPC", L),
                cost_ms=est.latency_ms,
                action="he2mpc",
                detail=f"shape={shape}",
            )
        )

    # -- 4. MPC->HE conversion (same i) --
    if src.d == "MPC":
        shape = conversion_shape_at(i) if conversion_shape_at else (
            chain[i].input_shape if i < len(chain) else chain[-1].output_shape
        )
        est = cm.estimate_conversion("MPC", "HE", shape, net)
        edges.append(
            TransitionEdge(
                src=src,
                dst=State(i, "HE", L),
                cost_ms=est.latency_ms,
                action="mpc2he",
                detail=f"shape={shape}",
            )
        )

    # -- 5. Bootstrapping (same i, HE only) --
    if src.d == "HE" and src.l < L and allow_bootstrap:
        est = cm.estimate_bootstrap(net)
        if est.resolution != "infeasible":
            edges.append(
                TransitionEdge(
                    src=src,
                    dst=State(i, "HE", L),
                    cost_ms=est.latency_ms,
                    action="bootstrap",
                    detail="",
                )
            )

    return edges

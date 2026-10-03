"""Single-Entry Single-Exit (SESE) region detection + transfer functions.

Paper §4.2.3 identifies SESE regions on the operator graph G. A
sub-block ``b`` is a subgraph bounded by a unique fork ``u`` (out-degree
> 1) and a unique join ``v`` (in-degree > 1) such that every path from
``u`` into ``b`` flows exclusively to ``v``. For Transformer blocks the
canonical SESE regions are:

    R1 = {Attention sublayer}       fork=block_in, join=Residual_Add_1
    R2 = {FFN sublayer}             fork=LayerNorm, join=Residual_Add_2

Inside a region the planner must handle merge-alignment at the join:
both incoming branches into a Residual_Add in HE must carry ciphertexts
at compatible levels. We enumerate candidate alignment strategies
(level-drop one branch, bootstrap the other, both switch to MPC and
add there, then convert back, etc.) and pick the cost-minimal one for
each ``(s_src, s_dst)`` boundary pair.

The transfer function

    T_R(s_src, s_dst) = min cost of traversing region R
                        starting from boundary state s_src
                        ending at boundary state s_dst

is computed once per (region, network setting) and consumed by the
macro-level planner in ``planner.py``.

Implementation detail
---------------------
We do not materialise an exponentially large cross-product graph.
Instead we run independent Dijkstras on the two branches separately
and combine them at the join with explicit alignment steps. For each
boundary pair and each candidate alignment strategy we take the min.
This is polynomial in L and branch length and matches the §4.2.3
claim of ``O(N_regions * L^2)`` macro planning after region
pre-compilation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode

from .cost_model import NetworkSetting, StateExpandedCostModel
from .profile_schema import HE_LEVEL_BUDGET, LatencyProfile
from .state_graph import ChainOp, State, TransitionEdge, outgoing

Shape = Tuple[int, ...]
Domain = str


# ---------- Region discovery ----------


@dataclass
class SeseRegion:
    """A single SESE region: fork -> two branches -> join.

    ``main_path`` is the longer path (with the actual compute) and
    ``skip_path`` is the residual bypass. Both lists are node_ids in
    topological order between fork and join, excluding the fork but
    including the operators strictly before the join on each branch.
    """

    region_id: str
    fork: str          # fork node_id (out-degree > 1)
    join: str          # join node_id (in-degree > 1)
    main_path: List[str] = field(default_factory=list)
    skip_path: List[str] = field(default_factory=list)


def find_sese_regions(graph: OperatorGraph) -> List[SeseRegion]:
    """Detect SESE regions of the form fork -> {two branches} -> join.

    We only recognise the two-branch residual pattern here because
    that is the only SESE shape the BERT/ResNet family presents and
    the paper explicitly targets that family. Extending to k-branch
    SESE follows the same algorithm with an extra loop.
    """
    out_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    in_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        out_adj[e.src].append(e.dst)
        in_adj[e.dst].append(e.src)

    regions: List[SeseRegion] = []
    for fork in graph.nodes:
        if len(out_adj[fork.node_id]) < 2:
            continue
        # For each pair of children from fork, find the nearest common
        # descendant with in-degree >= 2 that consumes both children.
        children = out_adj[fork.node_id]
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                a, b = children[i], children[j]
                path_a = _descendants_in_order(a, out_adj)
                path_b = _descendants_in_order(b, out_adj)
                set_a = set(path_a)
                set_b = set(path_b)
                common = [n for n in path_a if n in set_b]
                # A valid join is a node with >=2 predecessors that
                # come from the two branches from this fork. The branches
                # are: {a} ∪ set_a ∪ {fork}  and  {b} ∪ set_b ∪ {fork}.
                # We only need at least one predecessor on EACH branch,
                # which is stronger than "2 predecessors in the union".
                fork_id = fork.node_id
                branch_a_nodes = set_a | {a}
                branch_b_nodes = set_b | {b}

                def _qualifies(n: str) -> bool:
                    if len(in_adj[n]) < 2:
                        return False
                    preds = set(in_adj[n])
                    # A predecessor is on branch A if it's in set_a ∪ {a, fork}
                    pred_on_a = preds & (branch_a_nodes | {fork_id})
                    pred_on_b = preds & (branch_b_nodes | {fork_id})
                    # Both branches must feed into this join with at least
                    # one distinct predecessor each (fork-only on both
                    # sides does not count).
                    return len(pred_on_a) >= 1 and len(pred_on_b) >= 1 and (pred_on_a | pred_on_b) != {fork_id}

                join = next((n for n in common if _qualifies(n)), None)
                if join is None:
                    continue
                # main_path = nodes from `a` up to (but not including) join
                # skip_path = nodes from `b` up to (but not including) join
                ma = _prefix_up_to(path_a, join)
                mb = _prefix_up_to(path_b, join)
                # Heuristic: the longer arm is "main", shorter is "skip".
                if len(ma) >= len(mb):
                    main_path, skip_path = ma, mb
                else:
                    main_path, skip_path = mb, ma
                regions.append(
                    SeseRegion(
                        region_id=f"{fork.node_id}__{join}",
                        fork=fork.node_id,
                        join=join,
                        main_path=main_path,
                        skip_path=skip_path,
                    )
                )
    # Deduplicate overlapping regions: keep the smallest (innermost).
    regions = _keep_innermost(regions)
    return regions


def _descendants_in_order(start: str, out_adj: Dict[str, List[str]]) -> List[str]:
    seen: Dict[str, int] = {}
    order: List[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen[node] = 1
        order.append(node)
        for child in out_adj.get(node, []):
            if child not in seen:
                stack.append(child)
    return order


def _prefix_up_to(path: List[str], stop: str) -> List[str]:
    out = []
    for n in path:
        if n == stop:
            break
        out.append(n)
    return out


def _keep_innermost(regions: List[SeseRegion]) -> List[SeseRegion]:
    if not regions:
        return []
    # Sort by (len(main+skip), region_id) ascending, and drop any
    # region that is a strict superset of another (by node set).
    regions_sorted = sorted(
        regions, key=lambda r: (len(r.main_path) + len(r.skip_path), r.region_id)
    )
    keep: List[SeseRegion] = []
    for r in regions_sorted:
        r_nodes = set(r.main_path) | set(r.skip_path) | {r.fork, r.join}
        dominated = False
        for k in keep:
            k_nodes = set(k.main_path) | set(k.skip_path) | {k.fork, k.join}
            if r.fork == k.fork and r.join == k.join and k_nodes.issubset(r_nodes):
                dominated = True
                break
        if not dominated:
            keep.append(r)
    return keep


# ---------- Dijkstra over the state graph ----------


def dijkstra_chain(
    source: State,
    chain: List[ChainOp],
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
    allow_bootstrap: bool = True,
) -> Dict[Tuple[int, Domain, int], Tuple[float, Optional[TransitionEdge]]]:
    """Shortest-path from ``source`` over the state-expanded chain.

    Returns ``dist[state_key] = (cost, predecessor_edge)``. Caller can
    query any destination state and backtrack via the predecessor
    chain.
    """
    dist: Dict[Tuple[int, Domain, int], Tuple[float, Optional[TransitionEdge]]] = {}
    dist[source.key()] = (0.0, None)
    pq: List[Tuple[float, int, str, int]] = [(0.0, source.i, source.d, source.l)]
    counter = 0
    while pq:
        cost, i, d, l = heapq.heappop(pq)
        state = State(i, d, l)
        if dist[state.key()][0] < cost - 1e-12:
            continue
        if i == len(chain):
            # no more transitions from end
            continue
        for edge in outgoing(state, chain, cm, net, L, allow_bootstrap=allow_bootstrap):
            ncost = cost + edge.cost_ms
            key = edge.dst.key()
            prev = dist.get(key)
            if prev is None or ncost + 1e-12 < prev[0]:
                dist[key] = (ncost, edge)
                heapq.heappush(pq, (ncost, edge.dst.i, edge.dst.d, edge.dst.l))
            counter += 1
            if counter > 200000:
                break
    return dist


# ---------- Region transfer function ----------


@dataclass
class RegionPlan:
    """Result of solving one (s_src, s_dst) pair inside a region.

    The ``steps`` list is the concrete per-node decisions to be played
    back by the planner when materialising the final plan.
    """
    cost_ms: float
    steps: List[TransitionEdge]

    @classmethod
    def infeasible(cls) -> "RegionPlan":
        return cls(cost_ms=float("inf"), steps=[])


@dataclass
class RegionTransferTable:
    region_id: str
    table: Dict[Tuple[Tuple[int, Domain, int], Tuple[int, Domain, int]], RegionPlan] = field(
        default_factory=dict
    )

    def lookup(self, s_src: State, s_dst: State) -> RegionPlan:
        return self.table.get((s_src.key(), s_dst.key()), RegionPlan.infeasible())


def _build_chain(
    node_ids: List[str], node_map: Dict[str, OperatorNode], level_deltas: Dict[str, int]
) -> List[ChainOp]:
    out: List[ChainOp] = []
    for nid in node_ids:
        n = node_map[nid]
        out.append(
            ChainOp(
                node_id=n.node_id,
                op_type=n.op_type,
                input_shape=n.input_shape,
                output_shape=n.output_shape,
                he_level_delta=level_deltas.get(n.op_type, 0),
            )
        )
    return out


def compute_region_transfer(
    region: SeseRegion,
    graph: OperatorGraph,
    level_deltas: Dict[str, int],
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
    boundary_states: List[State],
) -> RegionTransferTable:
    """Compute T_R(s_src, s_dst) over ``boundary_states``.

    Strategy
    --------
    * Traverse ``main_path`` as a state-expanded chain from the fork
      boundary to a pre-join state.
    * Traverse ``skip_path`` independently.
    * At the join, the ``Residual_Add`` requires *matched* incoming
      domains. If both arrive in HE, their levels must also match
      (we emit a level-drop to the min of the two); if only one is HE,
      we insert an HE<->MPC conversion on the other side.
    * We pick the min-cost among candidate (main_out_state,
      skip_out_state, add_domain) combinations.
    """

    node_map = {n.node_id: n for n in graph.nodes}
    main_chain = _build_chain(region.main_path, node_map, level_deltas)
    skip_chain = _build_chain(region.skip_path, node_map, level_deltas)

    join_node = node_map[region.join]
    join_op = ChainOp(
        node_id=join_node.node_id,
        op_type=join_node.op_type,
        input_shape=join_node.input_shape,
        output_shape=join_node.output_shape,
        he_level_delta=level_deltas.get(join_node.op_type, 0),
    )

    table: Dict[Tuple[Tuple[int, Domain, int], Tuple[int, Domain, int]], RegionPlan] = {}

    for s_src in boundary_states:
        # --- explore main path from s_src ---
        main_src = State(0, s_src.d, s_src.l)
        main_dist = dijkstra_chain(main_src, main_chain, cm, net, L)
        # --- explore skip path from s_src (both branches share the fork) ---
        skip_dist = dijkstra_chain(main_src, skip_chain, cm, net, L)

        # Enumerate end states for both arms.
        main_ends = [
            (State(len(main_chain), d, l), c)
            for (idx, d, l), (c, _) in main_dist.items()
            if idx == len(main_chain)
        ]
        skip_ends = [
            (State(len(skip_chain), d, l), c)
            for (idx, d, l), (c, _) in skip_dist.items()
            if idx == len(skip_chain)
        ]

        for s_dst in boundary_states:
            best: RegionPlan = RegionPlan.infeasible()
            for main_end, main_cost in main_ends:
                for skip_end, skip_cost in skip_ends:
                    # Align + execute join in either HE or MPC domain.
                    # For HE, try all four bootstrap strategies.
                    strategies = [
                        ("MPC", "none"),
                        ("HE",  "none"),
                        ("HE",  "main"),
                        ("HE",  "skip"),
                        ("HE",  "both"),
                    ]
                    for add_domain, bs_plan in strategies:
                        align_cost, align_steps, common_l = _align_for_join(
                            main_end, skip_end, add_domain, join_op, cm, net, L,
                            bootstrap_plan=bs_plan,
                        )
                        if align_cost == float("inf"):
                            continue
                        # Execute join in add_domain
                        join_est = cm.estimate_operator(
                            join_op.op_type, add_domain, join_op.input_shape,
                            join_op.output_shape, net
                        )
                        if join_est.resolution == "infeasible":
                            continue
                        # After join we are at State with i=1 past join,
                        # but for the region we represent that as a
                        # "post-join" state; then conversion to s_dst if
                        # needed.
                        post_state = State(
                            0,
                            add_domain,
                            (common_l - join_op.he_level_delta) if add_domain == "HE" else L,
                        )
                        if add_domain == "HE" and post_state.l < 0:
                            continue
                        # Transition from post_state to s_dst: allow at
                        # most one conversion (HE<->MPC) and bootstrap.
                        tail_cost, tail_steps = _tail_to_boundary(
                            post_state, s_dst, join_op.output_shape, cm, net, L
                        )
                        if tail_cost == float("inf"):
                            continue
                        total = main_cost + skip_cost + align_cost + join_est.latency_ms + tail_cost
                        if total < best.cost_ms:
                            # Reconstruct the step list from the
                            # Dijkstra predecessor maps + align_steps +
                            # join step + tail_steps.
                            main_steps = _backtrack(main_dist, main_end, main_src)
                            skip_steps = _backtrack(skip_dist, skip_end, main_src)
                            join_edge = TransitionEdge(
                                src=State(-1, add_domain, post_state.l + join_op.he_level_delta if add_domain == "HE" else L),
                                dst=post_state,
                                cost_ms=join_est.latency_ms,
                                action="he_exec" if add_domain == "HE" else "mpc_exec",
                                detail=f"{join_op.node_id}:{join_op.op_type}",
                            )
                            steps = (
                                main_steps
                                + [TransitionEdge(main_src, main_src, 0.0, "__branch_skip__", "")]
                                + skip_steps
                                + align_steps
                                + [join_edge]
                                + tail_steps
                            )
                            best = RegionPlan(cost_ms=total, steps=steps)
            table[(s_src.key(), s_dst.key())] = best
    return RegionTransferTable(region_id=region.region_id, table=table)


def _backtrack(
    dist: Dict[Tuple[int, Domain, int], Tuple[float, Optional[TransitionEdge]]],
    end_state: State,
    src_state: State,
) -> List[TransitionEdge]:
    steps: List[TransitionEdge] = []
    cur = end_state.key()
    guard = 0
    while cur != src_state.key():
        info = dist.get(cur)
        if info is None or info[1] is None:
            break
        steps.append(info[1])
        cur = info[1].src.key()
        guard += 1
        if guard > 10000:
            break
    return list(reversed(steps))


def _align_for_join(
    main_end: State,
    skip_end: State,
    add_domain: Domain,
    join_op: ChainOp,
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
    bootstrap_plan: str = "none",
) -> Tuple[float, List[TransitionEdge], int]:
    """Compute alignment cost + common level to execute the join.

    Residual_Add at a join requires both operands to share the domain
    where the add happens. For ``add_domain == HE`` we also need
    matching HE levels. The optimal strategy is picked by enumerating:

    - ``bootstrap_plan="none"``: level-drop the higher branch to the
      lower (free under CKKS encoding).
    - ``bootstrap_plan="main"``: bootstrap the main branch to L before
      add, so the common level is ``min(L, skip_end.l) = skip_end.l``.
    - ``bootstrap_plan="skip"``: bootstrap the skip branch to L before
      add, so the common level is ``main_end.l``.
    - ``bootstrap_plan="both"``: bootstrap both to L; common level L.

    The caller (``compute_region_transfer``) tries every plan and
    takes the minimum, ensuring exhaustive enumeration of alignment
    strategies. ``bootstrap_plan`` is a no-op when ``add_domain=='MPC''
    '' (MPC has no ciphertext levels).
    """
    steps: List[TransitionEdge] = []
    cost = 0.0

    def _conv(x: State, to: Domain) -> Tuple[float, State]:
        if x.d == to:
            return 0.0, x
        est = cm.estimate_conversion(x.d, to, join_op.input_shape, net)
        new = State(x.i, to, L if to == "HE" else L)
        return est.latency_ms, new

    c_m, m = _conv(main_end, add_domain)
    c_s, s = _conv(skip_end, add_domain)
    cost += c_m + c_s
    if c_m > 0:
        steps.append(
            TransitionEdge(main_end, m, c_m,
                           "he2mpc" if add_domain == "MPC" else "mpc2he",
                           f"align@join shape={join_op.input_shape}")
        )
    if c_s > 0:
        steps.append(
            TransitionEdge(skip_end, s, c_s,
                           "he2mpc" if add_domain == "MPC" else "mpc2he",
                           f"align@join shape={join_op.input_shape}")
        )
    if add_domain == "MPC":
        # bootstrap_plan irrelevant for MPC.
        return cost, steps, L

    # HE side: try the requested bootstrap strategy.
    if bootstrap_plan == "main" and m.l < L:
        bs_est = cm.estimate_bootstrap(net)
        if bs_est.resolution == "infeasible":
            return float("inf"), [], -1
        cost += bs_est.latency_ms
        m2 = State(m.i, "HE", L)
        steps.append(TransitionEdge(m, m2, bs_est.latency_ms, "bootstrap",
                                    "pre-add bootstrap main"))
        m = m2
    if bootstrap_plan == "skip" and s.l < L:
        bs_est = cm.estimate_bootstrap(net)
        if bs_est.resolution == "infeasible":
            return float("inf"), [], -1
        cost += bs_est.latency_ms
        s2 = State(s.i, "HE", L)
        steps.append(TransitionEdge(s, s2, bs_est.latency_ms, "bootstrap",
                                    "pre-add bootstrap skip"))
        s = s2
    if bootstrap_plan == "both":
        bs_est = cm.estimate_bootstrap(net)
        if bs_est.resolution == "infeasible":
            return float("inf"), [], -1
        # Only bootstrap a branch if it needs it.
        if m.l < L:
            cost += bs_est.latency_ms
            m2 = State(m.i, "HE", L)
            steps.append(TransitionEdge(m, m2, bs_est.latency_ms, "bootstrap",
                                        "pre-add bootstrap main"))
            m = m2
        if s.l < L:
            cost += bs_est.latency_ms
            s2 = State(s.i, "HE", L)
            steps.append(TransitionEdge(s, s2, bs_est.latency_ms, "bootstrap",
                                        "pre-add bootstrap skip"))
            s = s2

    common_l = min(m.l, s.l)
    return cost, steps, common_l


def _tail_to_boundary(
    post: State,
    target: State,
    shape: Shape,
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
) -> Tuple[float, List[TransitionEdge]]:
    """Cheapest 0-1 transitions from post to target (conversion +/- bootstrap)."""
    if post.d == target.d and post.l >= target.l:
        # We can always drop level for free (or no-op if same).
        return 0.0, []
    cost = 0.0
    steps: List[TransitionEdge] = []
    cur = post
    if cur.d != target.d:
        est = cm.estimate_conversion(cur.d, target.d, shape, net)
        if est.resolution == "infeasible":
            return float("inf"), []
        cost += est.latency_ms
        cur = State(cur.i, target.d, L if target.d == "HE" else L)
        steps.append(
            TransitionEdge(
                post, cur, est.latency_ms,
                "he2mpc" if target.d == "MPC" else "mpc2he",
                f"tail shape={shape}",
            )
        )
    if cur.d == "HE" and cur.l < target.l:
        est = cm.estimate_bootstrap(net)
        if est.resolution == "infeasible":
            return float("inf"), []
        cost += est.latency_ms
        nxt = State(cur.i, "HE", L)
        steps.append(TransitionEdge(cur, nxt, est.latency_ms, "bootstrap", ""))
        cur = nxt
    return cost, steps

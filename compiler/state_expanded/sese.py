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
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode

from .cost_model import NetworkSetting, StateExpandedCostModel
from .profile_schema import HE_LEVEL_BUDGET, LatencyProfile
from .state_graph import level_out, ChainOp, State, TransitionEdge, outgoing

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

    topo = _topo_index(graph, out_adj, in_adj)
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
                path_a = _descendants_in_order(a, out_adj, topo)
                path_b = _descendants_in_order(b, out_adj, topo)
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
                anc = _ancestors(join, in_adj)
                ma = [n for n in _prefix_up_to(path_a, join) if n in anc]
                mb = [n for n in _prefix_up_to(path_b, join) if n in anc]
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


def _topo_index(graph: OperatorGraph, out_adj, in_adj) -> Dict[str, int]:
    indeg = {n.node_id: len(in_adj[n.node_id]) for n in graph.nodes}
    ready = sorted(n for n, k in indeg.items() if k == 0)
    order: Dict[str, int] = {}
    while ready:
        u = ready.pop(0)
        order[u] = len(order)
        for v in out_adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort()
    return order


def _descendants_in_order(start: str, out_adj: Dict[str, List[str]], topo: Dict[str, int]) -> List[str]:
    """``start`` and every node reachable from it, in topological order (nested
    branches must not be cut off before the join, as a depth-first order would)."""
    seen = {start}
    stack = [start]
    while stack:
        for child in out_adj.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return sorted(seen, key=lambda n: topo.get(n, 0))


def _ancestors(node: str, in_adj: Dict[str, List[str]]) -> set:
    seen = set()
    stack = [node]
    while stack:
        for p in in_adj.get(stack.pop(), []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


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
    node_ids: List[str], node_map: Dict[str, OperatorNode], level_deltas: Dict[str, int],
    preds: Optional[Dict[str, List[str]]] = None,
) -> List[ChainOp]:
    out: List[ChainOp] = []
    for nid in node_ids:
        n = node_map[nid]
        p = (preds or {}).get(nid, [])
        out.append(
            ChainOp(
                node_id=n.node_id,
                op_type=n.op_type,
                input_shape=n.input_shape,
                output_shape=n.output_shape,
                he_level_delta=level_deltas.get(n.op_type, 0),
                he_level_out=level_out(level_deltas, n.op_type),
                edge_shape=_edge_shape_attr(n),
                pred=p[0] if len(p) == 1 else None,
            )
        )
    return out


def _edge_shape_attr(n: OperatorNode) -> Optional[Shape]:
    s = (n.attributes or {}).get("in_edge_shape")
    return tuple(s) if s else None


# A branch is a sequence of operators and nested regions: a nested region's fork runs as an
# operator, then the region (interior and join) is one step priced by its transfer table.
@dataclass
class NestedRegion:
    region: "SeseRegion"
    table: "RegionTransferTable"


def _branch_elements(path: List[str], node_map, level_deltas, nested: Dict[str, NestedRegion],
                     preds: Optional[Dict[str, List[str]]] = None) -> list:
    out: list = []
    inside: set = set()
    for nid in path:
        if nid in inside:
            continue
        out.extend(_build_chain([nid], node_map, level_deltas, preds))
        if nid in nested:
            q = nested[nid]
            out.append(q)
            inside |= set(q.region.main_path) | set(q.region.skip_path) | {q.region.join}
    return out


def dijkstra_elements(source: State, elements: list, cm, net, L: int, boundary_states: List[State]):
    """dijkstra_chain over operators and nested regions; dist values are
    (cost, edge, region plan or None)."""
    dist = {source.key(): (0.0, None, None)}
    pq = [(0.0, source.i, source.d, source.l)]
    while pq:
        cost, i, d, l = heapq.heappop(pq)
        if dist[(i, d, l)][0] < cost - 1e-12 or i == len(elements):
            continue
        el = elements[i]
        if isinstance(el, NestedRegion):
            for s_dst in boundary_states:
                plan = el.table.lookup(State(0, d, l), s_dst)
                if plan.cost_ms == float("inf"):
                    continue
                key, ncost = (i + 1, s_dst.d, s_dst.l), cost + plan.cost_ms
                if key not in dist or ncost + 1e-12 < dist[key][0]:
                    edge = TransitionEdge(State(i, d, l), State(*key), plan.cost_ms, "sese", el.region.region_id)
                    dist[key] = (ncost, edge, plan)
                    heapq.heappush(pq, (ncost, *key))
            continue
        for e in outgoing(State(0, d, l), [el], cm, net, L):
            ni = i + 1 if e.action in ("he_exec", "mpc_exec") else i
            key, ncost = (ni, e.dst.d, e.dst.l), cost + e.cost_ms
            if key not in dist or ncost + 1e-12 < dist[key][0]:
                dist[key] = (ncost, TransitionEdge(State(i, d, l), State(*key), e.cost_ms, e.action, e.detail), None)
                heapq.heappush(pq, (ncost, *key))
    return dist


def _backtrack_elements(dist, end_state: State, src_state: State) -> List[TransitionEdge]:
    chunks: List[List[TransitionEdge]] = []
    cur = end_state.key()
    while cur != src_state.key():
        _c, edge, plan = dist.get(cur, (0.0, None, None))
        if edge is None:
            break
        chunks.append(list(plan.steps) if plan is not None else [edge])
        cur = edge.src.key()
    return [e for chunk in reversed(chunks) for e in chunk]


def _explore_branch(src: State, elements: list, cm, net, L, boundary_states):
    """End states of a branch with their costs, and a function that returns the steps to one."""
    if all(isinstance(e, ChainOp) for e in elements):
        dist = dijkstra_chain(src, elements, cm, net, L)
        ends = [(State(len(elements), d, l), c) for (i, d, l), (c, _) in dist.items() if i == len(elements)]
        return ends, (lambda end: _backtrack(dist, end, src))
    dist = dijkstra_elements(src, elements, cm, net, L, boundary_states)
    ends = [(State(len(elements), d, l), c) for (i, d, l), (c, _e, _p) in dist.items() if i == len(elements)]
    return ends, (lambda end: _backtrack_elements(dist, end, src))


def compute_region_transfer(
    region: SeseRegion,
    graph: OperatorGraph,
    level_deltas: Dict[str, int],
    cm: StateExpandedCostModel,
    net: NetworkSetting,
    L: int,
    boundary_states: List[State],
    nested: Optional[Dict[str, NestedRegion]] = None,
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
    nested = nested or {}
    preds: Dict[str, List[str]] = {}
    for e in graph.edges:
        preds.setdefault(e.dst, []).append(e.src)
    main_chain = _branch_elements(region.main_path, node_map, level_deltas, nested, preds)
    skip_chain = _branch_elements(region.skip_path, node_map, level_deltas, nested, preds)
    # the tensors arriving at the join on each branch (they differ when the join is not an
    # Add, e.g. Attn*V receives the attention probabilities and V)
    edge_shapes = {(e.src, e.dst): tuple(e.tensor_shape) for e in graph.edges}
    main_last = region.main_path[-1] if region.main_path else region.fork
    skip_last = region.skip_path[-1] if region.skip_path else region.fork
    main_shape = edge_shapes.get((main_last, region.join), tuple(node_map[region.join].input_shape))
    skip_shape = edge_shapes.get((skip_last, region.join), tuple(node_map[region.join].input_shape))

    join_node = node_map[region.join]
    join_op = ChainOp(
        node_id=join_node.node_id,
        op_type=join_node.op_type,
        input_shape=join_node.input_shape,
        output_shape=join_node.output_shape,
        he_level_delta=level_deltas.get(join_node.op_type, 0),
        he_level_out=level_out(level_deltas, join_node.op_type),
    )

    table: Dict[Tuple[Tuple[int, Domain, int], Tuple[int, Domain, int]], RegionPlan] = {}

    for s_src in boundary_states:
        # --- explore main path from s_src ---
        main_src = State(0, s_src.d, s_src.l)
        main_ends, main_steps_to = _explore_branch(main_src, main_chain, cm, net, L, boundary_states)
        # --- explore skip path from s_src (both branches share the fork) ---
        skip_ends, skip_steps_to = _explore_branch(main_src, skip_chain, cm, net, L, boundary_states)

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
                            bootstrap_plan=bs_plan, main_shape=main_shape, skip_shape=skip_shape,
                            main_edge=f"{main_last}->{region.join}", skip_edge=f"{skip_last}->{region.join}",
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
                        if add_domain == "HE" and common_l < join_op.he_level_delta:
                            continue
                        post_state = State(
                            0,
                            add_domain,
                            (join_op.he_level_out if join_op.he_level_out is not None
                             else common_l - join_op.he_level_delta) if add_domain == "HE" else L,
                        )
                        # Transition from post_state to s_dst: allow at
                        # most one conversion (HE<->MPC) and bootstrap.
                        tail_cost, tail_steps = _tail_to_boundary(
                            post_state, s_dst, join_op.output_shape, cm, net, L, edge=f"{region.join}->*"
                        )
                        if tail_cost == float("inf"):
                            continue
                        total = main_cost + skip_cost + align_cost + join_est.latency_ms + tail_cost
                        if total < best.cost_ms:
                            # Reconstruct the step list from the
                            # Dijkstra predecessor maps + align_steps +
                            # join step + tail_steps.
                            main_steps = main_steps_to(main_end)
                            skip_steps = skip_steps_to(skip_end)
                            join_edge = TransitionEdge(
                                src=State(-1, add_domain, common_l if add_domain == "HE" else L),
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
    main_shape: Optional[Shape] = None,
    skip_shape: Optional[Shape] = None,
    main_edge: str = "",
    skip_edge: str = "",
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
    main_shape = tuple(main_shape or join_op.input_shape)
    skip_shape = tuple(skip_shape or join_op.input_shape)

    def _conv(x: State, to: Domain, shape, edge) -> Tuple[float, float, State]:
        """(bootstrap before it, the conversion, the state after); the bootstrap step, if any, is
        recorded here: the hardened HE->MPC conversion needs levels left."""
        if x.d == to:
            return 0.0, 0.0, x
        pre = 0.0
        if x.d == "HE" and x.l < cm.he_to_mpc_min_level:
            bs = cm.estimate_bootstrap_at(net, shape, x.l, edge)
            if bs.resolution == "infeasible":
                return float("inf"), 0.0, x
            steps.append(TransitionEdge(x, State(x.i, "HE", L), bs.latency_ms, "bootstrap",
                                        f"pre-conversion edge={edge}"))
            pre, x = bs.latency_ms, State(x.i, "HE", L)
        est = cm.estimate_conversion(x.d, to, shape, net)
        return pre, est.latency_ms, State(x.i, to, L)

    b_m, c_m, m = _conv(main_end, add_domain, main_shape, main_edge)
    b_s, c_s, s = _conv(skip_end, add_domain, skip_shape, skip_edge)
    if math.isinf(b_m) or math.isinf(b_s):
        return float("inf"), [], -1
    cost += b_m + c_m + b_s + c_s
    if main_end.d != add_domain:
        steps.append(
            TransitionEdge(main_end, m, c_m,
                           "he2mpc" if add_domain == "MPC" else "mpc2he",
                           f"align@join shape={main_shape} edge={main_edge}")
        )
    if skip_end.d != add_domain:
        steps.append(
            TransitionEdge(skip_end, s, c_s,
                           "he2mpc" if add_domain == "MPC" else "mpc2he",
                           f"align@join shape={skip_shape} edge={skip_edge}")
        )
    if add_domain == "MPC":
        # bootstrap_plan irrelevant for MPC.
        return cost, steps, L

    # HE side: try the requested bootstrap strategy.
    if bootstrap_plan == "main" and m.l < L:
        bs_est = cm.estimate_bootstrap_at(net, main_shape, m.l, main_edge)
        if bs_est.resolution == "infeasible":
            return float("inf"), [], -1
        cost += bs_est.latency_ms
        m2 = State(m.i, "HE", L)
        steps.append(TransitionEdge(m, m2, bs_est.latency_ms, "bootstrap",
                                    f"pre-add bootstrap main edge={main_edge}"))
        m = m2
    if bootstrap_plan == "skip" and s.l < L:
        bs_est = cm.estimate_bootstrap_at(net, skip_shape, s.l, skip_edge)
        if bs_est.resolution == "infeasible":
            return float("inf"), [], -1
        cost += bs_est.latency_ms
        s2 = State(s.i, "HE", L)
        steps.append(TransitionEdge(s, s2, bs_est.latency_ms, "bootstrap",
                                    f"pre-add bootstrap skip edge={skip_edge}"))
        s = s2
    if bootstrap_plan == "both":
        bs_m = cm.estimate_bootstrap_at(net, main_shape, m.l if m.l < L else None, main_edge)
        bs_s = cm.estimate_bootstrap_at(net, skip_shape, s.l if s.l < L else None, skip_edge)
        if "infeasible" in (bs_m.resolution, bs_s.resolution):
            return float("inf"), [], -1
        # Only bootstrap a branch if it needs it.
        if m.l < L:
            cost += bs_m.latency_ms
            m2 = State(m.i, "HE", L)
            steps.append(TransitionEdge(m, m2, bs_m.latency_ms, "bootstrap",
                                        f"pre-add bootstrap main edge={main_edge}"))
            m = m2
        if s.l < L:
            cost += bs_s.latency_ms
            s2 = State(s.i, "HE", L)
            steps.append(TransitionEdge(s, s2, bs_s.latency_ms, "bootstrap",
                                        f"pre-add bootstrap skip edge={skip_edge}"))
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
    edge: str = "",
) -> Tuple[float, List[TransitionEdge]]:
    """Cheapest 0-1 transitions from post to target (conversion +/- bootstrap).
    ``edge`` ("<join>->*"): they apply to the join's output, for all its consumers."""
    if post.d == target.d and post.l >= target.l:
        # We can always drop level for free (or no-op if same).
        return 0.0, []
    cost = 0.0
    steps: List[TransitionEdge] = []
    cur = post
    if cur.d == "HE" and target.d == "MPC" and cur.l < cm.he_to_mpc_min_level:   # the conversion needs levels
        est = cm.estimate_bootstrap_at(net, shape, cur.l, edge)
        if est.resolution == "infeasible":
            return float("inf"), []
        cost += est.latency_ms
        nxt = State(cur.i, "HE", L)
        steps.append(TransitionEdge(cur, nxt, est.latency_ms, "bootstrap", f"pre-conversion edge={edge}"))
        cur = nxt
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
                f"tail shape={shape} edge={edge}",
            )
        )
    if cur.d == "HE" and cur.l < target.l:
        est = cm.estimate_bootstrap_at(net, shape, cur.l, edge)
        if est.resolution == "infeasible":
            return float("inf"), []
        cost += est.latency_ms
        nxt = State(cur.i, "HE", L)
        steps.append(TransitionEdge(cur, nxt, est.latency_ms, "bootstrap", f"edge={edge}"))
        cur = nxt
    return cost, steps

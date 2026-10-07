"""End-to-end planner (paper §4.2.3 "global plan reconstruction").

Pipeline
--------
1. Detect SESE regions on G = (V, E).
2. Linearise G by replacing each region with a macro-node. Between
   macro-nodes the topology is a linear chain of the "spine" operators
   (e.g. ``qk -> sm -> av -> op`` is inside R1; after R1 we continue
   with ``ln1 -> ...``).
3. Pre-compute each region's transfer function T_R(s_src, s_dst) for
   every boundary state pair.
4. Run Dijkstra on the aggregate graph (linear spine + macro edges).
5. Backtrack: expand each macro edge by replaying the stored region
   plan; emit a flat per-node execution plan.

The output is ``CompiledPlan`` which contains
(a) the per-node domain assignment + bootstrap/conversion decisions,
(b) the total cost breakdown (node, conversion, bootstrap),
(c) a pretty-printed sequence of actions for inspection.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode

from .cost_model import NetworkSetting, StateExpandedCostModel
from .profile_schema import HE_LEVEL_BUDGET, LatencyProfile
from .sese import (
    RegionPlan,
    RegionTransferTable,
    SeseRegion,
    NestedRegion,
    compute_region_transfer,
    dijkstra_chain,
    find_sese_regions,
    _backtrack,
    _edge_shape_attr,
)
from .state_graph import level_out, ChainOp, State, TransitionEdge

Domain = str


# ---------- Level deltas hard-wired from method_he_nexus modules ----------


DEFAULT_LEVEL_DELTAS: Dict[str, int] = {
    "Linear_QKV": 1,
    "Attention_QK_MatMul": 1,
    "Softmax": 8,
    "Attention_V_MatMul": 1,
    "Out_Projection": 1,
    "Residual_Add": 0,
    "LayerNorm": 3,
    "FFN_Linear_1": 1,
    "GeLU": 4,
    "FFN_Linear_2": 1,
}


@dataclass
class CompiledPlan:
    network: NetworkSetting
    node_assignment: Dict[str, Domain]
    total_cost_ms: float
    node_cost_ms: float
    conversion_cost_ms: float
    bootstrap_cost_ms: float
    steps: List[TransitionEdge]
    region_ids: List[str] = field(default_factory=list)
    # Populated by ``compile_plan_safe``: which strategy produced the
    # chosen cost ('compiler', 'all_HE', 'all_MPC', 'linHE_nlMPC',
    # 'linMPC_nlHE', or 'attnMPC_ffnHE'). 'compiler' means the
    # state-expanded plan was optimal; any other value means a static
    # baseline would have been at least as cheap.
    strategy_used: str = "compiler"

    def pretty(self) -> str:
        lines = [f"[plan] network={self.network.label()} total={self.total_cost_ms:.3f} ms"]
        lines.append(
            f"  node_compute={self.node_cost_ms:.3f}  "
            f"conversion={self.conversion_cost_ms:.3f}  "
            f"bootstrap={self.bootstrap_cost_ms:.3f}"
        )
        for nid, d in self.node_assignment.items():
            lines.append(f"    {nid:10s} -> {d}")
        for step in self.steps:
            lines.append(
                f"  [{step.action}] {step.detail} ({step.cost_ms:.3f} ms)"
                f"  {step.src.key() if step.src else ''} -> {step.dst.key() if step.dst else ''}"
            )
        return "\n".join(lines)


# ---------- Spine linearisation ----------


def _linearise(graph: OperatorGraph, regions: List[SeseRegion]) -> List[Tuple[str, Optional[SeseRegion]]]:
    """Return a linear spine of (node_id, region_or_None) entries.

    Each entry is either a plain operator node (region=None) or a
    region boundary (``fork_id``, region). For each region the
    ``fork`` is executed as the region's boundary — the inside of the
    region is replaced by a macro-edge that uses the transfer function.
    """
    out_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    in_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        out_adj[e.src].append(e.dst)
        in_adj[e.dst].append(e.src)

    # Topological order.
    indeg = {n.node_id: len(in_adj[n.node_id]) for n in graph.nodes}
    ready = [n.node_id for n in graph.nodes if indeg[n.node_id] == 0]
    topo: List[str] = []
    while ready:
        ready.sort()
        u = ready.pop(0)
        topo.append(u)
        for v in out_adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)

    region_by_fork = {r.fork: r for r in regions}
    region_nodes_inside: Dict[str, set] = {}
    for r in regions:
        region_nodes_inside[r.region_id] = set(r.main_path) | set(r.skip_path)

    spine: List[Tuple[str, Optional[SeseRegion]]] = []
    skip = set()
    for nid in topo:
        if nid in skip:
            continue
        if nid in region_by_fork:
            r = region_by_fork[nid]
            # First execute the fork as an ordinary operator, THEN enter
            # the region macro. The region's transfer function covers
            # [interior] + join only; the fork operator runs outside.
            spine.append((nid, None))
            spine.append((nid, r))
            # skip every interior node; the region_plan will replay them
            skip.update(region_nodes_inside[r.region_id])
            continue
        spine.append((nid, None))
    return spine


# ---------- Planner ----------


def compile_plan(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
    level_deltas: Optional[Dict[str, int]] = None,
    initial_domain: Optional[Domain] = None,
    final_domain: Optional[Domain] = None,
) -> CompiledPlan:
    """Compile one graph into one execution plan under ``net``."""
    level_deltas = level_deltas or DEFAULT_LEVEL_DELTAS
    L = profile.he_level_budget
    cm = StateExpandedCostModel(profile)

    regions = find_sese_regions(graph)
    spine = _linearise(graph, regions)

    node_map = {n.node_id: n for n in graph.nodes}
    graph_preds: Dict[str, List[str]] = {}
    for e in graph.edges:
        graph_preds.setdefault(e.dst, []).append(e.src)
    # Build a linear chain of ChainOps that includes the fork nodes
    # (executed as ordinary operators on the spine); each region is
    # represented by a *macro-edge* inserted *between* its fork and the
    # next spine entry that is the join node.
    #
    # We store per-spine-index the possible outgoing edges:
    # - for a plain operator i: standard state transitions over that
    #   single operator.
    # - for a fork of region r: a macro transition from (i, d_src, l_src)
    #   to (i+1, d_dst, l_dst) with cost T_r(s_src, s_dst), where the
    #   "next" position is the join (logically collapsed into a single
    #   advance).

    # Pre-compute region transfer tables over all boundary states.
    boundary_states: List[State] = [State(0, "HE", l) for l in range(L + 1)] + [
        State(0, "MPC", L)
    ]
    transfer_tables: Dict[str, RegionTransferTable] = {}
    # nested regions first (fewest interior nodes); a region's branches use the tables of the
    # regions nested in them
    interior = {r.region_id: set(r.main_path) | set(r.skip_path) for r in regions}
    for r in sorted(regions, key=lambda r: (len(interior[r.region_id]), r.region_id)):
        nested = {q.fork: NestedRegion(q, transfer_tables[q.region_id]) for q in regions
                  if q is not r and q.fork in interior[r.region_id] and q.region_id in transfer_tables}
        tt = compute_region_transfer(r, graph, level_deltas, cm, net, L, boundary_states, nested)
        transfer_tables[r.region_id] = tt

    # Walk the spine and compute cost-aware shortest path.
    # Map spine entries to (advance_kind, payload).
    spine_entries: List[Tuple[str, Optional[SeseRegion]]] = spine
    # The spine as a chain from index 0..N where each step i -> i+1
    # executes either a plain op (fork not a region fork) or the region
    # macro (region). For a region, we skip the join in the spine since
    # we collapse fork+interior+join into one macro step.
    # Remove join nodes from spine (they are consumed inside the region).
    # A join is executed inside its region's macro step, so its plain-op
    # entry is dropped. When the join is also the fork of the next region
    # (pre-LN blocks: Add -> LN -> FFN -> Add), its macro entry for that next
    # region must be kept, otherwise the whole next region is never planned.
    join_ids = {r.join for r in regions}
    collapsed = [(nid, reg) for (nid, reg) in spine_entries
                 if not (nid in join_ids and reg is None)]

    # Standard Dijkstra over a state graph sized O((|collapsed|+1) * L).
    # Keys: (spine_idx, domain, level).
    # For plain op steps we use ``outgoing`` once per step.
    # For macro steps we expand all (s_src, s_dst) entries in T_r.

    import heapq as hq
    INF = float("inf")
    dist: Dict[Tuple[int, Domain, int], Tuple[float, Optional[TransitionEdge], Optional[RegionPlan]]] = {}
    # Source states: try both HE and MPC starting points; Dijkstra from
    # a virtual super-source finds the overall minimum.
    init_candidates: List[Domain] = (
        [initial_domain] if initial_domain is not None else ["HE", "MPC"]
    )
    pq: List[Tuple[float, int, Domain, int]] = []
    for d0 in init_candidates:
        key = (0, d0, L)
        dist[key] = (0.0, None, None)
        hq.heappush(pq, (0.0, 0, d0, L))

    while pq:
        cost, i, d, l = hq.heappop(pq)
        if dist.get((i, d, l), (INF, None, None))[0] < cost - 1e-12:
            continue
        if i == len(collapsed):
            continue
        nid, region = collapsed[i]
        current_state = State(i, d, l)
        if region is None:
            # Plain op: use state_graph.outgoing on a 1-op chain.
            node = node_map[nid]
            chain_op = ChainOp(
                node_id=node.node_id,
                op_type=node.op_type,
                input_shape=node.input_shape,
                output_shape=node.output_shape,
                he_level_delta=level_deltas.get(node.op_type, 0),
                he_level_out=level_out(level_deltas, node.op_type),
                edge_shape=_edge_shape_attr(node),
                pred=graph_preds[nid][0] if len(graph_preds.get(nid, [])) == 1 else None,
            )
            # Allow conversions/bootstrap at this position before exec.
            from .state_graph import outgoing as _out
            fake_state = State(0, d, l)
            edges = _out(fake_state, [chain_op], cm, net, L)
            for e in edges:
                # Forward edges either progress the op or stay (conv/bootstrap).
                if e.action in {"he_exec", "mpc_exec"}:
                    ni, nd, nl = i + 1, e.dst.d, e.dst.l
                else:
                    ni, nd, nl = i, e.dst.d, e.dst.l
                ncost = cost + e.cost_ms
                nkey = (ni, nd, nl)
                if nkey not in dist or ncost + 1e-12 < dist[nkey][0]:
                    # Rewrap the edge in spine coordinates.
                    wrap = TransitionEdge(
                        src=State(i, d, l),
                        dst=State(ni, nd, nl),
                        cost_ms=e.cost_ms,
                        action=e.action,
                        detail=e.detail,
                    )
                    dist[nkey] = (ncost, wrap, None)
                    hq.heappush(pq, (ncost, ni, nd, nl))
        else:
            tt = transfer_tables[region.region_id]
            for s_dst in boundary_states:
                plan = tt.lookup(State(0, d, l), s_dst)
                if plan.cost_ms == INF:
                    continue
                ni, nd, nl = i + 1, s_dst.d, s_dst.l
                ncost = cost + plan.cost_ms
                nkey = (ni, nd, nl)
                if nkey not in dist or ncost + 1e-12 < dist[nkey][0]:
                    wrap = TransitionEdge(
                        src=State(i, d, l),
                        dst=State(ni, nd, nl),
                        cost_ms=plan.cost_ms,
                        action="sese",
                        detail=region.region_id,
                    )
                    dist[nkey] = (ncost, wrap, plan)
                    hq.heappush(pq, (ncost, ni, nd, nl))

    # Select the best terminal state at spine index len(collapsed).
    final_idx = len(collapsed)
    terminal_domains: List[Domain] = (
        [final_domain] if final_domain is not None else ["HE", "MPC"]
    )
    best: Optional[Tuple[float, Tuple[int, Domain, int]]] = None
    for (i, d, l), (c, _e, _p) in dist.items():
        if i != final_idx:
            continue
        if d not in terminal_domains:
            continue
        if best is None or c < best[0]:
            best = (c, (i, d, l))
    if best is None:
        raise RuntimeError("No feasible plan reached the terminal state.")

    # Backtrack until we reach a true source state (any (0, d, L) with
    # predecessor None).
    cur = best[1]
    raw_steps: List[TransitionEdge] = []
    region_plans_used: List[RegionPlan] = []
    guard = 0
    while True:
        info = dist.get(cur)
        if info is None or info[1] is None:
            break
        _c, edge, plan = info
        raw_steps.append(edge)
        if plan is not None:
            region_plans_used.append(plan)
        cur = edge.src.key()
        guard += 1
        if guard > 10000:
            break
    raw_steps.reverse()
    region_plans_used.reverse()

    # Flatten macro steps.
    flat_steps: List[TransitionEdge] = []
    node_assignment: Dict[str, Domain] = {}
    node_cost = 0.0
    conv_cost = 0.0
    bs_cost = 0.0
    region_ids: List[str] = []
    region_plan_iter = iter(region_plans_used)

    for e in raw_steps:
        if e.action == "sese":
            rp = next(region_plan_iter)
            region_ids.append(e.detail)
            for inner in rp.steps:
                if inner.action == "__branch_skip__":
                    continue
                flat_steps.append(inner)
                _accumulate(inner, node_assignment, add_node=add_node_cost_cb)
            # Accumulate costs from the flattened steps instead of macro cost.
        else:
            flat_steps.append(e)
            _accumulate(e, node_assignment, add_node=add_node_cost_cb)

    # Re-sum from flattened edges for verification.
    for s in flat_steps:
        if s.action in {"he_exec", "mpc_exec"}:
            node_cost += s.cost_ms
        elif s.action in {"he2mpc", "mpc2he"}:
            conv_cost += s.cost_ms
        elif s.action == "bootstrap":
            bs_cost += s.cost_ms

    total = node_cost + conv_cost + bs_cost

    return CompiledPlan(
        network=net,
        node_assignment=node_assignment,
        total_cost_ms=total,
        node_cost_ms=node_cost,
        conversion_cost_ms=conv_cost,
        bootstrap_cost_ms=bs_cost,
        steps=flat_steps,
        region_ids=region_ids,
    )


def add_node_cost_cb(nid: str, domain: Domain, cost: float) -> None:
    pass


def _accumulate(step: TransitionEdge, assignment: Dict[str, Domain], add_node) -> None:
    if step.action in {"he_exec", "mpc_exec"}:
        # parse nid from detail "nid:op"
        detail = step.detail or ""
        nid = detail.split(":", 1)[0] if ":" in detail else detail
        assignment[nid] = "HE" if step.action == "he_exec" else "MPC"


# ---------- Baselines ----------


def evaluate_uniform_domain(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
    domain: Domain,
    level_deltas: Optional[Dict[str, int]] = None,
) -> float:
    """All-HE or all-MPC reference cost.

    For ``MPC`` this is just the sum of MPC operator costs.

    For ``HE`` we account for mandatory bootstrapping *residual-aware*:
    each node's incoming HE level is the minimum over its predecessors'
    output levels (CKKS adds require matched-level operands, and after
    an add the result sits at the minimum common level). If that
    minimum is below the node's delta, a bootstrap is inserted before
    the node. This matches what ``_materialise_static_plan`` computes
    and is the basis of the ``compile_plan_safe`` guarantee.
    """
    level_deltas = level_deltas or DEFAULT_LEVEL_DELTAS
    if domain == "MPC":
        # MPC has no ciphertext level; just sum node costs.
        cm = StateExpandedCostModel(profile)
        total = 0.0
        for n in graph.nodes:
            est = cm.estimate_operator(n.op_type, domain, n.input_shape, n.output_shape, net)
            if est.resolution == "infeasible":
                return float("inf")
            total += est.latency_ms
        return total

    # HE: delegate to the residual-aware materialiser.
    policy = {op: "HE" for op in level_deltas}
    plan = _materialise_static_plan(
        graph, profile, net, policy, "all_HE", level_deltas
    )
    return plan.total_cost_ms


def _evaluate_fixed_policy(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
    domain_of_op: Dict[str, Domain],
) -> float:
    """Evaluate an arbitrary fixed placement.

    ``domain_of_op`` maps op_type -> domain; any node whose op_type is
    not in the map falls back to HE (safe default for residual adds).
    """
    cm = StateExpandedCostModel(profile)
    total = 0.0
    domain_of: Dict[str, Domain] = {}
    for n in graph.nodes:
        d = domain_of_op.get(n.op_type, "HE")
        domain_of[n.node_id] = d
        est = cm.estimate_operator(n.op_type, d, n.input_shape, n.output_shape, net)
        if est.resolution == "infeasible":
            return float("inf")
        total += est.latency_ms
    for e in graph.edges:
        if domain_of[e.src] != domain_of[e.dst]:
            est = cm.estimate_conversion(
                domain_of[e.src], domain_of[e.dst], e.tensor_shape, net
            )
            total += est.latency_ms
    return total


# Named static-placement baselines. Each policy maps a BERT op_type to a
# fixed execution domain; each corresponds to a family of prior works.

POLICY_LINEAR_HE_NONLINEAR_MPC: Dict[str, Domain] = {
    # BumbleBee / BOLT / Cheetah-style: heavy linear on HE, nonlinear
    # approximations on MPC via dedicated protocols.
    "Linear_QKV":          "HE",
    "Attention_QK_MatMul": "HE",
    "Attention_V_MatMul":  "HE",
    "Out_Projection":      "HE",
    "FFN_Linear_1":        "HE",
    "FFN_Linear_2":        "HE",
    "Residual_Add":        "HE",
    "Softmax":             "MPC",
    "LayerNorm":           "MPC",
    "GeLU":                "MPC",
}

POLICY_LINEAR_MPC_NONLINEAR_HE: Dict[str, Domain] = {
    # CryptoNets / polynomial-approximation-style: nonlinearities
    # replaced by HE polynomials; linear ops run over secret shares to
    # avoid HE ciphertext slot blow-up on big matmuls.
    "Linear_QKV":          "MPC",
    "Attention_QK_MatMul": "MPC",
    "Attention_V_MatMul":  "MPC",
    "Out_Projection":      "MPC",
    "FFN_Linear_1":        "MPC",
    "FFN_Linear_2":        "MPC",
    "Residual_Add":        "MPC",
    "Softmax":             "HE",
    "LayerNorm":           "HE",
    "GeLU":                "HE",
}

POLICY_ATTN_MPC_FFN_HE: Dict[str, Domain] = {
    # "Attention@MPC, FFN@HE": attention has small matrices (BxSxS
    # scores) where MPC is competitive, while FFN has the large
    # 768<->3072 matmul which packs well on HE ciphertexts.
    "Linear_QKV":          "MPC",
    "Attention_QK_MatMul": "MPC",
    "Attention_V_MatMul":  "MPC",
    "Out_Projection":      "MPC",
    "Softmax":             "MPC",
    "Residual_Add":        "HE",
    "LayerNorm":           "HE",
    "FFN_Linear_1":        "HE",
    "GeLU":                "HE",
    "FFN_Linear_2":        "HE",
}


def evaluate_static_hybrid(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
) -> float:
    """Legacy API: returns the classic BumbleBee-style static hybrid."""
    return _evaluate_fixed_policy(graph, profile, net, POLICY_LINEAR_HE_NONLINEAR_MPC)


def evaluate_named_static_hybrids(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
) -> Dict[str, float]:
    """Return the set of named static baselines keyed by policy label.

    Keys:
      ``linHE_nlMPC``  : linear@HE, nonlinear@MPC  (BumbleBee, BOLT, Cheetah)
      ``linMPC_nlHE``  : linear@MPC, nonlinear@HE  (CryptoNets-style)
      ``attnMPC_ffnHE``: attention block@MPC, FFN block@HE
    """
    return {
        "linHE_nlMPC":   _evaluate_fixed_policy(graph, profile, net, POLICY_LINEAR_HE_NONLINEAR_MPC),
        "linMPC_nlHE":   _evaluate_fixed_policy(graph, profile, net, POLICY_LINEAR_MPC_NONLINEAR_HE),
        "attnMPC_ffnHE": _evaluate_fixed_policy(graph, profile, net, POLICY_ATTN_MPC_FFN_HE),
    }


# ---------- Safety-net compiler: guaranteed never slower than baselines ----------


def _materialise_static_plan(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
    domain_of_op: Dict[str, Domain],
    strategy_label: str,
    level_deltas: Dict[str, int],
) -> CompiledPlan:
    """Build a CompiledPlan for a fixed-placement baseline.

    Cost and assignment are computed from the same cost model. Level
    budget is respected: an HE op's inputs whose level is below what the
    op needs are bootstrapped first (one bootstrap per input edge), and
    so are HE outputs converted to MPC with fewer levels left than the
    conversion needs. No conversions are inserted beyond the edges
    implied by the fixed placement; every step names its edge. A source
    node with no policy (the model input) takes its first consumer's
    domain: the client provides it in that domain.
    """
    cm = StateExpandedCostModel(profile)
    L = profile.he_level_budget
    min_conv = cm.he_to_mpc_min_level

    # Topological order.
    indeg = {n.node_id: 0 for n in graph.nodes}
    out_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    in_adj: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        indeg[e.dst] += 1
        out_adj[e.src].append(e.dst)
        in_adj[e.dst].append(e.src)
    node_map = {n.node_id: n for n in graph.nodes}
    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    topo: List[str] = []
    while ready:
        u = ready.pop(0)
        topo.append(u)
        for v in out_adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)

    assignment: Dict[str, Domain] = {}
    steps: List[TransitionEdge] = []
    node_cost = 0.0
    conv_cost = 0.0
    bs_cost = 0.0
    he_level: Dict[str, int] = {nid: L for nid in topo}

    def infeasible() -> CompiledPlan:
        return CompiledPlan(
            network=net,
            node_assignment=assignment,
            total_cost_ms=float("inf"),
            node_cost_ms=float("inf"),
            conversion_cost_ms=conv_cost,
            bootstrap_cost_ms=bs_cost,
            steps=steps,
            strategy_used=strategy_label,
        )

    def bootstrap(shape, detail, level, producer) -> bool:
        nonlocal bs_cost
        bs_est = cm.estimate_bootstrap_at(net, shape, level, producer)
        if bs_est.resolution == "infeasible":
            return False
        bs_cost += bs_est.latency_ms
        steps.append(
            TransitionEdge(
                src=None, dst=None,
                cost_ms=bs_est.latency_ms,
                action="bootstrap",
                detail=detail,
            )  # type: ignore[arg-type]
        )
        return True

    for nid in topo:
        node = node_map[nid]
        d = domain_of_op.get(node.op_type) or domain_of_op.get(node.op_type.split(".", 1)[0])   # kernel variants
        if d is None:
            first = next((node_map[c] for c in out_adj[nid]), None) if not in_adj[nid] else None
            d = domain_of_op.get(first.op_type, "HE") if first is not None else "HE"
        assignment[nid] = d
        delta = level_deltas.get(node.op_type, 0)

        # Insert conversion on each incoming edge whose source has a
        # different domain.
        preds = in_adj[nid]
        preds_levels = []
        for p in preds:
            src_d = assignment.get(p)
            if src_d is None:
                # Shouldn't happen post-topo, but be defensive.
                continue
            edge_shape = next(
                (e.tensor_shape for e in graph.edges if e.src == p and e.dst == nid),
                node.input_shape,
            )
            if src_d != d:
                if src_d == "HE" and he_level[p] < min_conv:
                    if not bootstrap(edge_shape, f"pre-conversion edge={p}->{nid}", he_level[p], p):
                        return infeasible()
                est = cm.estimate_conversion(src_d, d, edge_shape, net)
                conv_cost += est.latency_ms
                steps.append(
                    TransitionEdge(
                        src=None, dst=None,
                        cost_ms=est.latency_ms,
                        action="he2mpc" if d == "MPC" else "mpc2he",
                        detail=f"{p}->{nid} shape={edge_shape} edge={p}->{nid}",
                    )  # type: ignore[arg-type]
                )
                # After conversion to HE, the receiving side has fresh
                # level L; after conversion to MPC, level is irrelevant.
                preds_levels.append(L)
            elif d == "HE" and he_level[p] < delta:
                # For HE ops, ensure sufficient level by bootstrapping the input.
                if not bootstrap(edge_shape, f"before {nid} edge={p}->{nid}", he_level[p], p):
                    return infeasible()
                preds_levels.append(L)
            else:
                preds_levels.append(he_level[p])

        if d == "HE":
            current_l = min(preds_levels) if preds_levels else L
            reset = level_out(level_deltas, node.op_type)
            he_level[nid] = reset if reset is not None else current_l - delta
        else:
            he_level[nid] = L

        # Node cost.
        est = cm.estimate_operator(node.op_type, d, node.input_shape, node.output_shape, net)
        if est.resolution == "infeasible":
            return infeasible()
        node_cost += est.latency_ms
        steps.append(
            TransitionEdge(
                src=None, dst=None,
                cost_ms=est.latency_ms,
                action="he_exec" if d == "HE" else "mpc_exec",
                detail=f"{nid}:{node.op_type}",
            )  # type: ignore[arg-type]
        )

    return CompiledPlan(
        network=net,
        node_assignment=assignment,
        total_cost_ms=node_cost + conv_cost + bs_cost,
        node_cost_ms=node_cost,
        conversion_cost_ms=conv_cost,
        bootstrap_cost_ms=bs_cost,
        steps=steps,
        strategy_used=strategy_label,
    )


def compile_plan_safe(
    graph: OperatorGraph,
    profile: LatencyProfile,
    net: NetworkSetting,
    level_deltas: Optional[Dict[str, int]] = None,
) -> CompiledPlan:
    """Guaranteed never-slower-than-any-static-baseline compiler.

    Runs the state-expanded compiler AND materialises five static
    baselines (all-HE, all-MPC, three static hybrids) under the same
    cost model. Returns the minimum-cost CompiledPlan with
    ``strategy_used`` indicating which strategy was chosen.

    If the state-expanded planner is globally optimal — as proved in
    OPTIMALITY.md — it will always win this comparison. The safety net
    makes the "never worse than a static baseline" property a
    first-class guarantee rather than a hope: if any tie-breaking or
    cost-model quirk would ever put a static baseline below the
    compiler's plan, this function returns that baseline.
    """
    level_deltas = level_deltas or DEFAULT_LEVEL_DELTAS
    candidates: List[CompiledPlan] = []

    # 1. State-expanded compiler.
    compiler_plan = compile_plan(graph, profile, net, level_deltas=level_deltas)
    compiler_plan = CompiledPlan(
        network=compiler_plan.network,
        node_assignment=compiler_plan.node_assignment,
        total_cost_ms=compiler_plan.total_cost_ms,
        node_cost_ms=compiler_plan.node_cost_ms,
        conversion_cost_ms=compiler_plan.conversion_cost_ms,
        bootstrap_cost_ms=compiler_plan.bootstrap_cost_ms,
        steps=compiler_plan.steps,
        region_ids=compiler_plan.region_ids,
        strategy_used="compiler",
    )
    candidates.append(compiler_plan)

    # 2. Fixed-placement baselines.
    static_policies = {
        "all_HE":         {op: "HE" for op in DEFAULT_LEVEL_DELTAS},
        "all_MPC":        {op: "MPC" for op in DEFAULT_LEVEL_DELTAS},
        "linHE_nlMPC":    POLICY_LINEAR_HE_NONLINEAR_MPC,
        "linMPC_nlHE":    POLICY_LINEAR_MPC_NONLINEAR_HE,
        "attnMPC_ffnHE":  POLICY_ATTN_MPC_FFN_HE,
    }
    for label, policy in static_policies.items():
        p = _materialise_static_plan(
            graph, profile, net, policy, label, level_deltas
        )
        if not math.isinf(p.total_cost_ms):
            candidates.append(p)

    # 3. Pick the minimum.
    winner = min(candidates, key=lambda c: c.total_cost_ms)
    return winner

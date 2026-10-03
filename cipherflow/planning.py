"""cipherflow.compile: graph + profile + deployment -> Plan."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

from compiler.state_expanded.cost_model import StateExpandedCostModel
from compiler.state_expanded.planner import compile_plan, compile_plan_safe
from compiler.state_expanded.profile_schema import LatencyProfile, OperatorRecord

from . import backends as _backends
from . import profiles as _profiles
from .deploy import Deployment
from .errors import CostExtrapolationError, CostExtrapolationWarning, UnsupportedOperator
from .plan import Plan
from .validate import check_graph

DOMAINS = ("HE", "MPC")


def _check_methods(graph, methods) -> None:
    ops = _backends.supported_ops(methods)
    missing = sorted({n.op_type for n in graph.nodes
                      if n.op_type not in ops and n.op_type not in _backends.PSEUDO_OPS})
    if missing:
        raise UnsupportedOperator(f"no backend method for op types {missing}")


def _check_coverage(graph, methods, profile) -> None:
    ops = _backends.supported_ops(methods)
    profiled = {(r.op_type, r.domain) for r in profile.operators}
    unprofiled = sorted({n.op_type for n in graph.nodes if n.op_type in ops
                         and not any((n.op_type, d) in profiled for d in ops[n.op_type])})
    if unprofiled:
        raise UnsupportedOperator(f"profile has no cost record for op types {unprofiled}")


def _record(op_type, domain, node, *, feasible, method, note) -> OperatorRecord:
    return OperatorRecord(op_type=op_type, domain=domain, method=method,
                          input_shape=tuple(node.input_shape), output_shape=tuple(node.output_shape),
                          local_compute_ms=0.0, comm_bytes=0, comm_rounds=0, he_level_delta=0,
                          feasible=feasible, metadata={"source": "cipherflow", "note": note})


def _prepare_profile(graph, methods, profile: LatencyProfile) -> Tuple[LatencyProfile, List[str]]:
    """Profile the planner sees: the measured one plus
    * zero-cost records for structural nodes (Input),
    * infeasible records where a node violates its method's contract, or where a
      domain has no method or no cost record at all for the op type.
    Raises UnsupportedOperator when a node is feasible in no domain."""
    ops = _backends.supported_ops(methods)
    viol = _backends.violations(graph, methods)
    has_records = {(r.op_type, r.domain) for r in profile.operators}
    blocked: Dict[Tuple[str, str], str] = {}
    notes: List[str] = []
    for n in graph.nodes:
        if n.op_type in _backends.PSEUDO_OPS:
            continue
        for d in DOMAINS:
            if d not in ops.get(n.op_type, ()):
                blocked[(n.node_id, d)] = f"no {d} method"
            elif (n.op_type, d) not in has_records:
                blocked[(n.node_id, d)] = f"profile has no {d} cost record for {n.op_type}"
            elif (n.node_id, d) in viol:
                blocked[(n.node_id, d)] = viol[(n.node_id, d)]

    cm = StateExpandedCostModel(profile)
    impossible = []
    for n in graph.nodes:
        if n.op_type in _backends.PSEUDO_OPS:
            continue
        ok = [d for d in DOMAINS if (n.node_id, d) not in blocked and _feasible(cm, n, d)]
        if not ok:
            why = "; ".join(f"{d}: {blocked.get((n.node_id, d), 'infeasible in the profile')}" for d in DOMAINS)
            impossible.append(f"{n.node_id} ({n.op_type} {list(n.input_shape)}): {why}")
    if impossible:
        raise UnsupportedOperator("no domain can run " + " | ".join(impossible[:5])
                                  + (f" (+{len(impossible) - 5} more)" if len(impossible) > 5 else ""))

    keys_blocked = {}
    for (nid, d), why in blocked.items():
        n = _node(graph, nid)
        keys_blocked[(n.op_type, d, tuple(n.input_shape), tuple(n.output_shape))] = (n, why)
    extra: List[OperatorRecord] = []
    for (op, d, _i, _o), (n, why) in keys_blocked.items():
        extra.append(_record(op, d, n, feasible=False, method="contract", note=why))
    seen_pseudo = set()
    for n in graph.nodes:
        if n.op_type in _backends.PSEUDO_OPS:
            for d in DOMAINS:
                k = (n.op_type, d, tuple(n.input_shape), tuple(n.output_shape))
                if k not in seen_pseudo:
                    seen_pseudo.add(k)
                    extra.append(_record(n.op_type, d, n, feasible=True, method="client",
                                         note="model input provided by the client"))
    kept = [r for r in profile.operators
            if (r.op_type, r.domain, tuple(r.input_shape), tuple(r.output_shape)) not in keys_blocked]
    by_reason: Dict[str, int] = {}
    for (nid, d), why in blocked.items():
        if not why.startswith("no "):
            msg = f"{_node(graph, nid).op_type} kept out of {d}: {why}"
            by_reason[msg] = by_reason.get(msg, 0) + 1
    notes += [f"{msg} ({k} node{'s' if k > 1 else ''})" for msg, k in by_reason.items()]
    return replace(profile, operators=kept + extra), notes


def _feasible(cm, node, domain) -> bool:
    try:
        rec, _ = cm._find_operator(node.op_type, domain, node.input_shape, node.output_shape)
    except KeyError:
        return False
    return rec.feasible


def _node(graph, nid):
    for n in graph.nodes:
        if n.node_id == nid:
            return n
    raise KeyError(nid)


@dataclass
class CostReport:
    """Which cost entries the planner consults are exact profile records and which
    are scaled from the nearest profiled shape (by element count)."""
    scaled_ops: Dict[Tuple[str, str, tuple], tuple] = field(default_factory=dict)   # (op, dom, in) -> profiled in
    scaled_conversions: Dict[Tuple[str, str, tuple], tuple] = field(default_factory=dict)
    total: int = 0
    node_source: Dict[Tuple[str, str], str] = field(default_factory=dict)            # (node, dom) -> exact|scaled

    @property
    def n_scaled(self) -> int:
        return len(self.scaled_ops) + len(self.scaled_conversions)

    def summary(self, profile_name: str) -> str:
        ex = [f"{op}/{d} {list(s)} <- {list(p)}" for (op, d, s), p in list(self.scaled_ops.items())[:3]]
        ex += [f"{a}->{b} conversion {list(s)} <- {list(p)}"
               for (a, b, s), p in list(self.scaled_conversions.items())[:max(0, 3 - len(ex))]]
        return (f"{self.n_scaled} of {self.total} cost entries are not in profile {profile_name!r} and were "
                f"scaled from the nearest profiled shape (e.g. {'; '.join(ex)}). Plan latencies for these "
                f"shapes are estimates: measure them with profile='auto' on the target machine, or pass "
                f"strict=True to refuse")


def cost_report(graph, profile: LatencyProfile) -> CostReport:
    cm = StateExpandedCostModel(profile)
    rep = CostReport()
    seen = set()
    for n in graph.nodes:
        for d in DOMAINS:
            try:
                rec, how = cm._find_operator(n.op_type, d, n.input_shape, n.output_shape)
            except KeyError:
                continue
            if not rec.feasible:
                continue
            rep.node_source[(n.node_id, d)] = "exact" if how == "exact" else "scaled"
            key = (n.op_type, d, tuple(n.input_shape), tuple(n.output_shape))
            if key in seen:
                continue
            seen.add(key)
            rep.total += 1
            if how != "exact":
                rep.scaled_ops[(n.op_type, d, tuple(n.input_shape))] = tuple(rec.input_shape)
    for shape in sorted({tuple(e.tensor_shape) for e in graph.edges}):
        for a, b in (("HE", "MPC"), ("MPC", "HE")):
            try:
                rec, how = cm._find_conversion(a, b, shape)
            except KeyError:
                continue
            rep.total += 1
            if how != "exact":
                rep.scaled_conversions[(a, b, shape)] = tuple(rec.tensor_shape)
    return rep


def compile(graph, deployment: Deployment, *,
            profile: Union[str, LatencyProfile, None] = None,
            methods: Union[str, Sequence[_backends.Method]] = "auto",
            safe: bool = True, strict: bool = False, verbose: bool = True) -> Plan:
    """Plan hybrid HE/MPC execution of `graph` for `deployment`.

    profile: a LatencyProfile, a JSON path, a built-in name ('cpu', 'gpu'), or 'auto'.
             'auto' requires Deployment.local(...): it loads this machine's cached
             MicroBench profile and measures any entries the graph still needs.
             Defaults to the deployment's hardware profile.
    methods: 'auto' for the built-in NEXUS-CKKS / SCI-SHAFT methods, or a list of Method.
    safe:    also materialise the static baselines and return the cheapest plan
             (guarantees the result is never worse than a static placement).
    strict:  refuse (CostExtrapolationError) when a cost entry is not in the profile,
             instead of scaling it from the nearest profiled shape with a warning.

    Raises UnsupportedGraph for structures the planner does not handle and
    UnsupportedOperator for operators no backend can run; see cipherflow.errors.
    """
    meths = _backends.builtin_methods() if methods == "auto" else list(methods)
    check_graph(graph)
    _check_methods(graph, meths)
    spec = profile if profile is not None else (deployment.hardware.profile or "auto")
    if spec == "auto":
        if not deployment.hardware.local:
            raise ValueError(
                f"profile='auto' measures this machine, but the deployment targets "
                f"{deployment.hardware.name or deployment.hardware.device!r}. Use Deployment.local(...) "
                "on the target machine, or pass that machine's profile file.")
        from .measure import microbench
        prof = microbench(graph, meths, device=deployment.hardware.device, verbose=verbose)
        prof_name = "auto"
    else:
        prof = _profiles.load(spec)
        prof_name = spec if isinstance(spec, str) else (prof.platform or "custom")
    _check_coverage(graph, meths, prof)
    prof, notes = _prepare_profile(graph, meths, prof)

    report = cost_report(graph, prof)
    plan_warnings = list(notes)
    if report.n_scaled:
        msg = report.summary(prof_name)
        if strict:
            raise CostExtrapolationError(msg)
        warnings.warn(msg, CostExtrapolationWarning, stacklevel=2)
        plan_warnings.insert(0, msg)

    deltas = _backends.level_deltas(meths)
    net = deployment.network.to_setting()
    compiled = (compile_plan_safe(graph, prof, net, level_deltas=deltas) if safe
                else compile_plan(graph, prof, net, level_deltas=deltas))
    plan = Plan.from_compiled(compiled, graph, prof, deployment)
    for n in plan.nodes:
        n.cost_source = ("input" if n.op_type in _backends.PSEUDO_OPS
                         else report.node_source.get((n.id, n.domain), "exact"))
    plan.warnings = plan_warnings
    return plan

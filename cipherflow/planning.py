"""cipherflow.compile: graph + profile + deployment -> Plan; cipherflow.match: which backends fit."""
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
from .errors import (CostExtrapolationError, CostExtrapolationWarning, UnsupportedBackend,
                     UnsupportedOperator)
from .calibration import Calibration
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
        raise UnsupportedOperator(f"profile has no cost record for op types {unprofiled}" + (
            "; the paper's profiles (paper-cpu, paper-gpu) predate the Q/K/V projections: plan the "
            "paper's graph (cf.models.bert(qkv_projection=False), cf.models.vit_paper()) to reproduce "
            "the paper, or use a measured profile" if "Linear_QKV" in unprofiled else ""))


def _record(op_type, domain, node, *, feasible, method, note) -> OperatorRecord:
    return OperatorRecord(op_type=op_type, domain=domain, method=method,
                          input_shape=tuple(node.input_shape), output_shape=tuple(node.output_shape),
                          local_compute_ms=0.0, comm_bytes=0, comm_rounds=0, he_level_delta=0,
                          feasible=feasible, metadata={"source": "cipherflow", "note": note})


def _prepare_profile(graph, methods, profile: LatencyProfile, calibration=None) -> Tuple[LatencyProfile, List[str]]:
    """Profile the planner sees: the measured one plus
    * zero-cost records for structural nodes (Input),
    * infeasible records where a node violates its method's contract, or where a
      domain has no method or no cost record at all for the op type.
    Raises UnsupportedOperator when a node is feasible in no domain."""
    ops = _backends.supported_ops(methods)
    viol = _backends.violations(graph, methods, calibration)
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
                blocked[(n.node_id, d)] = f"profile has no {d} cost record for {n.op_type}" + (
                    " (the paper's profiles predate it: plan the paper's graph, e.g. "
                    "cf.models.bert(qkv_projection=False), to reproduce the paper)"
                    if n.op_type == "Linear_QKV" else "")
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


def _with_variant_records(profile: LatencyProfile, graph, kernels) -> LatencyProfile:
    """MPC cost records for the op types of kernel variants ("GeLU.wide"): the base op's."""
    if not kernels:
        return profile
    variant_ops = {n.op_type for n in graph.nodes if n.node_id in kernels}
    have = {(r.op_type, r.domain) for r in profile.operators}
    extra = [replace(r, op_type=v) for v in sorted(variant_ops) if (v, "MPC") not in have
             for r in profile.operators if r.domain == "MPC" and r.op_type == v.split(".", 1)[0]]
    return replace(profile, operators=list(profile.operators) + extra)


def _with_scaled_bootstrap(profile: LatencyProfile, calibration) -> LatencyProfile:
    """With a calibration and a bootstrap record that states its flat range: the nodes whose
    outputs exceed it are bootstrapped scaled (one more level; the cost model reads this)."""
    flat = (profile.bootstrap.metadata or {}).get("flat_range") if profile.bootstrap is not None else None
    if calibration is None or not flat:
        return profile
    from .kernel_params import bootstrap_amplitude, bootstrap_scale
    amp = {nid: bootstrap_amplitude(s["out_absmax"], s.get("out_absmean", 0.0))
           for nid, s in calibration.nodes.items() if s.get("out_absmax") is not None}
    scaled = {nid: bootstrap_scale(a, flat) for nid, a in amp.items() if bootstrap_scale(a, flat) > 1}
    return replace(profile, metadata=dict(profile.metadata or {}, scaled_bootstrap=scaled))


def _kernel_notes(graph, kernels) -> List[str]:
    """One note per (op type, kernel): nodes whose calibrated inputs need a wide-range kernel."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    ops = {n.node_id: n.op_type for n in graph.nodes}
    for nid, k in kernels.items():
        params = ", ".join(f"{a} {b}" for a, b in k.items() if a != "kernel")
        groups.setdefault((ops[nid], k["kernel"]), []).append(params)
    return [f"{op} runs its {kind}-range HE kernel on {len(ps)} node{'s' if len(ps) > 1 else ''} "
            f"(calibrated inputs outside the base kernel's range; {', '.join(sorted(set(ps)))})"
            for (op, kind), ps in groups.items()]


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


# ----------------------------------------------------------------- backend candidates

@dataclass
class Candidate:
    """One (HE backend, MPC backend) pair to plan with, and the profile to use."""
    he: str
    mpc: str
    profile: Union[str, LatencyProfile]          # built-in name, path, LatencyProfile, or "auto"
    methods: Optional[List[_backends.Method]] = None   # None: built-in methods (levels from the profile if needed)

    @property
    def name(self) -> str:
        return f"{self.he}+{self.mpc}"

    @property
    def backend(self) -> Optional[_backends.MPCBackend]:
        return _backends.MPC_BACKENDS.get(self.mpc)

    @property
    def he_backend(self) -> Optional[_backends.HEBackend]:
        return _backends.HE_BACKENDS.get(self.he)


def _security_reason(name: str, security: str) -> Optional[str]:
    b = _backends.MPC_BACKENDS.get(name)
    allowed = _backends.SECURITY[security]
    if b is None:
        return (None if security == "any" else
                f"unknown MPC backend, so its security model is unknown; pass security='any' or set "
                f"the profile's metadata['mpc_backend'] to one of {sorted(_backends.MPC_BACKENDS)}")
    if b.security not in allowed:
        return (f"security model {b.security} ({_backends.SECURITY_TEXT[b.security]}) is not allowed by "
                f"security={security!r}; pass security={'2pc-dealer' if b.security == '2pc-dealer' else 'any'!r} "
                f"to allow it")
    return None


def _methods(cand: Candidate, profile: Optional[LatencyProfile]) -> List[_backends.Method]:
    """Methods for a candidate. HE level consumption comes from the HE backend's table, or from
    the profile's measured records (he_level_delta, metadata level_out) when it has none."""
    if cand.methods is not None:
        return cand.methods
    hb = cand.he_backend
    mpc = cand.mpc if cand.mpc in _backends.MPC_BACKENDS else "sci"
    if hb is not None and hb.level_deltas is not None:
        meths = _backends.builtin_methods(mpc, he=cand.he)
    else:
        deltas, outs = _backends.profile_level_rules(profile) if profile is not None else ({}, {})
        meths = _backends.builtin_methods(mpc, he=cand.he if hb is not None else "openfhe-fast",
                                          level_deltas=deltas, level_out=outs)
        if hb is None:                                  # custom HE backend: its own name, no contracts
            meths = [replace(m, backend=cand.he, contract=None) if m.domain == "HE" else m for m in meths]
    if cand.mpc not in _backends.MPC_BACKENDS:          # custom MPC backend: its own name
        meths = [replace(m, backend=cand.mpc) if m.domain == "MPC" else m for m in meths]
    return meths


def resolve_candidates(deployment: Deployment, profile=None, he: str = "auto", mpc: str = "auto",
                       security: str = "2pc", min_precision: Optional[float] = None,
                       methods="auto") -> Tuple[List[Candidate], Dict[str, str]]:
    """(HE, MPC) backend pairs to plan with, and the reason each other pair is excluded."""
    _backends.check_security(security)
    for arg, reg, what in ((he, _backends.HE_BACKENDS, "he"), (mpc, _backends.MPC_BACKENDS, "mpc")):
        if arg != "auto" and arg not in reg:
            raise ValueError(f"{what} must be 'auto' or one of {sorted(reg)}, got {arg!r}")
    hw = deployment.hardware
    spec = profile if profile is not None else (hw.profile or "auto")
    pairs = [(h, m) for h in _backends.HE_BACKENDS if he in ("auto", h)
             for m in _backends.MPC_BACKENDS if mpc in ("auto", m)]
    cands: List[Candidate] = []
    excluded: Dict[str, str] = {}

    def why_not(h, m):
        return _security_reason(m, security) or _backends.precision_reason(h, min_precision)

    if methods != "auto":                       # caller-supplied methods: their backends, as given
        meths = list(methods)
        hs = sorted({x.backend for x in meths if x.domain == "HE"})
        ms = sorted({x.backend for x in meths if x.domain == "MPC"})
        h = hs[0] if len(hs) == 1 else ("none" if not hs else "custom")
        m = ms[0] if len(ms) == 1 else ("none" if not ms else "custom")
        why = (None if m == "none" else _security_reason(m, security)) or \
              (None if h == "none" else _backends.precision_reason(h, min_precision))
        if why:
            raise UnsupportedBackend(f"{h}+{m}: {why}")
        return [Candidate(h, m, spec, meths)], excluded

    if spec == "auto":
        if not hw.local:
            raise ValueError(
                f"profile='auto' measures this machine, but the deployment targets "
                f"{hw.name or hw.device!r}. Use Deployment.local(...) on the target machine, or pass "
                "that machine's profile file.")
        from .measure import BackendUnavailable, measurable
        wired = measurable(hw.device)
        if not wired:
            raise BackendUnavailable(
                f"automatic MicroBench is not wired for device={hw.device!r} yet; pass a measured profile "
                "file (profile='path/to/profile.json'), or plan for CPU execution with "
                "Deployment.local(..., device='cpu')")
        for h, m in pairs:
            why = why_not(h, m)
            if why is None and f"{h}+{m}" not in wired:
                why = f"automatic MicroBench is not wired for {h}+{m} on {hw.device} (available: {wired})"
            if why:
                excluded[f"{h}+{m}"] = why
            else:
                cands.append(Candidate(h, m, "auto"))
        return cands, excluded

    hw_table = _profiles.hardware_profiles()
    if isinstance(spec, str) and spec in hw_table:   # a hardware name: try its built-in profiles
        for h, m in pairs:
            why = why_not(h, m)
            if why is None and (h, m) not in hw_table[spec]:
                hb = _backends.HE_BACKENDS[h]
                why = (f"no built-in {spec!r} profile for {h}+{m}"
                       + (f" ({hb.status}: {hb.note})" if hb.status == "in development" else ""))
            if why:
                excluded[f"{h}+{m}"] = why
            else:
                cands.append(Candidate(h, m, hw_table[spec][(h, m)]))
        return cands, excluded

    prof = _profiles.load(spec)                       # one profile: its backends
    h, m = _backends.infer_he_backend(prof), _backends.infer_backend(prof)
    if h is None or m is None:
        raise UnsupportedBackend(f"the profile has no {'HE' if h is None else 'MPC'} cost records")
    if he != "auto" and h != he:
        raise UnsupportedBackend(f"the profile holds {h} HE costs, but he={he!r} was requested")
    if mpc != "auto" and m != mpc:
        raise UnsupportedBackend(f"the profile holds {m} costs, but mpc={mpc!r} was requested")
    why = why_not(h, m)
    if why:
        excluded[f"{h}+{m}"] = why
        return cands, excluded
    return [Candidate(h, m, spec if isinstance(spec, str) else prof)], excluded


def _profile_for(graph, deployment, cand: Candidate, *, measure: bool, verbose: bool):
    if cand.profile == "auto":
        from .measure import microbench
        if not measure:
            return None, "auto"
        return (microbench(graph, _methods(cand, None), device=deployment.hardware.device,
                           backend=cand.mpc, he=cand.he, verbose=verbose), f"auto ({cand.name})")
    prof = _profiles.load(cand.profile)
    name = cand.profile if isinstance(cand.profile, str) else (prof.metadata or {}).get("label") or prof.platform
    return prof, name


def _plan_with(graph, deployment, cand: Candidate, *, safe, strict, verbose, calibration=None):
    _check_methods(graph, _methods(cand, None) if cand.methods is None else cand.methods)
    prof, prof_name = _profile_for(graph, deployment, cand, measure=True, verbose=verbose)
    meths = _methods(cand, prof)
    _check_coverage(graph, meths, prof)
    base_graph = graph
    graph, kernels = _backends.assign_kernels(graph, meths, calibration)
    prof = _with_variant_records(prof, graph, kernels)
    prof, notes = _prepare_profile(graph, meths, prof, calibration)
    prof = _with_scaled_bootstrap(prof, calibration)
    report = cost_report(graph, prof)
    scaled = report.summary(prof_name) if report.n_scaled else None
    if scaled and strict:
        raise CostExtrapolationError(scaled)
    hb = cand.he_backend
    if hb is not None and hb.status != "measured":
        notes.append(f"HE costs of {cand.he} are {hb.status}: {hb.note}")
    deltas = _backends.level_deltas(meths)
    net = deployment.network.to_setting()
    compiled = (compile_plan_safe(graph, prof, net, level_deltas=deltas) if safe
                else compile_plan(graph, prof, net, level_deltas=deltas))
    plan = Plan.from_compiled(compiled, graph, prof, deployment)
    base_ops = {n.node_id: n.op_type for n in base_graph.nodes}
    sources = {n.node_id: (n.attributes or {}).get("source") for n in base_graph.nodes}
    for n in plan.nodes:                         # where each operator is in the PyTorch model
        n.source = sources.get(n.id)
    for n in plan.nodes:
        if n.id in kernels:                      # back to the model's op type; the kernel says which
            n.op_type = base_ops[n.id]
            n.kernel = dict(kernels[n.id]) if n.domain == "HE" else None
    notes += _kernel_notes(base_graph, {n.id: n.kernel for n in plan.nodes if n.kernel})
    for n in plan.nodes:
        if n.op_type in _backends.PSEUDO_OPS:
            n.cost_source, n.backend = "input", "client"
        else:
            n.cost_source = report.node_source.get((n.id, n.domain), "exact")
            n.backend = cand.he if n.domain == "HE" else cand.mpc
    if calibration is None:
        unchecked = sorted({n.op_type for n in plan.nodes if n.domain == "HE" and any(
            m.value_contract is not None for m in meths if m.op_type == n.op_type and m.domain == "HE")})
        if unchecked:
            notes.append(f"HE {', '.join(unchecked)} approximate on fixed input ranges ({cand.he}: "
                         f"{_ranges_text(cand.he)}); real models often exceed them. Pass "
                         "calibration=cipherflow.runtime.calibrate(model, inputs) to keep operators whose "
                         "inputs fall outside out of HE")
    if plan.counts()["conversions"] and not all((c.metadata or {}).get("protocol") == "hardened"
                                                 for c in prof.conversions):
        notes.append(_UNHARDENED + " (slot-domain mask; the hardened protocol is measured for the OpenFHE "
                     "backend), so their cost is underestimated")
    if calibration is not None:
        notes += _conversion_bound_notes(plan, prof, calibration)
    plan.warnings = ([scaled] if scaled else []) + notes
    plan.mpc_backend, plan.he_backend = cand.mpc, cand.he
    plan.security = cand.backend.security if cand.backend else ("none" if cand.mpc == "none" else "unknown")
    plan.precision_bits = hb.precision_bits if hb is not None else None
    plan.profile["name"] = prof_name
    return plan, scaled


_UNHARDENED = "HE<->MPC conversions are priced with an unhardened protocol"


def _unhardened(plan: Plan) -> bool:
    return any(w.startswith(_UNHARDENED) for w in plan.warnings)


def _conversion_bound_notes(plan, prof, calibration) -> List[str]:
    """The hardened conversions assume |x| < 2^bound_bits (statistical hiding of HE -> MPC, wrap
    removal of MPC -> HE); calibrated values above it would be converted wrongly."""
    from .kernel_params import CONVERSION_BOUND_BITS
    bits = [(c.metadata or {}).get("bound_bits") for c in prof.conversions]
    bound = 2.0 ** min([b for b in bits if b is not None] or [CONVERSION_BOUND_BITS])
    over = {}
    for a in plan.actions:
        if a.kind == "convert" and a.edge:
            peak = (calibration.nodes.get(a.edge[0]) or {}).get("out_absmax", 0.0)
            if peak >= bound:
                over[a.edge[0]] = peak
    if not over:
        return []
    worst = sorted(over, key=over.get, reverse=True)
    return [f"calibrated values up to {over[worst[0]]:.4g} cross HE<->MPC conversions ({', '.join(worst[:4])}"
            f"{', ...' if len(worst) > 4 else ''}); the conversion protocol assumes |x| < {bound:g}"]


def _ranges_text(he: str) -> str:
    from .kernel_params import HE_NUMERICS
    p = HE_NUMERICS.get(he)
    if p is None:
        return "see cipherflow.kernel_params"
    return (f"GeLU [{-p.gelu_bound:g}, {p.gelu_bound:g}], LayerNorm variance [{p.ln_var_lo:g}, {p.ln_var_hi:g}], "
            f"Softmax scores [{-p.sm_bound:g}, {p.sm_bound:g}] with row means of exp(x - {p.sm_bound:g}) >= {p.sm_umin:g}")


_CANDIDATE_ERRORS = (UnsupportedOperator, CostExtrapolationError, LookupError, RuntimeError, ValueError)


def compile(graph, deployment: Deployment, *,
            profile: Union[str, LatencyProfile, None] = None,
            he: str = "auto", mpc: str = "auto", security: str = "2pc",
            min_precision: Optional[float] = None,
            methods: Union[str, Sequence[_backends.Method]] = "auto",
            safe: bool = True, strict: bool = False, verbose: bool = True,
            calibration=None) -> Plan:
    """Plan hybrid HE/MPC execution of `graph` for `deployment`.

    profile:  a LatencyProfile, a JSON path, a built-in name (see cipherflow.profiles.BUILTIN;
              paper-cpu / paper-gpu reproduce the paper), a hardware name ('cpu', 'gpu': its
              built-in measured profiles, one per HE/MPC backend pair), or 'auto'. 'auto' requires
              Deployment.local(...): it loads this machine's cached MicroBench profile and
              measures any entries the graph still needs. Defaults to the deployment's hardware.
    he, mpc:  HE backend (cipherflow.backends.HE_BACKENDS) and MPC backend (MPC_BACKENDS), or
              'auto' to plan with every pair that fits and return the fastest plan
              (plan.alternatives lists the others and why any were not used). Plans that
              convert with an unhardened protocol (NEXUS profiles) rank after all others.
    security: '2pc' (two parties, no third party; default), '2pc-dealer' (also allow a trusted
              dealer for preprocessing, e.g. SHAFT), or 'any'.
    min_precision: minimum HE bootstrapping precision in bits; HE backends below it (or whose
              precision is not established) are not used.
    methods:  'auto' for the built-in methods of each backend pair, or a list of Method.
    safe:     also materialise the static baselines and return the cheapest plan
              (guarantees the result is never worse than a static placement).
    strict:   refuse (CostExtrapolationError) cost entries that are not in the profile, instead
              of scaling them from the nearest profiled shape with a warning; with 'auto'
              backends, pairs that would need scaled entries are skipped.
    calibration: a cipherflow.Calibration (or its JSON path) from cipherflow.runtime.calibrate:
              operators whose calibrated inputs fall outside an HE method's approximation are
              kept out of HE (see cipherflow.kernel_params).

    Raises UnsupportedGraph for structures the planner does not handle, UnsupportedOperator
    for operators no backend can run, and UnsupportedBackend when no backend pair fits;
    see cipherflow.errors.
    """
    check_graph(graph)
    if calibration is not None:
        calibration = Calibration.load(calibration)
    cands, excluded = resolve_candidates(deployment, profile, he, mpc, security, min_precision, methods)
    if not cands:
        raise UnsupportedBackend("no backend pair can be used: " +
                                 "; ".join(f"{n}: {why}" for n, why in excluded.items()))
    plans: Dict[str, Tuple[Plan, Optional[str]]] = {}
    for c in cands:
        try:
            plans[c.name] = _plan_with(graph, deployment, c, safe=safe, strict=strict, verbose=verbose,
                                       calibration=calibration)
        except _CANDIDATE_ERRORS as exc:
            if len(cands) == 1:
                raise
            excluded[c.name] = f"{type(exc).__name__}: {exc}"
    if not plans:
        raise UnsupportedBackend("no backend pair can plan this model: " +
                                 "; ".join(f"{n}: {why}" for n, why in excluded.items()))
    # plans whose conversions are priced with an unhardened protocol are cheaper than they would be
    # with a secure one: they rank after every other plan; equal costs (e.g. all-MPC plans of
    # several HE backends) go to a measured HE backend, then to the plan with fewer warnings
    def _rank(n):
        p = plans[n][0]
        hb = _backends.HE_BACKENDS.get(p.he_backend or "")
        return (_unhardened(p), p.cost["total_ms"], hb is None or hb.status != "measured", len(p.warnings))
    best_name = min(plans, key=_rank)
    plan, scaled = plans[best_name]
    plan.alternatives = {n: {"total_ms": p.cost["total_ms"], "security": p.security,
                             "precision_bits": p.precision_bits, "warnings": len(p.warnings),
                             **({"unhardened_conversions": True} if _unhardened(p) else {})}
                         for n, (p, _s) in plans.items() if n != best_name}
    plan.alternatives.update({n: {"excluded": why} for n, why in excluded.items()})
    if scaled:
        warnings.warn(scaled, CostExtrapolationWarning, stacklevel=2)
    return plan


# ----------------------------------------------------------------- compatibility report

@dataclass
class MatchRow:
    backend: str                 # "<he>+<mpc>"
    status: str                  # "ok" | "excluded"
    detail: str
    security: str = ""
    precision_bits: Optional[float] = None
    profile: str = ""


@dataclass
class MatchReport:
    graph_id: str
    rows: List[MatchRow]

    def compatible(self) -> List[str]:
        return [r.backend for r in self.rows if r.status == "ok"]

    def __str__(self) -> str:
        w = max([len(r.backend) for r in self.rows] + [7])
        lines = [f"backends for {self.graph_id}:"]
        for r in self.rows:
            lines.append(f"  {r.backend:{w}s}  {r.status:8s}  {r.detail}")
        return "\n".join(lines)


def match(graph, deployment: Deployment, *, profile=None, he: str = "auto", mpc: str = "auto",
          security: str = "2pc", min_precision: Optional[float] = None, methods="auto",
          calibration=None) -> MatchReport:
    """Which (HE, MPC) backend pairs can plan `graph` on `deployment`, without planning or
    measuring: operator coverage, method contracts, security, precision, profile coverage."""
    check_graph(graph)
    cands, excluded = resolve_candidates(deployment, profile, he, mpc, security, min_precision, methods)
    rows: List[MatchRow] = []
    for c in cands:
        sec = c.backend.security if c.backend else "unknown"
        prec = c.he_backend.precision_bits if c.he_backend else None
        try:
            if c.profile == "auto":
                from .measure import microbench
                meths = _methods(c, None)
                _check_methods(graph, meths)
                todo = microbench(graph, meths, device=deployment.hardware.device, backend=c.mpc, he=c.he,
                                  dry_run=True, verbose=False)
                n = len(todo.ops) + 2 * len(todo.conversion_shapes) + int(todo.bootstrap)
                rows.append(MatchRow(c.name, "ok", f"profile='auto': {n} cost entries still to measure"
                                     if n else "profile='auto': all cost entries cached", sec, prec, "auto"))
                continue
            prof, prof_name = _profile_for(graph, deployment, c, measure=False, verbose=False)
            meths = _methods(c, prof)
            _check_methods(graph, meths)
            _check_coverage(graph, meths, prof)
            prof, notes = _prepare_profile(graph, meths, prof,
                                           Calibration.load(calibration) if calibration is not None else None)
            rep = cost_report(graph, prof)
            detail = (f"{rep.total - rep.n_scaled}/{rep.total} cost entries exact"
                      + (f", {rep.n_scaled} scaled from other shapes" if rep.n_scaled else ""))
            hb = c.he_backend
            if hb is not None and hb.status != "measured":
                detail += f"; HE costs {hb.status}"
            if notes:
                detail += "; " + "; ".join(notes)
            rows.append(MatchRow(c.name, "ok", detail, sec, prec, str(prof_name)))
        except _CANDIDATE_ERRORS as exc:
            rows.append(MatchRow(c.name, "excluded", f"{type(exc).__name__}: {exc}", sec, prec))
    for n, why in excluded.items():
        h, _, m = n.partition("+")
        b, hb = _backends.MPC_BACKENDS.get(m), _backends.HE_BACKENDS.get(h)
        rows.append(MatchRow(n, "excluded", why, b.security if b else "unknown",
                             hb.precision_bits if hb else None))
    return MatchReport(graph.graph_id, rows)

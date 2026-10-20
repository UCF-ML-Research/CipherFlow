"""Extensions: new primitives for the backends, and new operators for the planner.

An Extension brings primitives, each one implementation of an operator in one domain with
the CKKS levels it consumes, the inputs it accepts and its cost, and, for the PyTorch front
end, the module classes that are its operators:

    ext = cf.Extension("fast-softmax")
    ext.primitive("Softmax", "HE", levels=4, cost=cf.Cost(local_ms=9_000))
    plan = cf.compile(graph, deployment, extensions=[ext])

    rag = cf.Extension("rag")
    rag.operator(TopK, "TopK")                                   # nn.Module class -> operator
    rag.primitive("TopK", "MPC", cost=cf.Cost(local_ms=800, comm_bytes=40_000_000, comm_rounds=200))
    graph = cf.models.from_torch(retriever, x, extensions=[rag])

A primitive for an operator that the backend pair already implements in that domain replaces
the built-in implementation (and its kernel variants). For a new operator it is the operator's
only implementation in that domain, so a node can run only in the domains its primitives cover.
An extension's costs are its own: plans mark the operators it runs (`node.backend`, the
`[name]` tag in `plan.explain(steps=True)`) and say that the extension priced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from compiler.state_expanded.profile_schema import OperatorRecord

from .backends import Method
from .errors import UnsupportedOperator

DOMAINS = ("HE", "MPC")


@dataclass(frozen=True)
class Cost:
    """One call of a primitive: local compute time, and the bytes and rounds it sends."""
    local_ms: float
    comm_bytes: int = 0
    comm_rounds: int = 0


# A fixed Cost, or (node, profile) -> Cost for costs that depend on the node's shapes. The profile
# is the backend pair's measured one, so a primitive can be priced from a measured operator (like()).
CostFn = Union[Cost, Callable[[object, object], Cost]]


@dataclass(frozen=True)
class Primitive:
    op: str                      # operator type it implements, e.g. "Softmax" or a new one, "TopK"
    domain: str                  # "HE" | "MPC"
    cost: CostFn
    levels: int = 0              # CKKS levels it consumes (HE)
    level_out: Optional[int] = None                              # HE: bootstraps internally to this level
    accepts: Optional[Callable[[object], Optional[str]]] = None  # node -> None, or why it cannot run it

    def cost_of(self, node, profile) -> Cost:
        c = self.cost(node, profile) if callable(self.cost) else self.cost
        if not isinstance(c, Cost):
            raise TypeError(f"cost of {self.op} in {self.domain} must be a cipherflow.Cost, got {type(c).__name__}")
        return c


@dataclass
class Extension:
    name: str
    primitives: List[Primitive] = field(default_factory=list)
    modules: Dict[type, str] = field(default_factory=dict)       # PyTorch module class -> operator type
    note: str = ""
    # an HE backend of its own (cipherflow.lowering.PrimitiveBackend): its level budget and bootstrapping
    he_levels: Optional[int] = None
    bootstrap: Optional[Cost] = None
    bootstrap_slots: Optional[int] = None

    def primitive(self, op: str, domain: str, *, cost: CostFn, levels: int = 0, level_out: Optional[int] = None,
                  accepts: Optional[Callable[[object], Optional[str]]] = None) -> "Extension":
        """Add an implementation of operator `op` in `domain` ("HE" or "MPC")."""
        if domain not in DOMAINS:
            raise ValueError(f"domain must be 'HE' or 'MPC', got {domain!r}")
        if domain == "MPC" and (levels or level_out is not None):
            raise ValueError("an MPC primitive consumes no CKKS levels")
        if levels < 0:
            raise ValueError("levels must be >= 0")
        if any(p.op == op and p.domain == domain for p in self.primitives):
            raise ValueError(f"{self.name} already has a {domain} primitive for {op}")
        self.primitives.append(Primitive(op, domain, cost, int(levels), level_out, accepts))
        return self

    def operator(self, module_class: type, op: str) -> "Extension":
        """Trace instances of the PyTorch module class as one operator of type `op` (from_torch)."""
        self.modules[module_class] = op
        return self


def like(op: str, domain: str, scale: Callable[[object, OperatorRecord], float]) -> Callable:
    """A cost function that prices a node from the profile's measured record of `op` in `domain`
    (the largest one), with time and bytes multiplied by scale(node, record); rounds unchanged."""
    def cost(node, profile) -> Cost:
        recs = [r for r in profile.operators if r.op_type == op and r.domain == domain and r.feasible
                and (r.metadata or {}).get("source") != "extension"]
        if not recs:
            raise UnsupportedOperator(f"the profile has no {domain} record of {op} to price {node.op_type} from")
        r = max(recs, key=lambda r: numel(r.input_shape))
        f = float(scale(node, r))
        return Cost(r.local_compute_ms * f, int(r.comm_bytes * f), int(r.comm_rounds))
    return cost


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


# ----------------------------------------------------------------- used by the planner and the front end

def _check(extensions) -> List[Extension]:
    exts = list(extensions or ())
    seen: Dict[Tuple[str, str], str] = {}
    for e in exts:
        if not isinstance(e, Extension):
            raise TypeError(f"extensions must be cipherflow.Extension objects, got {type(e).__name__}")
        for p in e.primitives:
            k = (p.op, p.domain)
            if k in seen:
                raise ValueError(f"extensions {seen[k]} and {e.name} both implement {p.op} in {p.domain}")
            seen[k] = e.name
    return exts


def owners(extensions) -> Dict[Tuple[str, str], str]:
    """(operator, domain) -> name of the extension that implements it."""
    return {(p.op, p.domain): e.name for e in _check(extensions) for p in e.primitives}


def with_methods(methods: Sequence[Method], extensions) -> List[Method]:
    """The backend pair's methods with the extensions' primitives in place of (or next to) them."""
    out = list(methods)
    for e in _check(extensions):
        for p in e.primitives:
            out = [m for m in out if not (m.domain == p.domain and (m.op_type == p.op or m.base == p.op))]
            out.append(Method(p.op, p.domain, e.name, p.levels if p.domain == "HE" else 0, p.accepts, p.level_out))
    return out


def with_records(profile, graph, extensions):
    """The profile with one record per graph node an extension implements, priced by the extension
    (the backend's own records for those operators and domains are dropped)."""
    exts = _check(extensions)
    if not exts:
        return profile
    own = [e for e in exts if e.he_levels is not None or e.bootstrap is not None]
    if len(own) > 1:
        raise ValueError(f"extensions {own[0].name} and {own[1].name} both bring their own HE level budget")
    if own:                                       # the extension's HE backend: its levels and bootstrapping
        e = own[0]
        boot = profile.bootstrap
        if e.bootstrap is not None:
            from compiler.state_expanded.profile_schema import BootstrapRecord
            boot = BootstrapRecord(method=f"extension:{e.name}", local_compute_ms=float(e.bootstrap.local_ms),
                                   comm_bytes=int(e.bootstrap.comm_bytes), comm_rounds=int(e.bootstrap.comm_rounds),
                                   metadata={"source": "extension", "extension": e.name,
                                             **({"slots": int(e.bootstrap_slots)} if e.bootstrap_slots else {}),
                                             **({"levels_after": int(e.he_levels)} if e.he_levels else {})})
        profile = replace(profile, bootstrap=boot,
                          he_level_budget=int(e.he_levels) if e.he_levels is not None else profile.he_level_budget)
    prims = {(p.op, p.domain): (e, p) for e in exts for p in e.primitives}
    kept = [r for r in profile.operators if (r.op_type, r.domain) not in prims]
    seen = set()
    for n in graph.nodes:
        for d in DOMAINS:
            hit = prims.get((n.op_type, d))
            key = (n.op_type, d, tuple(n.input_shape), tuple(n.output_shape))
            if hit is None or key in seen:
                continue
            seen.add(key)
            e, p = hit
            c = p.cost_of(n, profile)
            meta = {"source": "extension", "extension": e.name}
            if p.level_out is not None:
                meta["level_out"] = int(p.level_out)
            kept.append(OperatorRecord(op_type=n.op_type, domain=d, method=f"extension:{e.name}",
                                       input_shape=tuple(n.input_shape), output_shape=tuple(n.output_shape),
                                       local_compute_ms=float(c.local_ms), comm_bytes=int(c.comm_bytes),
                                       comm_rounds=int(c.comm_rounds),
                                       he_level_delta=p.levels if d == "HE" else 0, feasible=True, metadata=meta))
    return replace(profile, operators=kept)


def notes(plan, extensions) -> List[str]:
    """One note per extension whose primitives the plan uses."""
    used: Dict[str, set] = {}
    by = owners(extensions)
    for n in plan.nodes:
        name = by.get((n.op_type, n.domain))
        if name:
            used.setdefault(name, set()).add(f"{n.op_type} in {n.domain}")
    return [f"extension {name} prices {', '.join(sorted(ops))} with its own costs, not CipherFlow's measurements"
            for name, ops in used.items()]


def module_ops(extensions) -> Dict[type, Tuple[str, str]]:
    """PyTorch module class -> (operator type, extension name), for the front end."""
    return {cls: (op, e.name) for e in _check(extensions) for cls, op in e.modules.items()}

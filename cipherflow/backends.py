"""Backends and their method declarations (planning side only in v1).

An MPCBackend describes one MPC framework: its security model, the devices it
runs on, and how its cost records are recognised in a profile. A plan uses one
MPC backend next to the HE backend (NEXUS-CKKS); cipherflow.compile picks it
with `mpc=` ("auto" tries every backend that satisfies `security=`).

A Method declares what the planner needs to know about one operator
implementation: its op_type, cryptographic domain, HE level consumption
(delta), backend name, and shape contract. A node that violates a method's
contract cannot be placed in that method's domain; the planner treats it as
infeasible there rather than pricing it from another shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from compiler.state_expanded.planner import DEFAULT_LEVEL_DELTAS

from .kernel_params import HE_NUMERICS, WIDE_KERNELS, unknown_ranges, value_contracts

# Returns None when the node satisfies the contract, else the reason it does not.
Contract = Callable[[object], Optional[str]]

# ----------------------------------------------------------------- MPC backends

@dataclass(frozen=True)
class MPCBackend:
    name: str
    label: str
    security: str                       # "2pc" | "2pc-dealer" | "3pc"
    devices: Tuple[str, ...]            # "cpu", "cuda"
    method_prefixes: Tuple[str, ...]    # how this backend's records are named in profiles
    note: str = ""


MPC_BACKENDS: Dict[str, MPCBackend] = {
    "sci": MPCBackend(
        "sci", "SCI/BOLT (EzPC)", "2pc", ("cpu",),
        ("sci", "bolt", "method_mpc", "method_runtime_default"),
        "two-party, no third party (HE- and OT-based protocols)"),
    "shaft": MPCBackend(
        "shaft", "SHAFT (CrypTen)", "2pc-dealer", ("cpu", "cuda"),
        ("shaft",),
        "two-party online phase; Beaver triples come from a trusted dealer (CrypTen provider TFP), "
        "and the offline phase is not in the costs"),
}

HE_BACKEND = "nexus"            # default HE backend (v0.1 behaviour)

# security= value -> security models it accepts
SECURITY: Dict[str, Tuple[str, ...]] = {
    "2pc": ("2pc",),                          # two parties, no third party
    "2pc-dealer": ("2pc", "2pc-dealer"),      # also allow a trusted dealer for preprocessing
    "any": ("2pc", "2pc-dealer", "3pc"),
}
SECURITY_TEXT = {"2pc": "two parties, no third party", "2pc-dealer": "two parties + trusted dealer",
                 "3pc": "three parties, honest majority"}


def check_security(security: str) -> Tuple[str, ...]:
    if security not in SECURITY:
        raise ValueError(f"security must be one of {sorted(SECURITY)}, got {security!r}")
    return SECURITY[security]


def infer_backend(profile) -> Optional[str]:
    """Name of the MPC backend whose costs a profile holds (None if it has no MPC records).
    profile.metadata["mpc_backend"] takes precedence; otherwise record method names are matched
    against MPC_BACKENDS; "custom" when they match no known backend or several."""
    meta = (profile.metadata or {}).get("mpc_backend")
    if meta:
        return str(meta)
    names = {r.method for r in profile.operators if r.domain == "MPC"}
    if not names:
        return None
    found = set()
    for n in names:
        hits = [b.name for b in MPC_BACKENDS.values()
                if any(n.lower().startswith(p) for p in b.method_prefixes)]
        found.add(hits[0] if len(hits) == 1 else "custom")
    return found.pop() if len(found) == 1 else "custom"


# Structural operators with no backend method: "Input" is the model input when it
# feeds more than one operator (e.g. the first residual). Its domain in the plan
# says how the client provides the input (HE: encrypted, MPC: secret-shared).
PSEUDO_OPS = ("Input",)


@dataclass(frozen=True)
class Method:
    op_type: str
    domain: str            # "HE" | "MPC"
    backend: str
    level_delta: int = 0   # levels the HE method needs on input; 0 for MPC
    contract: Optional[Contract] = field(default=None, compare=False, repr=False)
    level_out: Optional[int] = None   # HE method bootstraps internally: output level (else input - delta)
    # (node, calibrated statistics) -> why the method's approximation is not valid on them (None: valid)
    value_contract: Optional[Callable] = field(default=None, compare=False, repr=False)
    # kernel variant (e.g. a wide-range GeLU): op_type is "<base>.<variant>", base the graph's op type
    base: str = ""
    # (node, calibrated statistics) -> the variant's kernel parameters (e.g. {"bound": 64})
    params: Optional[Callable] = field(default=None, compare=False, repr=False)

    @property
    def variant(self) -> str:
        return self.op_type[len(self.base) + 1:] if self.base else ""

    def violation(self, node, calibration=None) -> Optional[str]:
        """Why this method cannot run `node`: its shape contract, then, with a calibration, its
        value contract on the node's calibrated statistics. A variant chooses its kernel's
        range from the calibration, so it needs one."""
        if self.base and self.domain == "HE" and calibration is None:
            return "the kernel's range is chosen from a calibration"
        why = self.contract(node) if self.contract is not None else None
        if why is None and calibration is not None and self.value_contract is not None:
            stats = calibration.nodes.get(node.node_id)
            why = (self.value_contract(node, stats) if stats is not None
                   else "not in the calibration (calibrate the model this graph was traced from)")
        return why


# ----------------------------------------------------------------- NEXUS contracts
# The NEXUS kernels and the HE cost formulas derived from them fix the model
# width: hidden size 768, 12 heads of dimension 64, Q/K/V packed as [3,B,S,768].
# Per-call token limits of the kernels (e.g. 16 rows per LayerNorm call) are not
# contracts here: the cost model prices larger inputs as repeated calls.

HIDDEN, HEADS = 768, 12


def _tokens_768(node) -> Optional[str]:
    s = tuple(node.input_shape)
    if len(s) != 3 or s[-1] != HIDDEN:
        return f"NEXUS needs input [B,S,{HIDDEN}], got {list(s)}"
    return None


def _qk(node) -> Optional[str]:
    s, o = tuple(node.input_shape), tuple(node.output_shape)
    if len(s) != 4 or s[0] != 3 or s[-1] != HIDDEN:
        return f"NEXUS needs packed Q/K/V [3,B,S,{HIDDEN}], got {list(s)}"
    if len(o) != 4 or o[1] != HEADS:
        return f"NEXUS needs {HEADS} heads, got scores {list(o)}"
    return None


def _av(node) -> Optional[str]:
    s, o = tuple(node.input_shape), tuple(node.output_shape)
    if len(s) != 4 or s[1] != HEADS:
        return f"NEXUS needs attention probabilities [B,{HEADS},S,S], got {list(s)}"
    if not o or o[-1] != HIDDEN:
        return f"NEXUS needs output [B,S,{HIDDEN}], got {list(o)}"
    return None


NEXUS_CONTRACTS: Dict[str, Contract] = {
    "Linear_QKV": _tokens_768,
    "LayerNorm": _tokens_768,
    "FFN_Linear_1": _tokens_768,
    "Attention_QK_MatMul": _qk,
    "Attention_V_MatMul": _av,
}


# OpenFHE backend (native/openfhe/he_ops.cpp): token tensors are packed as 128-token blocks,
# 256 columns per ciphertext; attention as 2 heads (of 64 dimensions) per ciphertext.

def _ofhe_tokens(shape) -> Optional[str]:
    s = tuple(shape)
    if len(s) != 3 or s[0] != 1 or s[1] != 128 or s[2] % 256:
        return f"OpenFHE kernels need [1,128,H] with H a multiple of 256, got {list(s)}"
    return None


def _ofhe_scores(shape) -> Optional[str]:
    s = tuple(shape)
    if len(s) != 4 or s[0] != 1 or s[2:] != (128, 128) or s[1] % 2:
        return f"OpenFHE kernels need scores [1,heads,128,128] with an even number of heads, got {list(s)}"
    return None


def _ofhe_tok(node):
    return _ofhe_tokens(node.input_shape) or _ofhe_tokens(node.output_shape)


def _ofhe_qk(node) -> Optional[str]:
    s = tuple(node.input_shape)
    if len(s) != 4 or s[0] != 3 or s[1] != 1 or s[2] != 128 or s[3] % 128:
        return f"OpenFHE QK^T needs packed [3,1,128,64*heads] with an even number of heads, got {list(s)}"
    return _ofhe_scores(node.output_shape)


def _ofhe_qkv(node) -> Optional[str]:
    o = tuple(node.output_shape)
    if len(o) != 4 or o[0] != 3:
        return f"OpenFHE Q/K/V projections need packed output [3,1,128,H], got {list(o)}"
    return _ofhe_tokens(node.input_shape) or _ofhe_tokens(o[1:])


def _ofhe_av(node) -> Optional[str]:
    return _ofhe_scores(node.input_shape) or _ofhe_tokens(node.output_shape)


OPENFHE_CONTRACTS: Dict[str, Contract] = {
    "Residual_Add": _ofhe_tok, "GeLU": _ofhe_tok, "LayerNorm": _ofhe_tok,
    "Linear_QKV": _ofhe_qkv, "Out_Projection": _ofhe_tok, "FFN_Linear_1": _ofhe_tok, "FFN_Linear_2": _ofhe_tok,
    "Softmax": lambda n: _ofhe_scores(n.input_shape),
    "Attention_QK_MatMul": _ofhe_qk, "Attention_V_MatMul": _ofhe_av,
}


# ----------------------------------------------------------------- HE backends

@dataclass(frozen=True)
class HEBackend:
    name: str
    label: str
    devices: Tuple[str, ...]
    method_prefixes: Tuple[str, ...]      # how this backend's records are named in profiles
    params: str                           # CKKS parameters, in words
    security: str                         # "128-bit" (checked by the library) | "unverified"
    precision_bits: Optional[float]       # -log2 of the max bootstrapping error on random values in [-1,1]
    status: str                           # "measured" | "reference-scaled" | "in development"
    levels: Optional[int] = None          # levels available after bootstrapping (the planner's L)
    # HE levels each operator consumes; None = taken from the profile's measured records
    level_deltas: Optional[Dict[str, int]] = field(default=None, compare=False)
    contracts: Dict[str, Contract] = field(default_factory=dict, compare=False)
    note: str = ""
    # value ranges the approximations are valid on (checked when compile gets a calibration)
    value_contracts: Dict[str, Callable] = field(default_factory=dict, compare=False)
    # op type -> [(variant, value contract, kernel parameters)]: further kernels, used with a
    # calibration when the profile prices them ("<op>.<variant>" records)
    variants: Dict[str, list] = field(default_factory=dict, compare=False)


_OPENFHE_NOTE = ("native/openfhe kernels on 128-token blocks; GeLU, LayerNorm and Softmax use Chebyshev "
                 "approximations on assumed input ranges (GeLU [-8,8], variance [0.25,25], scores [-6,6]); "
                 "HE<->MPC conversion with the hardened protocol of native/openfhe/CONVERSION.md")

HE_BACKENDS: Dict[str, HEBackend] = {
    "nexus": HEBackend(
        "nexus", "NEXUS (SEAL on CPU, Phantom on GPU)", ("cpu", "cuda"), ("nexus", "method_he_nexus"),
        "operators at N=2^16 (20 primes); MatMul at N=2^13; bootstrapping demo at N=2^15 with 8192 slots",
        "unverified", None, "reference-scaled", 20,
        level_deltas=dict(DEFAULT_LEVEL_DELTAS), contracts=NEXUS_CONTRACTS,
        value_contracts=unknown_ranges("NEXUS"),
        note="operator costs are NEXUS reference kernels scaled to each shape; the operators use three "
             "different parameter sets, so they do not run as one pipeline, and the bootstrapping demo's "
             "modulus exceeds the 128-bit bound for N=2^15"),
    "openfhe-fast": HEBackend(
        "openfhe-fast", "OpenFHE CKKS, one-pass bootstrapping", ("cpu",), ("openfhe",),
        "N=2^16, SPARSE_ENCAPSULATED secret, scale 2^50, 8 levels after one-pass bootstrapping (17.6 s)",
        "128-bit", 10.5, "measured", 8, contracts=OPENFHE_CONTRACTS,
        note=_OPENFHE_NOTE, value_contracts=value_contracts(HE_NUMERICS["openfhe-fast"]),
        variants=WIDE_KERNELS),
    "openfhe-precise": HEBackend(
        "openfhe-precise", "OpenFHE CKKS, two-pass bootstrapping", ("cpu",), ("openfhe",),
        "N=2^16, SPARSE_ENCAPSULATED secret, scale 2^50, 7 levels after two-pass bootstrapping (34.8 s)",
        "128-bit", 23.5, "measured", 7, contracts=OPENFHE_CONTRACTS,
        note=_OPENFHE_NOTE, value_contracts=value_contracts(HE_NUMERICS["openfhe-precise"]),
        variants=WIDE_KERNELS),
}


def precision_reason(he: str, min_precision: Optional[float]) -> Optional[str]:
    """Why HE backend `he` does not meet `min_precision` bits (None if it does)."""
    if min_precision is None:
        return None
    b = HE_BACKENDS.get(he)
    if b is None or b.precision_bits is None:
        return f"bootstrapping precision of {he} is not established (min_precision={min_precision:g})"
    if b.precision_bits < min_precision:
        return f"bootstrapping precision {b.precision_bits:g} bits < min_precision={min_precision:g}"
    return None


def infer_he_backend(profile) -> Optional[str]:
    """Name of the HE backend whose costs a profile holds (None if it has no HE records).
    profile.metadata["he_backend"] takes precedence; otherwise record method names are matched
    against HE_BACKENDS (the OpenFHE variants share names, so they need the metadata)."""
    meta = (profile.metadata or {}).get("he_backend")
    if meta:
        return str(meta)
    names = {r.method for r in profile.operators if r.domain == "HE"}
    if not names:
        return None
    found = set()
    for n in names:
        hits = sorted({b.name for b in HE_BACKENDS.values()
                       if any(n.lower().startswith(p) for p in b.method_prefixes)})
        found.add(hits[0] if len(hits) == 1 else "custom")
    return found.pop() if len(found) == 1 else "custom"


def profile_level_rules(profile) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(levels needed, output level if the method bootstraps internally) per op type, from
    the profile's HE records: he_level_delta, and metadata["level_out"]."""
    deltas: Dict[str, int] = {}
    outs: Dict[str, int] = {}
    for r in profile.operators:
        if r.domain != "HE":
            continue
        deltas[r.op_type] = max(deltas.get(r.op_type, 0), int(r.he_level_delta))
        lo = (r.metadata or {}).get("level_out")
        if lo is not None:
            outs[r.op_type] = int(lo)
    return deltas, outs


def builtin_methods(mpc: Optional[str] = "sci", he: str = HE_BACKEND,
                    level_deltas: Optional[Dict[str, int]] = None,
                    level_out: Optional[Dict[str, int]] = None) -> List[Method]:
    """`he` methods plus the `mpc` backend's methods for the canonical op set (mpc=None: HE only).
    HE level consumption comes from the HE backend's table, or from `level_deltas` (measured,
    e.g. profile_level_rules) when the backend has none."""
    if mpc is not None and mpc not in MPC_BACKENDS:
        raise ValueError(f"unknown MPC backend {mpc!r}; known: {sorted(MPC_BACKENDS)}")
    if he not in HE_BACKENDS:
        raise ValueError(f"unknown HE backend {he!r}; known: {sorted(HE_BACKENDS)}")
    hb = HE_BACKENDS[he]
    deltas = level_deltas if level_deltas is not None else hb.level_deltas
    if deltas is None:
        raise ValueError(f"HE backend {he!r} takes its level consumption from a measured profile; "
                         "pass level_deltas (see profile_level_rules)")
    level_out = level_out or {}
    methods = []
    for op in DEFAULT_LEVEL_DELTAS:
        methods.append(Method(op, "HE", he, int(deltas.get(op, 0)), hb.contracts.get(op),
                              level_out.get(op), hb.value_contracts.get(op)))
        if mpc is not None:
            methods.append(Method(op, "MPC", mpc, 0))
    for op, variants in hb.variants.items():
        for name, contract, params in variants:
            key = f"{op}.{name}"
            if key not in deltas:                       # not priced by the profile
                continue
            methods.append(Method(key, "HE", he, int(deltas[key]), hb.contracts.get(op), level_out.get(key),
                                  contract, base=op, params=params))
            if mpc is not None:
                methods.append(Method(key, "MPC", mpc, 0, base=op))
    return methods


def assign_kernels(graph, methods: Iterable[Method], calibration=None):
    """Per node, the first HE method (the base kernel, then its variants) whose contracts hold.
    Returns the graph with the op type of every node that needs a variant replaced by the
    variant's ("GeLU" -> "GeLU.wide"), and node_id -> {"kernel": variant, **parameters}."""
    from dataclasses import replace as _replace
    he: Dict[str, List[Method]] = {}
    for m in methods:
        if m.domain == "HE":
            he.setdefault(m.base or m.op_type, []).append(m)
    kernels: Dict[str, Dict] = {}
    nodes = []
    for n in graph.nodes:
        ms = sorted(he.get(n.op_type, []), key=lambda m: bool(m.base))      # base kernel first
        chosen = next((m for m in ms if m.violation(n, calibration) is None), None)
        if chosen is not None and chosen.base:
            stats = calibration.nodes.get(n.node_id) if calibration is not None else None
            kernels[n.node_id] = {"kernel": chosen.variant, **(chosen.params(n, stats) if chosen.params else {})}
            n = _replace(n, op_type=chosen.op_type)
        nodes.append(n)
    if not kernels:
        return graph, kernels
    return _replace(graph, nodes=nodes), kernels


def level_deltas(methods: Iterable[Method]) -> Dict[str, int]:
    """Level rules for the planner: op_type -> levels needed, with ``.out`` for methods that
    bootstrap internally (op_type -> output level)."""
    from compiler.state_expanded.state_graph import LevelRules
    he = [m for m in methods if m.domain == "HE"]
    return LevelRules({m.op_type: m.level_delta for m in he},
                      {m.op_type: m.level_out for m in he if m.level_out is not None})


def supported_ops(methods: Iterable[Method]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for m in methods:
        out.setdefault(m.op_type, set()).add(m.domain)
    return out


def violations(graph, methods: Iterable[Method], calibration=None) -> Dict[Tuple[str, str], str]:
    """(node_id, domain) -> reason, for every node a method's contract rules out (with a
    calibration, also the methods' value contracts)."""
    by_key: Dict[Tuple[str, str], List[Method]] = {}
    variants: Dict[Tuple[str, str], List[Method]] = {}
    for m in methods:
        by_key.setdefault((m.op_type, m.domain), []).append(m)
        if m.base:
            variants.setdefault((m.base, m.domain), []).append(m)
    out: Dict[Tuple[str, str], str] = {}
    for n in graph.nodes:
        for d in ("HE", "MPC"):
            ms = by_key.get((n.op_type, d))
            if not ms:
                continue
            extra = variants.get((n.op_type, d), [])        # a node no kernel variant covers either
            reasons = [m.violation(n, calibration) for m in ms + extra]
            if all(reasons):          # no method of this domain accepts the node
                out[(n.node_id, d)] = reasons[-1] if extra else reasons[0]
    return out

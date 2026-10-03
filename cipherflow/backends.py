"""Backend method declarations (planning side only in v1).

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

# Returns None when the node satisfies the contract, else the reason it does not.
Contract = Callable[[object], Optional[str]]

# Structural operators with no backend method: "Input" is the model input when it
# feeds more than one operator (e.g. the first residual). Its domain in the plan
# says how the client provides the input (HE: encrypted, MPC: secret-shared).
PSEUDO_OPS = ("Input",)


@dataclass(frozen=True)
class Method:
    op_type: str
    domain: str            # "HE" | "MPC"
    backend: str
    level_delta: int = 0   # levels consumed in HE; 0 for MPC
    contract: Optional[Contract] = field(default=None, compare=False, repr=False)

    def violation(self, node) -> Optional[str]:
        return self.contract(node) if self.contract is not None else None


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
    "LayerNorm": _tokens_768,
    "FFN_Linear_1": _tokens_768,
    "Attention_QK_MatMul": _qk,
    "Attention_V_MatMul": _av,
}


def builtin_methods() -> List[Method]:
    """NEXUS-CKKS (HE) and SCI/SHAFT (MPC) methods for the canonical op set."""
    methods = []
    for op, delta in DEFAULT_LEVEL_DELTAS.items():
        methods.append(Method(op, "HE", "nexus_ckks", delta, NEXUS_CONTRACTS.get(op)))
        methods.append(Method(op, "MPC", "sci_shaft", 0))
    return methods


def level_deltas(methods: Iterable[Method]) -> Dict[str, int]:
    return {m.op_type: m.level_delta for m in methods if m.domain == "HE"}


def supported_ops(methods: Iterable[Method]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for m in methods:
        out.setdefault(m.op_type, set()).add(m.domain)
    return out


def violations(graph, methods: Iterable[Method]) -> Dict[Tuple[str, str], str]:
    """(node_id, domain) -> reason, for every node a method's contract rules out."""
    by_key: Dict[Tuple[str, str], List[Method]] = {}
    for m in methods:
        by_key.setdefault((m.op_type, m.domain), []).append(m)
    out: Dict[Tuple[str, str], str] = {}
    for n in graph.nodes:
        for d in ("HE", "MPC"):
            ms = by_key.get((n.op_type, d))
            if not ms:
                continue
            reasons = [m.violation(n) for m in ms]
            if all(reasons):          # no method of this domain accepts the node
                out[(n.node_id, d)] = reasons[0]
    return out

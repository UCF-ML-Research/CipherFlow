"""Backends given by their primitives: CipherFlow lowers the model's operators to them.

A PrimitiveBackend declares one domain's primitives and what one call of each costs. CipherFlow writes
every operator of the model as a program over those primitives (a kernel template), counts the calls
and, in HE, the CKKS levels, and prices the operator from the counts. The backend then plans like any
other, as an Extension:

    he = cf.PrimitiveBackend.he("my-ckks", slots=32768, levels=10, bootstrap=cf.Cost(9_000),
                                primitives={"add": cf.Cost(0.3), "mul_plain": cf.Cost(3),
                                            "mul": cf.Cost(40), "rotate": cf.Cost(35)})
    print(he.lower(graph))                      # every operator as primitive calls, levels and cost
    plan = cf.compile(graph, deployment, extensions=[he.extension()])

HE primitives cost one call on a ciphertext of `slots` values. MPC primitives cost one element
(local_ms and comm_bytes per element), and comm_rounds per call, a call working on all its elements
at once. The kernels are standard constructions with default parameters (`params`: polynomial
degrees, iteration counts) on a simple packing (tokens packed along the hidden dimension). That the
primitives are correct, precise enough on the model's values, and secure is the user's to ensure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .extensions import Cost, Extension, numel

HE_PRIMITIVES = ("add", "mul_plain", "mul", "rotate")
MPC_PRIMITIVES = ("add", "mul", "matmul", "trunc", "cmp")
FREE = Cost(0.0)                        # primitives a backend may leave out: they cost nothing

DEFAULT_PARAMS = {
    "exp_squarings": 8,                 # exp(x) ~ (1 + x / 2^r)^(2^r)
    "inverse_iterations": 4,            # Newton steps for 1 / x
    "rsqrt_init_degree": 3,             # initial 1 / sqrt(x) polynomial
    "rsqrt_iterations": 3,              # Newton steps for 1 / sqrt(x)
    "he_gelu_degree": 27,               # GeLU polynomial in HE (Paterson-Stockmeyer)
    "mpc_gelu_degree": 6,               # GeLU polynomial on the middle piece in MPC (Horner)
}
LINEAR = ("Linear_QKV", "Out_Projection", "FFN_Linear_1", "FFN_Linear_2")
OPERATORS = LINEAR + ("Attention_QK_MatMul", "Attention_V_MatMul", "Residual_Add", "Softmax", "LayerNorm", "GeLU")


@dataclass
class Kernel:
    """One operator as primitive calls: counts per primitive, sequential calls (MPC rounds), CKKS levels."""
    template: str
    counts: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)
    levels: int = 0

    def add(self, prim: str, count: float, calls: int = 1) -> "Kernel":
        self.counts[prim] = self.counts.get(prim, 0) + count
        self.calls[prim] = self.calls.get(prim, 0) + calls
        return self


def _log2(n) -> int:
    return max(1, math.ceil(math.log2(max(2, n))))


# ----------------------------------------------------------------- HE kernels (CKKS)

def _he_kernel(op: str, node, slots: int, p: Dict) -> Kernel:
    i, o = tuple(node.input_shape), tuple(node.output_shape)
    ct = lambda n: math.ceil(n / slots)                           # noqa: E731
    if op == "Residual_Add":
        return Kernel("element-wise addition").add("add", ct(numel(o)))
    if op in LINEAR:
        din, dout = i[-1], (3 * o[-1] if op == "Linear_QKV" else o[-1])
        tokens = numel(i) // din
        ct_in = math.ceil(tokens / max(1, slots // din))
        blocks = max(1, math.ceil(dout / din))
        k = Kernel(f"BSGS diagonal matrix-vector product, {din} diagonals x {blocks} block(s)", levels=1)
        k.add("mul_plain", ct_in * blocks * din).add("add", ct_in * blocks * din)
        return k.add("rotate", ct_in * blocks * 2 * math.ceil(math.sqrt(din)))
    if op in ("Attention_QK_MatMul", "Attention_V_MatMul"):
        if op == "Attention_QK_MatMul":
            heads, seq, d = o[1], o[2], i[-1] // o[1]
        else:
            heads, seq, d = i[1], i[2], o[-1] // i[1]
        macs = (numel(o) if op == "Attention_QK_MatMul" else numel(i)) * d
        mul = math.ceil(macs / slots)
        out = ct(numel(o))
        k = Kernel("ciphertext x ciphertext product: SIMD multiplications, rotations to align and sum", levels=2)
        return k.add("mul", mul).add("rotate", mul + out * _log2(d)).add("add", mul).add("mul_plain", out)
    if op == "Softmax":
        c, rows_len = ct(numel(i)), i[-1]
        r, it = p["exp_squarings"], p["inverse_iterations"]
        k = Kernel(f"exp by {r} squarings, row sums by rotation, 1/x by {it} Newton steps")
        k.add("mul_plain", 2 * c).add("mul", c * (r + 2 * it + 1))
        k.add("rotate", c * _log2(rows_len)).add("add", c * (_log2(rows_len) + it))
        k.levels = r + 1 + 1 + 2 * it + 1
        return k
    if op == "LayerNorm":
        c, h = ct(numel(i)), i[-1]
        d0, it = p["rsqrt_init_degree"], p["rsqrt_iterations"]
        k = Kernel(f"mean and variance by rotation, 1/sqrt by a degree-{d0} start and {it} Newton steps")
        k.add("rotate", 2 * c * _log2(h)).add("add", 2 * c * _log2(h) + c)
        k.add("mul_plain", 2 * c).add("mul", c * (1 + (d0 - 1) + 3 * it + 1))
        k.levels = 1 + 1 + 1 + (d0 - 1) + 3 * it + 1
        return k
    if op == "GeLU":
        c, deg = ct(numel(i)), p["he_gelu_degree"]
        nonscalar = math.ceil(math.sqrt(2 * deg)) + _log2(deg)
        k = Kernel(f"degree-{deg} polynomial, Paterson-Stockmeyer ({nonscalar} ciphertext products)")
        k.add("mul", c * nonscalar).add("mul_plain", c * deg).add("add", c * deg)
        k.levels = _log2(deg + 1) + 1
        return k
    raise KeyError(op)


def _he_needs(op: str) -> Tuple[str, ...]:
    if op == "Residual_Add":
        return ("add",)
    if op in LINEAR:
        return ("mul_plain", "rotate")
    if op == "GeLU":
        return ("mul", "mul_plain")
    return ("mul", "mul_plain", "rotate")


# ----------------------------------------------------------------- MPC kernels (secret sharing, fixed point)

def _mpc_kernel(op: str, node, prims: Dict[str, Cost], p: Dict) -> Kernel:
    i, o = tuple(node.input_shape), tuple(node.output_shape)
    n_in, n_out = numel(i), numel(o)
    if op == "Residual_Add":
        return Kernel("local addition of shares").add("add", n_out, 0)
    if op in LINEAR:
        din, dout = i[-1], (3 * o[-1] if op == "Linear_QKV" else o[-1])
        macs = numel(i) // din * din * dout
        prim = "matmul" if "matmul" in prims else "mul"
        k = Kernel(f"shares x the server's weights ({prim} per multiply-accumulate), then truncation")
        return k.add(prim, macs).add("trunc", n_out)
    if op in ("Attention_QK_MatMul", "Attention_V_MatMul"):
        d = (i[-1] // o[1]) if op == "Attention_QK_MatMul" else (o[-1] // i[1])
        macs = (n_out if op == "Attention_QK_MatMul" else n_in) * d
        return Kernel("shares x shares (mul per multiply-accumulate), then truncation").add("mul", macs).add("trunc", n_out)
    if op == "Softmax":
        rows, length = n_in // i[-1], i[-1]
        r, it = p["exp_squarings"], p["inverse_iterations"]
        k = Kernel(f"row max by a comparison tree, exp by {r} squarings, 1/x by {it} Newton steps")
        k.add("cmp", n_in, _log2(length)).add("mul", n_in, _log2(length))           # tree: select the larger
        k.add("mul", n_in * r, r).add("trunc", n_in * r, r)
        k.add("mul", rows * 2 * it + n_in, 2 * it + 1).add("trunc", rows * 2 * it + n_in, 2 * it + 1)
        return k
    if op == "LayerNorm":
        rows = n_in // i[-1]
        d0, it = p["rsqrt_init_degree"], p["rsqrt_iterations"]
        k = Kernel(f"mean (local), variance, 1/sqrt by a degree-{d0} start and {it} Newton steps")
        k.add("mul", n_in, 1).add("trunc", n_in, 1)
        k.add("mul", rows * ((d0 - 1) + 3 * it), (d0 - 1) + 3 * it).add("trunc", rows * ((d0 - 1) + 3 * it), (d0 - 1) + 3 * it)
        return k.add("mul", n_in, 1).add("trunc", n_in, 1)
    if op == "GeLU":
        deg = p["mpc_gelu_degree"]
        k = Kernel(f"two comparisons pick the piece; degree-{deg} polynomial (Horner) on the middle piece")
        k.add("cmp", 2 * n_in, 1).add("mul", n_in * deg, deg).add("trunc", n_in * deg, deg)
        return k.add("mul", 2 * n_in, 1)                                              # select between pieces
    raise KeyError(op)


def _mpc_needs(op: str, prims) -> Tuple[str, ...]:
    if op == "Residual_Add":
        return ()
    if op in LINEAR:
        return ("matmul",) if "matmul" in prims else ("mul",)
    if op in ("Softmax", "GeLU"):
        return ("cmp", "mul")
    return ("mul",)


# ----------------------------------------------------------------- the backend

@dataclass
class Lowered:
    node_id: str
    op_type: str
    kernel: Optional[Kernel]
    cost: Optional[Cost]
    bootstraps: int = 0
    level_out: Optional[int] = None
    missing: Tuple[str, ...] = ()
    source: Optional[str] = None


@dataclass
class Lowering:
    backend: str
    domain: str
    ops: List[Lowered]

    def __str__(self) -> str:
        lines = [f"{self.backend} ({self.domain}): every operator as primitive calls"]
        for x in self.ops:
            where = x.source or x.node_id
            if x.kernel is None:
                why = f"missing {', '.join(x.missing)}" if x.missing else "no kernel template"
                lines.append(f"  {x.op_type:20s} {where:28s} cannot run here: {why}")
                continue
            calls = ", ".join(f"{k} {v:,.0f}" for k, v in x.kernel.counts.items() if v)
            extra = f", bootstrap {x.bootstraps:,}" if x.bootstraps else ""
            lv = f"  levels {x.kernel.levels}" if self.domain == "HE" else ""
            rounds = f"  rounds {x.cost.comm_rounds:,}" if self.domain == "MPC" else ""
            lines.append(f"  {x.op_type:20s} {where:28s} {x.cost.local_ms / 1e3:9.2f} s{rounds}{lv}  [{calls}{extra}]")
            lines.append(f"  {'':20s} {'':28s} {x.kernel.template}")
        return "\n".join(lines)


@dataclass
class PrimitiveBackend:
    name: str
    domain: str                                   # "HE" | "MPC"
    primitives: Dict[str, Cost]
    slots: Optional[int] = None                   # HE: values per ciphertext
    levels: Optional[int] = None                  # HE: levels after bootstrapping (the level budget)
    bootstrap: Optional[Cost] = None              # HE: one ciphertext's bootstrapping
    params: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def he(cls, name: str, *, slots: int, levels: int, primitives: Dict[str, Cost],
           bootstrap: Optional[Cost] = None, params: Optional[Dict[str, int]] = None) -> "PrimitiveBackend":
        unknown = sorted(set(primitives) - set(HE_PRIMITIVES))
        if unknown:
            raise ValueError(f"unknown HE primitives {unknown}; known: {list(HE_PRIMITIVES)}")
        return cls(name, "HE", dict(primitives), int(slots), int(levels), bootstrap, dict(params or {}))

    @classmethod
    def mpc(cls, name: str, *, primitives: Dict[str, Cost], params: Optional[Dict[str, int]] = None) -> "PrimitiveBackend":
        unknown = sorted(set(primitives) - set(MPC_PRIMITIVES))
        if unknown:
            raise ValueError(f"unknown MPC primitives {unknown}; known: {list(MPC_PRIMITIVES)}")
        return cls(name, "MPC", dict(primitives), params=dict(params or {}))

    # -------------------------------------------------------------- lowering
    def _params(self) -> Dict[str, int]:
        return {**DEFAULT_PARAMS, **self.params}

    def lower_node(self, node) -> Lowered:
        op = node.op_type
        src = (node.attributes or {}).get("source") if hasattr(node, "attributes") else None
        if op not in OPERATORS:
            return Lowered(node.node_id, op, None, None, source=src)
        p = self._params()
        if self.domain == "HE":
            missing = tuple(m for m in _he_needs(op) if m not in self.primitives)
            if missing:
                return Lowered(node.node_id, op, None, None, missing=missing, source=src)
            k = _he_kernel(op, node, self.slots, p)
            boots, level_out = 0, None
            if k.levels > self.levels:                         # the kernel refreshes its ciphertexts itself
                if self.bootstrap is None:
                    return Lowered(node.node_id, op, None, None, missing=("bootstrap",), source=src)
                rounds = math.ceil(k.levels / self.levels) - 1
                boots = rounds * math.ceil(numel(node.input_shape) / self.slots)
                level_out = self.levels - (k.levels - rounds * self.levels)
            cost = _sum(k, self.primitives, mpc=False)
            if boots:
                cost = Cost(cost.local_ms + boots * self.bootstrap.local_ms,
                            cost.comm_bytes + boots * self.bootstrap.comm_bytes, cost.comm_rounds + self.bootstrap.comm_rounds)
            return Lowered(node.node_id, op, k, cost, boots, level_out, source=src)
        missing = tuple(m for m in _mpc_needs(op, self.primitives) if m not in self.primitives)
        if missing:
            return Lowered(node.node_id, op, None, None, missing=missing, source=src)
        k = _mpc_kernel(op, node, self.primitives, p)
        return Lowered(node.node_id, op, k, _sum(k, self.primitives, mpc=True), source=src)

    def lower(self, graph) -> Lowering:
        """Every operator of the graph as primitive calls, with its levels and cost."""
        return Lowering(self.name, self.domain,
                        [self.lower_node(n) for n in graph.nodes if n.op_type != "Input"])

    def extension(self) -> Extension:
        """The backend as an Extension: one primitive per operator, priced by its lowering. An
        operator the backend cannot lower (a primitive it lacks) is kept out of its domain."""
        ext = Extension(self.name, note=f"operators lowered to the primitives of {self.name}")
        if self.domain == "HE":
            ext.he_levels, ext.bootstrap, ext.bootstrap_slots = self.levels, self.bootstrap, self.slots
        for op in OPERATORS:
            levels, level_out = self._he_levels(op) if self.domain == "HE" else (0, None)
            ext.primitive(op, self.domain, levels=levels, level_out=level_out,
                          cost=lambda node, profile: self._cost(node),
                          accepts=lambda node: self._accepts(node))
        return ext

    def _he_levels(self, op: str) -> Tuple[int, Optional[int]]:
        """(levels the kernel needs on input, its output level when it bootstraps inside)."""
        from types import SimpleNamespace
        probe = SimpleNamespace(op_type=op, node_id=op, input_shape=_PROBE[op][0], output_shape=_PROBE[op][1],
                                attributes={})
        x = self.lower_node(probe)
        if x.kernel is None:
            return 0, None
        return min(x.kernel.levels, self.levels), x.level_out

    def _cost(self, node) -> Cost:
        x = self.lower_node(node)
        return x.cost if x.cost is not None else Cost(0.0)

    def _accepts(self, node) -> Optional[str]:
        x = self.lower_node(node)
        if x.kernel is not None:
            return None
        return (f"{self.name} lacks {', '.join(x.missing)}" if x.missing else f"{self.name} has no kernel for {node.op_type}")


# shapes to read a kernel's level use from (it does not depend on the shape)
_T, _QKV, _SC, _F = (1, 128, 768), (3, 1, 128, 768), (1, 12, 128, 128), (1, 128, 3072)
_PROBE = {"Linear_QKV": (_T, _QKV), "Out_Projection": (_T, _T), "FFN_Linear_1": (_T, _F), "FFN_Linear_2": (_F, _T),
          "Attention_QK_MatMul": (_QKV, _SC), "Attention_V_MatMul": (_SC, _T), "Residual_Add": (_T, _T),
          "Softmax": (_SC, _SC), "LayerNorm": (_T, _T), "GeLU": (_F, _F)}


def _sum(k: Kernel, prims: Dict[str, Cost], *, mpc: bool) -> Cost:
    local = sum(n * prims.get(p, FREE).local_ms for p, n in k.counts.items())
    sent = sum(n * prims.get(p, FREE).comm_bytes for p, n in k.counts.items())
    rounds = sum(c * prims.get(p, FREE).comm_rounds for p, c in k.calls.items()) if mpc else \
        sum(math.ceil(n) * prims.get(p, FREE).comm_rounds for p, n in k.counts.items())
    return Cost(float(local), int(sent), int(rounds))

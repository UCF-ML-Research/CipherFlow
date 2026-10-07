"""Execute a plan on real data with the backends' numerical models (see numerics.py).

    graph = cf.models.from_torch(encoder, x)
    plan  = cf.compile(graph, deployment)
    res   = cf.runtime.emulate(plan, encoder, x)
    print(res.summary())          # error against the plaintext model, plan checks, approximation ranges

The plan is checked first (check_plan): on every edge, the plan's actions (conversions and
bootstraps, each recorded with the edge it applies to) must take the producer's output to the
consumer's domain, and in HE to at least the level the consumer starts at; every action must belong
to an edge. Then the traced model runs in float64, operator by operator, each in its planned domain,
with each edge's actions applied to the values crossing it, so V reaches Attn*V the way the plan
routes it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.fx as fx

from ..errors import UnsupportedModel
from ..frontend import trace_torch
from . import numerics as nm

_CLIENT = "client"     # the model input before the plan's Input node (or first operator) takes it


# ---------------------------------------------------------------- plan checks

def edge_actions(plan) -> Dict[tuple, list]:
    """(producer, consumer) -> the plan's actions on that edge; (producer, "*") -> the actions on the
    producer's output for all its consumers (at a region's exit), applied first."""
    out: Dict[tuple, list] = {}
    for a in plan.actions:
        out.setdefault(tuple(a.edge) if a.edge else (None, a.before), []).append(a)
    return out


def check_plan(plan, graph) -> Dict[str, Any]:
    """Follow every edge through the plan's actions: the producer's output must reach the consumer's
    domain, and in HE at least the level the consumer starts at."""
    dom = plan.domains()
    lv = {n.id: (n.level_in, n.level_out) for n in plan.nodes}
    L = plan.profile.get("he_level_budget")
    min_conv = plan.profile.get("he_to_mpc_min_level", 0) or 0
    min_boot = plan.profile.get("bootstrap_min_level", 0) or 0
    scaled = plan.profile.get("scaled_bootstrap") or {}
    if {n.node_id for n in graph.nodes} != set(dom):
        raise ValueError("the plan was compiled for a different graph (node ids differ)")
    acts = edge_actions(plan)
    problems: List[str] = []
    used = set()
    edges = {(e.src, e.dst) for e in graph.edges}
    for u, v in sorted(edges):
        d, lvl = dom[u], lv[u][1]
        for key in ((u, "*"), (u, v)):
            for a in acts.get(key, []):
                used.add(id(a))
                if a.kind == "convert":
                    if a.src != d:
                        problems.append(f"{u} -> {v}: converts from {a.src} but the value is in {d}")
                    if d == "HE" and a.dst == "MPC" and lvl is not None and lvl < min_conv:
                        problems.append(f"{u} -> {v}: converts to MPC at level {lvl}, the conversion needs {min_conv}")
                    d, lvl = a.dst, (L if a.dst == "HE" else None)
                else:
                    if d != "HE":
                        problems.append(f"{u} -> {v}: bootstraps a value in {d}")
                    need = min_boot + (1 if u in scaled else 0)
                    if d == "HE" and lvl is not None and lvl < need:
                        problems.append(f"{u} -> {v}: bootstraps at level {lvl}, bootstrapping needs {need}"
                                        + (f" (scaled by 1/{scaled[u]:g})" if u in scaled else ""))
                    lvl = L
        if d != dom[v]:
            problems.append(f"{u} -> {v}: arrives in {d}, {v} runs in {dom[v]}")
        elif d == "HE" and lv[v][0] is not None and (lvl is None or lvl < lv[v][0]):
            problems.append(f"{u} -> {v}: arrives at level {lvl}, {v} starts at {lv[v][0]}")
    for a in plan.actions:
        if id(a) not in used:
            problems.append(f"{a.kind} before {a.before} on {a.edge or 'no recorded edge'} is on no edge of the graph")
    k = plan.counts()
    return {"conversions": k["conversions"], "bootstraps": k["bootstraps"], "problems": problems}


# ---------------------------------------------------------------- execution

@dataclass
class EmulationResult:
    output: torch.Tensor
    reference: torch.Tensor
    checks: Dict[str, Any]
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)    # per HE non-linear operator
    applied: Dict[str, int] = field(default_factory=dict)              # conversions / bootstraps executed

    @property
    def max_abs_err(self) -> float:
        return float((self.output - self.reference).abs().max())

    @property
    def rel_err(self) -> float:
        return float((self.output - self.reference).norm() / self.reference.norm())

    def out_of_range(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self.nodes.items()
                if v.get("outside", 0) > 0 or v.get("u_below_umin", 0) > 0 or not v.get("bootstrap_in_range", True)
                or not v.get("conversion_in_bound", True) or v.get("sums_outside", 0) > 0}

    def summary(self) -> str:
        lines = [f"output vs plaintext: max |err| {self.max_abs_err:.3g}, relative {self.rel_err:.3g}",
                 f"conversions {self.applied.get('conversions', 0)}, bootstraps {self.applied.get('bootstraps', 0)}"
                 f" executed (plan: {self.checks['conversions']} / {self.checks['bootstraps']})"]
        lines += [f"plan check: {p}" for p in self.checks["problems"]]
        for nid, s in self.out_of_range().items():
            lines.append(f"{nid}: {s.get('quantity', 'input')} in [{s['input_min']:.3g}, {s['input_max']:.3g}], "
                         f"interval {s['interval']}, {100 * s.get('outside', 0):.2g}% outside"
                         + (f", {100 * s['u_below_umin']:.2g}% of row sums below umin" if s.get("u_below_umin") else "")
                         + ("" if s.get("bootstrap_in_range", True) else ", bootstrapped values out of range")
                         + ("" if s.get("conversion_in_bound", True) else ", converted values above the protocol's bound"))
        return "\n".join(lines)


class _Run(fx.Interpreter):
    def __init__(self, tr, plan, he: nm.HENumerics, frac: int, seed: int, sigma: float):
        super().__init__(tr.gm)
        self.tr, self.he, self.frac, self.sigma = tr, he, frac, sigma
        self.dom = plan.domains()
        self.kernels = {n.id: n.kernel for n in plan.nodes if getattr(n, "kernel", None)}
        self.scaled = dict(plan.profile.get("scaled_bootstrap") or {})
        self.acts = edge_actions(plan)
        self.gen = torch.Generator().manual_seed(seed)
        self.origin: Dict[str, Optional[str]] = {}       # fx node -> CipherFlow node whose value it carries
        self.arrived: Dict[tuple, torch.Tensor] = {}
        self.exited: Dict[str, torch.Tensor] = {}        # fx value after its producer's region-exit actions
        self.done = set()                                # actions executed (each counted once)
        self.applied = {"conversions": 0, "bootstraps": 0}
        self.stats: Dict[str, Dict[str, Any]] = {}

    def _apply(self, actions, val, where: str):
        for a in actions:
            if a.kind == "convert":
                peak = float(val.abs().max()) if val.numel() else 0.0
                if peak >= 2.0 ** nm.CONVERSION_BOUND_BITS:
                    self.stats.setdefault(where, {}).update(
                        conversion_in_bound=False, input_min=float(val.min()), input_max=float(val.max()),
                        interval=[-2.0 ** nm.CONVERSION_BOUND_BITS, 2.0 ** nm.CONVERSION_BOUND_BITS])
                val = nm.to_mpc(val, self.frac, self.gen) if a.dst == "MPC" else nm.to_he(val, self.sigma, self.gen)
            else:
                val, ok = nm.bootstrap(val, self.he, self.gen, self.scaled.get(a.edge[0], 1.0) if a.edge else 1.0)
                if not ok:
                    self.stats.setdefault(where, {}).update(bootstrap_in_range=False, input_min=float(val.min()),
                                                            input_max=float(val.max()), interval=[-64, 64])
            if id(a) not in self.done:
                self.done.add(id(a))
                self.applied["conversions" if a.kind == "convert" else "bootstraps"] += 1
        return val

    # values crossing an edge of the plan: conversion or bootstrap where the plan has one
    def _arrive(self, a: fx.Node, consumer: str):
        val = self.env[a]
        src = self.origin.get(a.name)
        if src is None or not isinstance(val, torch.Tensor) or not val.is_floating_point():
            return val
        key = (a.name, consumer)
        if key in self.arrived:
            return self.arrived[key]
        dv = self.dom[consumer]
        if src == _CLIENT:                                   # the client provides it in that domain
            out = nm.fixed_point(val, self.frac) if dv == "MPC" else val
        elif src == consumer:
            out = val
        else:
            if a.name not in self.exited:                    # region exit: once, shared by all consumers
                self.exited[a.name] = self._apply(self.acts.get((src, "*"), []), val, consumer)
            out = self._apply(self.acts.get((src, consumer), []), self.exited[a.name], consumer)
        self.arrived[key] = out
        return out

    def run_node(self, n: fx.Node):
        kind, owner = self.tr.kind.get(n.name), self.tr.owner.get(n.name)
        if n.op == "placeholder":
            out = super().run_node(n)
            self.origin[n.name] = _CLIENT
            if owner is not None:                            # the graph's Input node takes the input
                out = self._client_to(owner, out)
                self.origin[n.name] = owner
            return out
        if kind != "op":
            out = super().run_node(n)
            srcs = [self.origin.get(a.name) for a in n.all_input_nodes if self.origin.get(a.name)]
            self.origin[n.name] = srcs[0] if srcs else None
            return out
        if owner is None or owner.startswith("MHA:"):
            raise UnsupportedModel(f"{n.name}: emulating nn.MultiheadAttention is not supported yet; write the "
                                   "attention with explicit Q/K/V projections")
        args = fx.node.map_arg(n.args, lambda a: self._arrive(a, owner))
        kwargs = fx.node.map_arg(n.kwargs, lambda a: self._arrive(a, owner))
        out = self._compute(n, self.tr.op[n.name], self.dom[owner], owner, args, kwargs)
        self.origin[n.name] = owner
        return out

    def _client_to(self, owner, val):
        return nm.fixed_point(val, self.frac) if self.dom[owner] == "MPC" and val.is_floating_point() else val

    def _call(self, n, args, kwargs):
        return getattr(self, n.op)(n.target, args, kwargs)

    def _compute(self, n, op, dom, owner, args, kwargs):
        if dom == "MPC":
            return nm.fixed_point(self._call(n, args, kwargs), self.frac)
        if op in ("Linear", "MatMul", "Residual_Add"):
            return self._call(n, args, kwargs)
        x = args[0]
        st = self.stats.setdefault(owner, {})
        if op == "GeLU":
            k = self.kernels.get(owner)
            if k and k.get("kernel") == "wide":
                return nm.he_gelu_wide(x, k["bound"], self.he, self.gen, st)
            return nm.he_gelu(x, self.he, st)
        if op == "LayerNorm":
            if n.op == "call_module":
                m = self.fetch_attr(n.target)
                w, b = m.weight, m.bias
            else:
                w = kwargs.get("weight", args[2] if len(args) > 2 else None)
                b = kwargs.get("bias", args[3] if len(args) > 3 else None)
            k = self.kernels.get(owner)
            if k and k.get("kernel") in ("wide", "wider"):
                eps = (self.fetch_attr(n.target).eps if n.op == "call_module"
                       else kwargs.get("eps", args[4] if len(args) > 4 else 1e-5))
                return nm.he_layernorm_wide(x, w, b, k, self.he, self.gen, st, eps)
            return nm.he_layernorm(x, w, b, self.he, self.gen, st)
        if op == "Softmax":
            dim = (self.fetch_attr(n.target).dim if n.op == "call_module"
                   else kwargs.get("dim", args[1] if len(args) > 1 else -1))
            if dim not in (-1, x.dim() - 1):
                raise UnsupportedModel(f"{n.name}: softmax over axis {dim}; the HE kernel normalizes the last axis")
            k = self.kernels.get(owner)
            if k and k.get("kernel") == "wide":
                return nm.he_softmax_wide(x, k, self.he, self.gen, st)
            return nm.he_softmax(x, self.he, self.gen, st)
        return self._call(n, args, kwargs)


class Emulator:
    """A plan applied to a model, reusable over many inputs (traces are cached per input shape).

        em = Emulator(plan, encoder)
        for x in batch_of_hidden_states:
            res = em(x)
    """

    def __init__(self, plan, module, *, concrete_args=None, seed: int = 0,
                 he_numerics: Optional[nm.HENumerics] = None, mpc_frac_bits: Optional[int] = None):
        self.plan, self.concrete_args = plan, concrete_args
        self.module = copy.deepcopy(module).double().eval()
        self.qkv = any(n.op_type == "Linear_QKV" for n in plan.nodes)
        self.he = he_numerics or nm.HE_NUMERICS.get(plan.he_backend or "openfhe-fast", nm.HE_NUMERICS["openfhe-fast"])
        self.frac = mpc_frac_bits or nm.MPC_FRAC_BITS.get(plan.mpc_backend or "sci", 12)
        self.sigma = nm.MPC_TO_HE_SIGMA.get(plan.mpc_backend or "sci", 4.0e-5)
        self.seed, self.calls = seed, 0
        self._traces: Dict[tuple, Any] = {}
        self._checks: Optional[Dict[str, Any]] = None

    def __call__(self, example_inputs) -> EmulationResult:
        inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
        inputs = tuple(x.double() if isinstance(x, torch.Tensor) and x.is_floating_point() else x for x in inputs)
        key = tuple(tuple(x.shape) if isinstance(x, torch.Tensor) else None for x in inputs)
        if key not in self._traces:
            self._traces[key] = trace_torch(self.module, inputs, concrete_args=self.concrete_args,
                                            qkv_projection=self.qkv)
            if self._checks is None:
                self._checks = check_plan(self.plan, self._traces[key].graph)
        tr = self._traces[key]
        args = list(inputs) + list(tr.run_args[len(inputs):])
        with torch.no_grad():
            reference = fx.Interpreter(tr.gm).run(*args)
            run = _Run(tr, self.plan, self.he, self.frac, self.seed + self.calls, self.sigma)
            output = run.run(*args)
        self.calls += 1
        first = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        return EmulationResult(first(output), first(reference), self._checks, run.stats, run.applied)


def emulate(plan, module, example_inputs, *, concrete_args=None, seed: int = 0,
            he_numerics: Optional[nm.HENumerics] = None, mpc_frac_bits: Optional[int] = None) -> EmulationResult:
    """Run `module` on `example_inputs` as `plan` places it (see the module docstring).
    he_numerics / mpc_frac_bits override the backends' numerical models (e.g. the bootstrapping
    noise measured on another machine: dataclasses.replace(HE_NUMERICS["openfhe-fast"], bootstrap_sigma=...)).
    For many inputs, use Emulator (it reuses the traced model)."""
    return Emulator(plan, module, concrete_args=concrete_args, seed=seed, he_numerics=he_numerics,
                    mpc_frac_bits=mpc_frac_bits)(example_inputs)

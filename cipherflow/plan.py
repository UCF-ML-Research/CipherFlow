"""Plan: the serializable output of cipherflow.compile (schema cipherflow.plan/v1)."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from compiler.state_expanded.cost_model import StateExpandedCostModel
from compiler.state_expanded.planner import CompiledPlan

SCHEMA = "cipherflow.plan/v1"


@dataclass
class NodePlacement:
    id: str
    op_type: str
    domain: str                      # "HE" | "MPC"
    level_in: Optional[int] = None   # HE only
    level_out: Optional[int] = None
    cost_source: Optional[str] = None  # "exact" profile record, "scaled" from the nearest shape, or "input"
    backend: Optional[str] = None      # the HE or MPC backend name, or "client" (Input)
    kernel: Optional[Dict] = None      # HE kernel variant and its parameters, e.g. {"kernel": "wide", "bound": 64}
    source: Optional[str] = None       # where the operator is in the PyTorch model (from_torch), e.g.
                                       # "blocks.0.attn.{q,k,v}", "blocks.0.attn: softmax"


@dataclass
class Action:
    kind: str                        # "convert" | "bootstrap"
    before: str                      # node id the action precedes ("<output>" at the end)
    src: Optional[str] = None        # convert: "HE"/"MPC"
    dst: Optional[str] = None
    tensor_shape: Optional[List[int]] = None
    level_in: Optional[int] = None   # bootstrap: level being refreshed
    cost_ms: float = 0.0
    edge: Optional[List[str]] = None  # [producer, consumer] the action applies to; consumer "*": the
                                      # producer's output, for all its consumers (at a region's exit)


def _edge(detail: str) -> Optional[List[str]]:
    m = re.search(r"edge=(\S+?)->(\S+)", detail)
    return [m.group(1), m.group(2)] if m else None


@dataclass
class Plan:
    graph_id: str
    deployment: Dict
    profile: Dict
    strategy: str
    nodes: List[NodePlacement]
    actions: List[Action]
    cost: Dict
    schema: str = SCHEMA
    edges: List[List[str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    he_backend: Optional[str] = None   # HE backend this plan uses
    precision_bits: Optional[float] = None   # its bootstrapping precision (None: not established)
    mpc_backend: Optional[str] = None  # MPC backend this plan uses
    security: Optional[str] = None     # its security model: "2pc" | "2pc-dealer" | ...
    alternatives: Dict[str, Dict] = field(default_factory=dict)   # other backends: cost or exclusion

    # ------------------------------------------------------------ construction
    @classmethod
    def from_compiled(cls, compiled: CompiledPlan, graph, profile, deployment) -> "Plan":
        cm = StateExpandedCostModel(profile)
        net = deployment.network.to_setting()
        node_map = {n.node_id: n for n in graph.nodes}
        levels: Dict[str, tuple] = {}
        actions: List[Action] = []
        pending: List[Action] = []
        comm_bytes = 0
        comm_rounds = 0

        for s in compiled.steps:
            if s.action in ("he_exec", "mpc_exec"):
                nid = (s.detail or "").split(":", 1)[0]
                n = node_map.get(nid)
                if n is None:
                    continue
                dom = "HE" if s.action == "he_exec" else "MPC"
                est = cm.estimate_operator(n.op_type, dom, n.input_shape, n.output_shape, net)
                comm_bytes += est.comm_bytes
                comm_rounds += est.comm_rounds
                if dom == "HE" and s.src is not None and s.dst is not None:
                    levels[nid] = (s.src.l, s.dst.l)
                for a in pending:
                    a.before = nid
                actions.extend(pending)
                pending = []
            elif s.action in ("he2mpc", "mpc2he"):
                detail = s.detail or ""
                shape = None
                m = re.search(r"shape=(\([^)]*\))", detail)
                if m:
                    try:
                        shape = list(ast.literal_eval(m.group(1)))
                    except (ValueError, SyntaxError):
                        shape = None
                src, dst = ("HE", "MPC") if s.action == "he2mpc" else ("MPC", "HE")
                if shape is not None:
                    est = cm.estimate_conversion(src, dst, tuple(shape), net)
                    comm_bytes += est.comm_bytes
                    comm_rounds += est.comm_rounds
                pending.append(Action("convert", "<output>", src, dst, shape, cost_ms=s.cost_ms,
                                      edge=_edge(detail)))
            elif s.action == "bootstrap":
                lvl = s.src.l if s.src is not None else None
                pending.append(Action("bootstrap", "<output>", level_in=lvl, cost_ms=s.cost_ms,
                                      edge=_edge(s.detail or "")))
        actions.extend(pending)

        unplanned = [n.node_id for n in graph.nodes if n.node_id not in compiled.node_assignment]
        if unplanned:
            raise RuntimeError(f"planner left {len(unplanned)} operator(s) unassigned: {unplanned[:8]}")
        nodes = []
        for n in graph.nodes:
            dom = compiled.node_assignment[n.node_id]
            lin, lout = levels.get(n.node_id, (None, None))
            nodes.append(NodePlacement(n.node_id, n.op_type, dom, lin, lout,
                                       source=(n.attributes or {}).get("source")))

        hw = deployment.hardware
        return cls(
            graph_id=graph.graph_id,
            deployment={"hardware": hw.name or hw.device, "device": hw.device,
                        "bandwidth_bps": deployment.network.bandwidth_bps,
                        "rtt_ms": deployment.network.rtt_ms},
            profile={"platform": profile.platform, "he_level_budget": profile.he_level_budget,
                     "he_to_mpc_min_level": cm.he_to_mpc_min_level,
                     "bootstrap_min_level": cm.bootstrap_min_level,
                     "scaled_bootstrap": cm.scaled_bootstrap,
                     "metadata": {k: v for k, v in (profile.metadata or {}).items()
                                  if isinstance(v, (str, int, float))}},
            strategy=compiled.strategy_used,
            nodes=nodes,
            actions=actions,
            cost={"total_ms": compiled.total_cost_ms, "node_ms": compiled.node_cost_ms,
                  "conversion_ms": compiled.conversion_cost_ms,
                  "bootstrap_ms": compiled.bootstrap_cost_ms,
                  "comm_bytes": comm_bytes, "comm_rounds": comm_rounds},
            edges=[[e.src, e.dst] for e in graph.edges],
        )

    # ------------------------------------------------------------ queries
    def domains(self) -> Dict[str, str]:
        return {n.id: n.domain for n in self.nodes}

    def operators(self) -> List[NodePlacement]:
        """Placed operators, without the structural Input node."""
        return [n for n in self.nodes if n.op_type != "Input"]

    def input_domain(self) -> Optional[str]:
        """How the client provides the input: HE (encrypted) or MPC (secret-shared)."""
        ins = [n for n in self.nodes if n.op_type == "Input"]
        return ins[0].domain if ins else (self.nodes[0].domain if self.nodes else None)

    def counts(self) -> Dict[str, int]:
        ops = self.operators()
        return {
            "HE": sum(n.domain == "HE" for n in ops),
            "MPC": sum(n.domain == "MPC" for n in ops),
            "conversions": sum(a.kind == "convert" for a in self.actions),
            "bootstraps": sum(a.kind == "bootstrap" for a in self.actions),
        }

    def steps(self) -> List[Dict]:
        """The plan in execution order: every operator with its domain, CKKS levels, kernel and
        place in the PyTorch model, each preceded by the conversions and bootstraps that run before
        it (an action's `edge` names the tensor it applies to); actions on the output come last."""
        before: Dict[str, List[Action]] = {}
        for a in self.actions:
            before.setdefault(a.before, []).append(a)

        def action(a: Action) -> Dict:
            d = {"step": a.kind, "edge": a.edge, "cost_ms": a.cost_ms}
            if a.kind == "convert":
                d.update(src=a.src, dst=a.dst, tensor_shape=a.tensor_shape)
            else:
                d.update(level_in=a.level_in)
            return d

        out: List[Dict] = []
        for n in self.nodes:
            out += [action(a) for a in before.get(n.id, [])]
            out.append({"step": "operator", "id": n.id, "op_type": n.op_type, "domain": n.domain,
                        "level_in": n.level_in, "level_out": n.level_out, "kernel": n.kernel,
                        "backend": n.backend, "source": n.source})
        out += [action(a) for a in before.get("<output>", [])]
        return out

    def explain_steps(self) -> str:
        """steps() as a table: one line per operator, conversions and bootstraps indented above it."""
        steps = self.steps()
        num = {s["id"]: i for i, s in enumerate(t for t in steps if t["step"] == "operator")}

        def op_name(s):
            kern, own = s["kernel"], s.get("backend")
            tag = kern["kernel"] if kern else (own if own not in (None, "client", self.he_backend, self.mpc_backend)
                                              else None)           # an extension's primitive
            return s["op_type"] + (f" [{tag}]" if tag else "")
        w = max([22] + [len(op_name(s)) for s in steps if s["step"] == "operator"])
        lines = [f"  {'#':>4}  {'operator':{w}s} {'domain':6s} {'CKKS level':10s} "
                 + ("PyTorch" if any(n.source for n in self.nodes) else "node")]
        k = 0
        for s in steps:
            if s.get("edge"):
                p, c = s["edge"]
                edge = f"(#{num.get(p, p)} -> " + ("all its consumers)" if c == "*" else f"#{num.get(c, c)})")
            else:
                edge = ""
            if s["step"] == "convert":
                lines.append(f"  {'':4s}    convert {s['src']} -> {s['dst']}  {s['tensor_shape']}  "
                             f"{edge}  {s['cost_ms'] / 1e3:.2f} s")
            elif s["step"] == "bootstrap":
                lines.append(f"  {'':4s}    bootstrap at level {s['level_in']}  {edge}  {s['cost_ms'] / 1e3:.2f} s")
            else:
                lv = f"{s['level_in']} -> {s['level_out']}" if s["level_in"] is not None else ""
                where = s["source"] or s["id"]
                if s["op_type"] == "Input":
                    where += f" ({'encrypted' if s['domain'] == 'HE' else 'secret-shared'} by the client)"
                lines.append(f"  {k:4d}  {op_name(s):{w}s} {s['domain']:6s} {lv:10s} {where}")
                k += 1
        return "\n".join(lines)

    def explain(self, steps: bool = False) -> str:
        """Summary of the plan; steps=True adds the operator-by-operator table (explain_steps)."""
        c, k = self.cost, self.counts()
        d = self.deployment
        lines = [
            f"CipherFlow plan  graph={self.graph_id}  hardware={d['hardware']}  "
            f"BW={d['bandwidth_bps'] / 1e6:g} Mbps  RTT={d['rtt_ms']:g} ms  strategy={self.strategy}",
            f"  latency {c['total_ms'] / 1e3:.1f} s  (operators {c['node_ms'] / 1e3:.1f} s, "
            f"conversions {c['conversion_ms'] / 1e3:.1f} s, bootstraps {c['bootstrap_ms'] / 1e3:.1f} s)",
            f"  placement: {k['HE']} HE / {k['MPC']} MPC operators, "
            f"{k['conversions']} conversions, {k['bootstraps']} bootstraps, "
            f"{c['comm_bytes'] / 2**30:.2f} GiB online communication",
        ]
        m = d.get("network_measured")
        if m:
            lines.append(f"  network measured to {m['peer']}: RTT {m['rtt_ms']:.3g} ms, client->server "
                         f"{m['up_bps'] / 1e6:.1f} Mbps, server->client {m['down_bps'] / 1e6:.1f} Mbps "
                         "(planned with the slower direction)")
        if self.he_backend:
            from .backends import HE_BACKENDS
            hb = HE_BACKENDS.get(self.he_backend)
            prec = f"{self.precision_bits:g}-bit bootstrapping" if self.precision_bits else "precision not established"
            lines.append(f"  HE backend: {self.he_backend}" + (f" ({hb.label}; {prec}; costs {hb.status})" if hb else ""))
        if self.mpc_backend:
            from .backends import MPC_BACKENDS, SECURITY_TEXT
            b = MPC_BACKENDS.get(self.mpc_backend)
            lines.append(f"  MPC backend: {self.mpc_backend}"
                         + (f" ({b.label}; {SECURITY_TEXT[b.security]})" if b else f" (security {self.security})"))
            for name, alt in self.alternatives.items():
                if "total_ms" in alt:
                    lines.append(f"    vs {name}: {alt['total_ms'] / 1e3:.1f} s")
                else:
                    lines.append(f"    {name} not used: {alt['excluded']}")
        if any(n.op_type == "Input" for n in self.nodes):
            lines.append(f"  input: {self.input_domain()} "
                         f"({'encrypted' if self.input_domain() == 'HE' else 'secret-shared'} by the client)")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        by_type: Dict[str, Dict[str, int]] = {}
        for n in self.operators():
            by_type.setdefault(n.op_type, {"HE": 0, "MPC": 0})[n.domain] += 1
        for op, cnt in by_type.items():
            lines.append(f"    {op:22s} HE {cnt['HE']:3d}  MPC {cnt['MPC']:3d}")
        if steps:
            lines += ["  operator by operator (execution order; conversions and bootstraps run before the "
                      "operator below them):", self.explain_steps()]
        return "\n".join(lines)

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, src: str) -> "Plan":
        text = src if src.lstrip().startswith("{") else Path(src).read_text()
        d = json.loads(text)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unsupported plan schema {d.get('schema')!r}")
        d["nodes"] = [NodePlacement(**n) for n in d["nodes"]]
        d["actions"] = [Action(**a) for a in d["actions"]]
        return cls(**d)

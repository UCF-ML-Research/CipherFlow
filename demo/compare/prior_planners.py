"""CipherFlow against CostSphere and EncFormer, re-implemented from their papers.

Neither paper released code, so both placement methods are re-implemented at CipherFlow's operator
granularity, with the same measured costs (the paper's profiles, `paper-cpu` and `paper-gpu`), and
every plan is scored by one evaluator. A difference in latency therefore comes from the planning
method, not from kernels or hardware. docs/comparison.md describes the method and the results.

    python -m demo.compare.prior_planners              # the paper's grid and both papers' networks (about 25 min)
    python -m demo.compare.prior_planners --quick      # BERT-base, four networks (about 2 min)
    python -m demo.compare.prior_planners --site       # the website's chart data (demo/web/compare_data.json)

The evaluator charges a conversion on every edge whose two ends run in different domains, at that
edge's shape, and a bootstrap whenever an HE operator's inherited CKKS level is too low.

CostSphere (IACR ePrint 2025/1315, Sec. IV-C):
  * the operators form one chain in topological order; a conversion is charged only between
    neighbours in the chain (Eq. 4, constraints (7)-(12)), at the measured cost of each direction;
  * HE state is the count of consecutive HE operators: a bootstrap when the count exceeds n_threshold,
    after which the count restarts (Eq. 4). The paper does not state n_threshold: we use
    floor(L / mean level use of the model's operators), and also the best n for each setting;
  * Eq. 13 as a sliding window, b_l >= sum_{j=l-n+1..l} x_j - (n-1), as an ablation (needs SciPy);
  * Eq. 14's depth penalty is off (a = 1, b = 0).
  Eq. 4 is solved exactly by dynamic programming over (domain, run length). CostSphere also splits
  Softmax, LayerNorm and GeLU into sub-operations and mixes domains inside them; CipherFlow's profiles
  price whole operators, so that is not reproduced.

EncFormer (IEEE TDSC 2026, arXiv 2604.09975, Sec. V-A):
  * a fixed split: QK^T, P.V, output projection, FFN1 and FFN2 in HE; Softmax, LayerNorm and GELU in
    MPC; the paper does not place the residual adds, so we take the cheaper of HE and MPC;
  * no bootstrapping: ciphertexts are refreshed by the conversions;
  * with --encformer-conversions, every planner gets EncFormer's conversion, 3.26-3.33x cheaper
    (complex packing and modulus trimming, Table VIII): bytes and local time x 0.30.
  Not modeled: GELU pre-evaluation in CKKS, the distilled Softmax/LayerNorm surrogates (they change
  the model), and stage-compatible packing (HE kernels).

CipherFlow's planner prices a conversion at its consumer's input shape. For QK^T that is the packed
Q/K/V [3,B,S,H], three times the hidden state [B,S,H] that the edge carries and the evaluator prices.
--edge-shapes also plans CipherFlow with the edge's shape.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from cipherflow import profiles
from compiler.state_expanded.bert_graph import BertShapeManifest, bert_multi_block_graph, full_model_manifest
from compiler.state_expanded.cost_model import NetworkSetting, StateExpandedCostModel
from compiler.state_expanded.planner import DEFAULT_LEVEL_DELTAS, POLICY_LINEAR_HE_NONLINEAR_MPC, compile_plan_safe

MODELS = {
    "bert": lambda: bert_multi_block_graph(12, full_model_manifest()),
    "vit": lambda: bert_multi_block_graph(12, BertShapeManifest(batch=1, seq=197)),
}
PLATFORMS = {"cpu": "paper-cpu", "gpu": "paper-gpu"}
GRID = [(f"{label}/{rtt:g}ms", bw, rtt) for label, bw in (("10Mbps", 10e6), ("100Mbps", 100e6), ("1Gbps", 1e9),
                                                         ("3Gbps", 3e9)) for rtt in (1.0, 20.0, 40.0, 80.0)]
PAPER_NETWORKS = [("EncFormer LAN", 1e9, 0.3), ("EncFormer WAN1", 400e6, 4.0), ("EncFormer WAN2", 100e6, 4.0),
                  ("EncFormer WAN3", 100e6, 80.0), ("CostSphere LAN", 2e9, 0.0), ("CostSphere WAN", 100e6, 80.0)]
ENCFORMER_CONVERSION = 0.30
# the website's chart: the bandwidths of its results chart (demo/web/demo_data.json) at RTT 40 ms
SITE_BANDWIDTHS = [("10Mbps", 10e6), ("30Mbps", 30e6), ("100Mbps", 100e6), ("300Mbps", 300e6), ("1Gbps", 1e9),
                   ("3Gbps", 3e9), ("10Gbps", 10e9)]
SITE_RTT = 40.0
SITE_DATA = Path(__file__).resolve().parents[1] / "web" / "compare_data.json"


# ----------------------------------------------------------------- the common evaluator
def topological_order(graph) -> List[str]:
    indeg = {n.node_id: 0 for n in graph.nodes}
    out: Dict[str, List[str]] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        indeg[e.dst] += 1
        out[e.src].append(e.dst)
    ready = sorted(k for k, d in indeg.items() if d == 0)
    order = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return order


def score(graph, cm, net, assign: Dict[str, str], L: int) -> Tuple[float, int, int]:
    """(latency in ms, conversions, bootstraps) of a per-operator assignment."""
    node = {n.node_id: n for n in graph.nodes}
    in_edges: Dict[str, list] = {n.node_id: [] for n in graph.nodes}
    for e in graph.edges:
        in_edges[e.dst].append(e)
    level: Dict[str, int] = {}
    total, n_conv, n_bs = 0.0, 0, 0
    for nid in topological_order(graph):
        n, d = node[nid], assign[nid]
        pred = []
        for e in in_edges[nid]:
            if assign[e.src] != d:
                total += cm.estimate_conversion(assign[e.src], d, e.tensor_shape, net).latency_ms
                n_conv += 1
                pred.append(L)
            else:
                pred.append(level[e.src])
        delta = DEFAULT_LEVEL_DELTAS.get(n.op_type, 0)
        if d == "HE":
            cur = min(pred) if pred else L
            if cur < delta:
                bs = cm.estimate_bootstrap(net)
                if bs.resolution == "infeasible":
                    return math.inf, n_conv, n_bs
                total += bs.latency_ms
                n_bs += 1
                cur = L
            level[nid] = cur - delta
        else:
            level[nid] = L
        est = cm.estimate_operator(n.op_type, d, n.input_shape, n.output_shape, net)
        if est.resolution == "infeasible":
            return math.inf, n_conv, n_bs
        total += est.latency_ms
    return total, n_conv, n_bs


def scaled_conversions(prof, factor: float):
    return replace(prof, conversions=[replace(c, local_compute_ms=c.local_compute_ms * factor,
                                              comm_bytes=int(c.comm_bytes * factor)) for c in prof.conversions])


# ----------------------------------------------------------------- CipherFlow
@contextmanager
def edge_shape_pricing():
    """Plan with each conversion priced at the edge's shape (the hidden state into QK^T, not packed Q/K/V)."""
    orig = StateExpandedCostModel.estimate_conversion

    def at_edge_shape(self, a, b, shape, net):
        shape = tuple(shape)
        if len(shape) == 4 and shape[0] == 3:
            shape = shape[1:]
        return orig(self, a, b, shape, net)

    StateExpandedCostModel.estimate_conversion = at_edge_shape
    try:
        yield
    finally:
        StateExpandedCostModel.estimate_conversion = orig


def cipherflow(graph, prof, cm, net, L, edge_shapes: bool = False) -> Tuple[float, int, int]:
    if edge_shapes:
        with edge_shape_pricing():
            plan = compile_plan_safe(graph, prof, net)
    else:
        plan = compile_plan_safe(graph, prof, net)
    ms, nc, nb = score(graph, cm, net, plan.node_assignment, L)
    return min(ms, plan.total_cost_ms), nc, nb


# ----------------------------------------------------------------- CostSphere
class Chain:
    """The model as CostSphere sees it: operators in topological order, each next to the following one."""

    def __init__(self, graph, cm, net):
        self.order = topological_order(graph)
        node = {n.node_id: n for n in graph.nodes}
        edge = {(e.src, e.dst): e for e in graph.edges}
        self.op: List[Dict[str, float]] = []
        self.conv: List[Dict[Tuple[str, str], float]] = [{}]
        for i, nid in enumerate(self.order):
            n = node[nid]
            costs = {}
            for d in ("HE", "MPC"):
                est = cm.estimate_operator(n.op_type, d, n.input_shape, n.output_shape, net)
                costs[d] = math.inf if est.resolution == "infeasible" else est.latency_ms
            self.op.append(costs)
            if i:
                prev = self.order[i - 1]
                shape = edge[(prev, nid)].tensor_shape if (prev, nid) in edge else n.input_shape
                self.conv.append({(a, b): cm.estimate_conversion(a, b, shape, net).latency_ms
                                  for a, b in (("HE", "MPC"), ("MPC", "HE"))})
        bs = cm.estimate_bootstrap(net)
        self.boot = math.inf if bs.resolution == "infeasible" else bs.latency_ms


def costsphere(chain: Chain, n: int) -> Tuple[Dict[str, str], float]:
    """Eq. 4, solved exactly: states (domain, consecutive-HE count); a bootstrap when the count exceeds n."""
    layers: List[Dict[Tuple[str, int], Tuple[float, Tuple[str, int]]]] = [{}]
    for d in ("HE", "MPC"):
        c = chain.op[0][d]
        if c < math.inf:
            cnt, extra = (1, 0.0) if d == "HE" else (0, 0.0)
            if cnt > n:
                cnt, extra = 0, chain.boot
            layers[0][(d, cnt)] = (c + extra, None)
    for i in range(1, len(chain.order)):
        nxt: Dict[Tuple[str, int], Tuple[float, Tuple[str, int]]] = {}
        for (pd, pc), (pcost, _) in layers[-1].items():
            for d in ("HE", "MPC"):
                c = chain.op[i][d]
                if c == math.inf:
                    continue
                total = pcost + c + (chain.conv[i][(pd, d)] if pd != d else 0.0)
                cnt = ((pc if pd == "HE" else 0) + 1) if d == "HE" else 0
                if cnt > n:
                    total += chain.boot
                    cnt = 0
                if (d, cnt) not in nxt or total < nxt[(d, cnt)][0]:
                    nxt[(d, cnt)] = (total, (pd, pc))
        layers.append(nxt)
    key = min(layers[-1], key=lambda k: layers[-1][k][0])
    cost = layers[-1][key][0]
    assign = {}
    for i in range(len(chain.order) - 1, -1, -1):
        assign[chain.order[i]] = key[0]
        key = layers[i][key][1]
    return assign, cost


def costsphere_window(chain: Chain, n: int) -> Tuple[Dict[str, str], float]:
    """Eq. 13 as a sliding window, solved as a MILP (SciPy / HiGHS)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    k = len(chain.order)
    X, HM, MH, B = 0, k, 2 * k - 1, 3 * k - 2              # x (1 = HE), HE->MPC, MPC->HE, bootstrap
    nv = 4 * k - 2
    c, lo, hi, const = np.zeros(nv), np.zeros(nv), np.ones(nv), 0.0
    for i in range(k):
        he, mpc = chain.op[i]["HE"], chain.op[i]["MPC"]
        if he == math.inf:
            hi[X + i], he = 0, 0.0
        if mpc == math.inf:
            lo[X + i], mpc = 1, 0.0
        c[X + i], const, c[B + i] = he - mpc, const + mpc, chain.boot
    for i in range(1, k):
        c[HM + i - 1], c[MH + i - 1] = chain.conv[i][("HE", "MPC")], chain.conv[i][("MPC", "HE")]
    rows, lbs = [], []
    for i in range(1, k):
        r = np.zeros(nv); r[[HM + i - 1, X + i]] = 1; r[X + i - 1] = -1; rows.append(r); lbs.append(0)
        r = np.zeros(nv); r[[MH + i - 1, X + i - 1]] = 1; r[X + i] = -1; rows.append(r); lbs.append(0)
    for i in range(n - 1, k):
        r = np.zeros(nv); r[B + i] = 1; r[X + i - n + 1:X + i + 1] = -1; rows.append(r); lbs.append(-(n - 1))
    res = milp(c, integrality=np.ones(nv), bounds=Bounds(lo, hi),
               constraints=LinearConstraint(np.array(rows), np.array(lbs), np.full(len(rows), np.inf)))
    if not res.success:
        raise RuntimeError(res.message)
    x = np.round(res.x[X:X + k]).astype(int)
    return {nid: ("HE" if x[i] else "MPC") for i, nid in enumerate(chain.order)}, float(res.fun + const)


def threshold_from_levels(graph, L: int) -> int:
    return max(1, int(L / statistics.mean(DEFAULT_LEVEL_DELTAS.get(n.op_type, 0) for n in graph.nodes)))


# ----------------------------------------------------------------- EncFormer
def encformer(graph, cm, net, L) -> Tuple[float, int, int, str]:
    best = None
    for add in ("HE", "MPC"):
        assign = {n.node_id: add if n.op_type == "Residual_Add" else POLICY_LINEAR_HE_NONLINEAR_MPC.get(n.op_type, "HE")
                  for n in graph.nodes}
        ms, nc, nb = score(graph, cm, net, assign, L)
        if best is None or ms < best[0]:
            best = (ms, nc, nb, add)
    return best


# ----------------------------------------------------------------- the comparison
def compare(models, platforms, networks, encformer_conversions=False, edge_shapes=True, window=True) -> List[dict]:
    rows = []
    for model in models:
        graph = MODELS[model]()
        for plat in platforms:
            prof = profiles.load(PLATFORMS[plat])
            if encformer_conversions:
                prof = scaled_conversions(prof, ENCFORMER_CONVERSION)
            cm, L = StateExpandedCostModel(prof), prof.he_level_budget
            n0 = threshold_from_levels(graph, L)
            for label, bw, rtt in networks:
                net = NetworkSetting(bw, rtt)
                row = {"model": model, "platform": plat, "network": label, "bandwidth_bps": bw, "rtt_ms": rtt,
                       "conversions": "EncFormer's" if encformer_conversions else "measured"}
                row["cipherflow_ms"], row["cipherflow_conversions"], row["cipherflow_bootstraps"] = \
                    cipherflow(graph, prof, cm, net, L)
                if edge_shapes:
                    row["cipherflow_edge_shapes_ms"] = cipherflow(graph, prof, cm, net, L, edge_shapes=True)[0]
                chain = Chain(graph, cm, net)
                a, own = costsphere(chain, n0)
                row["costsphere_ms"], row["costsphere_conversions"], row["costsphere_bootstraps"] = score(graph, cm, net, a, L)
                row["costsphere_own_estimate_ms"], row["costsphere_n"] = own, n0
                row["costsphere_best_ms"], row["costsphere_best_n"] = min(
                    (score(graph, cm, net, costsphere(chain, n)[0], L)[0], n) for n in range(1, L + 1))
                if window:
                    row["costsphere_eq13_ms"] = score(graph, cm, net, costsphere_window(chain, n0)[0], L)[0]
                (row["encformer_ms"], row["encformer_conversions"], row["encformer_bootstraps"],
                 row["encformer_residual_add"]) = encformer(graph, cm, net, L)
                rows.append(row)
            print(f"  {model} / {plat} done", flush=True)
    return rows


def summary(rows: List[dict]) -> str:
    def span(values):
        return f"{min(values):.2f}–{max(values):.2f}×"

    keys = [("costsphere_ms", "CostSphere"), ("costsphere_best_ms", "CostSphere, best n"),
            ("costsphere_eq13_ms", "CostSphere, Eq. 13"), ("encformer_ms", "EncFormer")]
    keys = [(k, t) for k, t in keys if k in rows[0]]
    out = ["Latency relative to CipherFlow (above 1×: slower than CipherFlow); the same costs and evaluator for all.", ""]
    for ref, name in (("cipherflow_ms", "CipherFlow"), ("cipherflow_edge_shapes_ms", "CipherFlow, edge-shape pricing")):
        if ref not in rows[0]:
            continue
        out += [f"Against {name}:", "", "| model | platform | " + " | ".join(t for _, t in keys) + " |",
                "|---|---|" + "---|" * len(keys)]
        for model in dict.fromkeys(r["model"] for r in rows):
            for plat in dict.fromkeys(r["platform"] for r in rows):
                sub = [r for r in rows if r["model"] == model and r["platform"] == plat]
                if sub:
                    out.append(f"| {model} | {plat} | " + " | ".join(span([r[k] / r[ref] for r in sub]) for k, _ in keys) + " |")
        out.append("")
    return "\n".join(out)


def site_data(path: Path = SITE_DATA) -> dict:
    """The website's comparison: BERT-base on both platforms at RTT 40 ms, at its results chart's bandwidths."""
    rows = compare(["bert"], list(PLATFORMS), [(f"{b}/{SITE_RTT:g}ms", bw, SITE_RTT) for b, bw in SITE_BANDWIDTHS],
                   window=False)
    keys = {"cf": "cipherflow_ms", "cfEdge": "cipherflow_edge_shapes_ms", "cs": "costsphere_ms",
            "csBest": "costsphere_best_ms", "enc": "encformer_ms"}
    data = {"model": "bert", "rtt_ms": SITE_RTT, "bandwidths": [b for b, _ in SITE_BANDWIDTHS],
            "plans": {plat: [{k: round(r[v], 1) for k, v in keys.items()} for r in rows if r["platform"] == plat]
                      for plat in PLATFORMS}}
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(summary(rows))
    print(f"the website's comparison: {path}")
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="BERT-base, four networks")
    ap.add_argument("--encformer-conversions", action="store_true",
                    help="give every planner EncFormer's conversion (bytes and local time x 0.30)")
    ap.add_argument("--out", default="prior_planners.csv", help="CSV of every plan's latency")
    ap.add_argument("--site", action="store_true",
                    help="write the website's chart data (demo/web/compare_data.json): BERT-base, RTT 40 ms")
    a = ap.parse_args(argv)
    if a.site:
        site_data()
        return 0
    if a.quick:
        models, networks = ["bert"], [GRID[1], GRID[6], GRID[9], PAPER_NETWORKS[1]]
    else:
        models, networks = list(MODELS), GRID + PAPER_NETWORKS
    try:
        import scipy  # noqa: F401
        window = True
    except ImportError:
        print("SciPy is not installed: the Eq. 13 ablation is skipped")
        window = False
    rows = compare(models, list(PLATFORMS), networks, a.encformer_conversions, window=window)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(summary(rows))
    print(f"every plan: {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

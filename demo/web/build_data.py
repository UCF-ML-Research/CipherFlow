"""Precompute CipherFlow plans for the interactive demo.

The demo's input is PyTorch code: the encoders in demo/transformer.py, traced with
`cf.models.from_torch` (Q/K/V projections left out, as the paper's profiles have no cost for them).
Its output is the plan of every operator: HE or MPC, the CKKS level, and the conversions and
bootstraps before it, each operator labelled with where it is in the PyTorch code.

Every plan in the demo comes from a real `cipherflow.compile` call; the page only
looks plans up on this grid (no interpolation). Hardware options:

* cpu      -- measured CPU profile (AMD Threadripper PRO 3955WX)
* gpu      -- GPU profile used in the paper

Both are the paper's profiles: MPC costs are SHAFT (CrypTen), whose Beaver triples
come from a trusted dealer, so plans are compiled with security="2pc-dealer".
* gpu_x0.5, gpu_x2, gpu_x4 -- projections of the GPU profile with local compute
  scaled 0.5x / 2x / 4x (communication bytes and rounds unchanged)

Usage (from the repository root):
    python -m demo.web.build_data            # about 7 min on 6 workers: 350 plans
    python -m demo.web.build_page            # -> docs/index.html (website), docs/planner/index.html
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cipherflow as cf
from cipherflow.backends import PSEUDO_OPS
from cipherflow.profiles import scale_compute
from compiler.state_expanded.cost_model import StateExpandedCostModel
from compiler.state_expanded.planner import POLICY_LINEAR_HE_NONLINEAR_MPC, evaluate_uniform_domain
from compiler.state_expanded.sese import find_sese_regions

MODELS = {
    "bert": {"label": "BERT-base", "detail": "12 post-LN encoder blocks, sequence length 128",
             "factory": "bert", "encoder": "BertEncoder", "block": "PostLNBlock", "shape": (1, 128, 768)},
    "vit": {"label": "ViT-B/16", "detail": "patch projection, 12 pre-LN encoder blocks, final LayerNorm; 197 tokens "
                       "(14x14 patches + CLS); costs scaled from the 128-token profile entries",
            "factory": "vit", "encoder": "ViTEncoder", "block": "PreLNBlock", "shape": (1, 197, 768)},
}
GRAPHS = {}          # model -> traced OperatorGraph (built before the worker processes fork)


def _trace(model):
    import torch
    from demo import transformer
    spec = MODELS[model]
    return cf.models.from_torch(getattr(transformer, spec["factory"])(12), torch.zeros(spec["shape"]),
                                qkv_projection=False)


def _code(model):
    """The PyTorch code the demo shows as its input."""
    from demo import transformer
    spec = MODELS[model]
    parts = [inspect.getsource(getattr(transformer, c)).rstrip()
             for c in ("SelfAttention", "MLP", spec["block"], spec["encoder"])]
    shape = ", ".join(map(str, spec["shape"]))
    return ("import math\nimport torch\nfrom torch import nn\nimport cipherflow as cf\n\n\n" + "\n\n\n".join(parts)
            + f"\n\n\nmodel = {spec['encoder']}(num_blocks=12)\n"
            f"graph = cf.models.from_torch(model, torch.randn({shape}), qkv_projection=False)\n")


HARDWARE = {
    "cpu": {"label": "CPU", "detail": "paper CPU profile (Threadripper PRO 3955WX); HE: NEXUS-CKKS, MPC: SHAFT (two parties + trusted dealer)", "base": "paper-cpu", "scale": 1.0},
    "gpu_x0.5": {"label": "GPU 0.5x", "detail": "paper GPU profile (NEXUS-CUDA + SHAFT, trusted dealer), compute scaled 0.5x (projection)", "base": "paper-gpu", "scale": 0.5},
    "gpu": {"label": "GPU", "detail": "GPU profile used in the paper; HE: NEXUS-CUDA, MPC: SHAFT (two parties + trusted dealer)", "base": "paper-gpu", "scale": 1.0},
    "gpu_x2": {"label": "GPU 2x", "detail": "paper GPU profile (NEXUS-CUDA + SHAFT, trusted dealer), compute scaled 2x (projection)", "base": "paper-gpu", "scale": 2.0},
    "gpu_x4": {"label": "GPU 4x", "detail": "paper GPU profile (NEXUS-CUDA + SHAFT, trusted dealer), compute scaled 4x (projection)", "base": "paper-gpu", "scale": 4.0},
}
BANDWIDTHS = [("10Mbps", 10e6), ("30Mbps", 30e6), ("100Mbps", 100e6), ("300Mbps", 300e6),
              ("1Gbps", 1e9), ("3Gbps", 3e9), ("10Gbps", 10e9)]
RTTS = [1.0, 10.0, 20.0, 40.0, 80.0]


SHORT = {"Linear_QKV": "QKV", "Attention_QK_MatMul": "QK\u1d40", "Softmax": "Softmax", "Attention_V_MatMul": "Attn\u00b7V",
         "Out_Projection": "Out-Proj", "Residual_Add": "Add", "LayerNorm": "LayerNorm",
         "FFN_Linear_1": "FFN-1", "GeLU": "GeLU", "FFN_Linear_2": "FFN-2"}


def _layout(graph):
    """Column = encoder block (the input, the patch projection and the final LayerNorm get their own
    columns), row = position in the block; label = operator, src = where it is in the PyTorch code."""
    srcs = [n.attributes.get("source", n.node_id) for n in graph.nodes]
    pre = [i for i, s in enumerate(srcs) if not s.startswith("blocks.")
           and i < next(k for k, t in enumerate(srcs) if t.startswith("blocks."))]
    out, rows = [], {}
    for n, src in zip(graph.nodes, srcs):
        m = re.match(r"blocks\.(\d+)", src)
        if m:
            blk = int(m.group(1))
            col, group, label = blk + (1 if pre else 0), f"block {blk + 1}", SHORT.get(n.op_type, n.op_type)
        elif n.op_type == "Input":
            col, group, label = 0, "input", "Input"
        elif graph.nodes.index(n) in pre:
            col, group, label = 0, "embed", "Patch proj."
        else:
            col, group, label = None, "output", "Final LN"
        out.append({"col": col, "row": None, "label": label, "group": group, "src": src})
    last = 1 + max(o["col"] for o in out if o["col"] is not None)
    for o in out:
        if o["col"] is None:
            o["col"] = last
        o["row"] = rows.get(o["col"], 0)
        rows[o["col"]] = o["row"] + 1
    if any(o["group"] == "output" for o in out):             # final LN level with the blocks' last row
        o = next(o for o in out if o["group"] == "output")
        o["row"] = max(rows[c] for c in rows if c != o["col"]) - 1
    return out


def _profile(hw_key):
    h = HARDWARE[hw_key]
    p = cf.load_profile(h["base"])
    return p if h["scale"] == 1.0 else scale_compute(p, h["scale"], label=h["label"])


def _with_input(profile, graph):
    """The profile plus the planner's zero-cost records for the client's input (as compile adds them)."""
    from dataclasses import replace
    from cipherflow.planning import _record
    extra = [_record(n.op_type, d, n, feasible=True, method="client", note="model input provided by the client")
             for n in graph.nodes if n.op_type in PSEUDO_OPS for d in ("HE", "MPC")]
    return replace(profile, operators=list(profile.operators) + extra)


def _static(graph, profile, net, factor):
    cm = StateExpandedCostModel(profile)
    dom = {n.node_id: POLICY_LINEAR_HE_NONLINEAR_MPC.get(n.op_type, "HE") for n in graph.nodes}
    pseudo = {n.node_id for n in graph.nodes if n.op_type in PSEUDO_OPS}
    total = 0.0
    for n in graph.nodes:
        if n.op_type not in PSEUDO_OPS:
            total += cm.estimate_operator(n.op_type, dom[n.node_id], n.input_shape, n.output_shape, net).latency_ms
    for e in graph.edges:
        if e.src not in pseudo and dom[e.src] != dom[e.dst]:    # the client provides the input as each consumer needs it
            total += cm.estimate_conversion(dom[e.src], dom[e.dst], e.tensor_shape, net).latency_ms * factor
    return total


def _one(job):
    import warnings
    warnings.simplefilter("ignore", cf.CostExtrapolationWarning)   # stated in the model detail
    model, hw, bw_label, bw, rtt = job
    graph = GRAPHS[model]
    profile = _profile(hw)
    dep = cf.Deployment(cf.Hardware("cuda" if hw.startswith("gpu") else "cpu", HARDWARE[hw]["label"]),
                        cf.Network(bw, rtt))
    plan = cf.compile(graph, dep, profile=profile, security="2pc-dealer")
    net = dep.network.to_setting()
    idx = {n.node_id: i for i, n in enumerate(graph.nodes)}
    baselines = {
        "BumbleBee": _static(graph, profile, net, 1.0),
        "BOLT": _static(graph, profile, net, 0.75),
        "BLB": _static(graph, profile, net, 0.5),
        "All-HE": evaluate_uniform_domain(graph, _with_input(profile, graph), net, "HE"),
        "All-MPC": evaluate_uniform_domain(graph, _with_input(profile, graph), net, "MPC"),
    }
    return {
        "key": f"{model}|{hw}|{bw_label}|{rtt:g}",
        "dom": "".join("H" if n.domain == "HE" else "M" for n in plan.nodes),
        "lv": [n.level_out if n.level_out is not None else -1 for n in plan.nodes],
        # kind, the operator it runs before, src/dst domain, the edge's producer and consumer (-1: all
        # consumers), cost in ms
        "act": [[a.kind[0], idx.get(a.before, -1), a.src or "", a.dst or "",
                 idx.get(a.edge[0], -1) if a.edge else -1, idx.get(a.edge[1], -1) if a.edge else -1,
                 round(a.cost_ms)] for a in plan.actions],
        "cost": {k: round(v, 1) for k, v in plan.cost.items()},
        "strategy": plan.strategy,
        "base": {k: round(v, 1) for k, v in baselines.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=str(Path(__file__).resolve().parent / "demo_data.json"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    graphs = {}
    for m, spec in MODELS.items():
        g = GRAPHS[m] = _trace(m)
        graphs[m] = {
            "label": spec["label"], "detail": spec["detail"], "code": _code(m),
            "torch": f"demo/transformer.py:{spec['factory']}",
            "input_shape": ",".join(map(str, spec["shape"])),
            "residual_regions": len(find_sese_regions(g)),
            "nodes": [dict(id=n.node_id, op=n.op_type, **lay, **{"in": list(n.input_shape), "out": list(n.output_shape)})
                      for n, lay in zip(g.nodes, _layout(g))],
            "edges": [[e.src, e.dst, list(e.tensor_shape)] for e in g.edges],
        }
    jobs = [(m, hw, bl, bw, rtt) for m in MODELS for hw in HARDWARE for bl, bw in BANDWIDTHS for rtt in RTTS]
    t0 = time.time()
    plans = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, r in enumerate(ex.map(_one, jobs, chunksize=2), 1):
            plans[r.pop("key")] = r
            if k % 25 == 0:
                print(f"{k}/{len(jobs)} plans ({time.time() - t0:.0f}s)", flush=True)
    he_budget = cf.load_profile("paper-gpu").he_level_budget
    out = {
        "generated": time.strftime("%Y-%m-%d"),
        "cipherflow_version": cf.__version__,
        "he_level_budget": he_budget,
        "models": graphs,
        "hardware": {k: {"label": v["label"], "detail": v["detail"]} for k, v in HARDWARE.items()},
        "bandwidths": [b for b, _ in BANDWIDTHS],
        "rtts": RTTS,
        "plans": plans,
    }
    Path(a.output).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.output}: {len(plans)} plans in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

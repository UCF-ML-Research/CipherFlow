"""Precompute CipherFlow plans for the interactive demo.

Every plan in the demo comes from a real `cipherflow.compile` call; the page only
looks plans up on this grid (no interpolation). Hardware options:

* cpu      -- measured CPU profile (AMD Threadripper PRO 3955WX)
* gpu      -- GPU profile used in the paper

Both are the paper's profiles: MPC costs are SHAFT (CrypTen), whose Beaver triples
come from a trusted dealer, so plans are compiled with security="2pc-dealer".
* gpu_x0.5, gpu_x2, gpu_x4 -- projections of the GPU profile with local compute
  scaled 0.5x / 2x / 4x (communication bytes and rounds unchanged)

Usage (from operator_execution_framework/):
    python -m cipherflow.demo.build_data -o cipherflow/demo/demo_data.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cipherflow as cf
from cipherflow.profiles import scale_compute
from compiler.state_expanded.cost_model import StateExpandedCostModel
from compiler.state_expanded.planner import POLICY_LINEAR_HE_NONLINEAR_MPC, evaluate_uniform_domain
from compiler.state_expanded.sese import find_sese_regions

MODELS = {
    "bert": {"label": "BERT-base", "detail": "12 encoder blocks, sequence length 128",
             "build": lambda: cf.models.bert(num_blocks=12, seq=128, qkv_projection=False)},
    "vit": {"label": "ViT-B/16", "detail": "patch embedding, 12 pre-LN encoder blocks, final LayerNorm; 197 tokens (14x14 patches + CLS); "
                       "costs scaled from the 128-token profile entries",
            "build": lambda: cf.models.vit(num_blocks=12, tokens=197, qkv_projection=False)},
}
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


def _layout(node_id, op_type, graph):
    """Column = encoder block (stand-alone nodes get their own column), row = position in block."""
    import re
    m = re.fullmatch(r"L(\d+)_(\w+)", node_id)
    has_pre = any(n.node_id == "patch_embed" for n in graph.nodes)
    if m:
        blk = int(m.group(1))
        block_ids = [n.node_id for n in graph.nodes if n.node_id.startswith(f"L{blk}_")]
        return {"col": blk + (1 if has_pre else 0), "row": block_ids.index(node_id), "label": SHORT.get(op_type, op_type),
                "group": f"block {blk + 1}"}
    n_blocks = 1 + max(int(re.match(r"L(\d+)_", n.node_id).group(1)) for n in graph.nodes if re.match(r"L\d+_", n.node_id))
    if node_id == "patch_embed":
        return {"col": 0, "row": 0, "label": "Patch embed", "group": "embed"}
    last_row = sum(1 for n in graph.nodes if n.node_id.startswith("L0_")) - 1
    return {"col": n_blocks + (1 if has_pre else 0), "row": last_row, "label": "Final LN", "group": "output"}


def _profile(hw_key):
    h = HARDWARE[hw_key]
    p = cf.load_profile(h["base"])
    return p if h["scale"] == 1.0 else scale_compute(p, h["scale"], label=h["label"])


def _static(graph, profile, net, factor):
    cm = StateExpandedCostModel(profile)
    dom = {n.node_id: POLICY_LINEAR_HE_NONLINEAR_MPC.get(n.op_type, "HE") for n in graph.nodes}
    total = 0.0
    for n in graph.nodes:
        total += cm.estimate_operator(n.op_type, dom[n.node_id], n.input_shape, n.output_shape, net).latency_ms
    for e in graph.edges:
        if dom[e.src] != dom[e.dst]:
            total += cm.estimate_conversion(dom[e.src], dom[e.dst], e.tensor_shape, net).latency_ms * factor
    return total


def _one(job):
    import warnings
    warnings.simplefilter("ignore", cf.CostExtrapolationWarning)   # stated in the model detail
    model, hw, bw_label, bw, rtt = job
    graph = MODELS[model]["build"]()
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
        "All-HE": evaluate_uniform_domain(graph, profile, net, "HE"),
        "All-MPC": evaluate_uniform_domain(graph, profile, net, "MPC"),
    }
    return {
        "key": f"{model}|{hw}|{bw_label}|{rtt:g}",
        "dom": "".join("H" if n.domain == "HE" else "M" for n in plan.nodes),
        "lv": [n.level_out if n.level_out is not None else -1 for n in plan.nodes],
        "act": [[a.kind[0], idx.get(a.before, -1), a.src or "", a.dst or ""] for a in plan.actions],
        "cost": {k: round(v, 1) for k, v in plan.cost.items()},
        "strategy": plan.strategy,
        "base": {k: round(v, 1) for k, v in baselines.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="cipherflow/demo/demo_data.json")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    graphs = {}
    for m, spec in MODELS.items():
        g = spec["build"]()
        graphs[m] = {
            "label": spec["label"], "detail": spec["detail"],
            "residual_regions": len(find_sese_regions(g)),
            "nodes": [dict(id=n.node_id, op=n.op_type, **_layout(n.node_id, n.op_type, g),
                           **{"in": list(n.input_shape), "out": list(n.output_shape)}) for n in g.nodes],
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

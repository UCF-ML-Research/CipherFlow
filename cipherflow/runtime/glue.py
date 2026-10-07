"""GLUE accuracy of CipherFlow plans, by emulation (needs torch, transformers, datasets).

    python -m cipherflow.runtime.glue --tasks sst2,mrpc,rte,stsb --out glue.json
    python -m cipherflow.runtime.glue --tasks sst2 --download-only       # e.g. on a login node

For each task, with a BERT-base fine-tuned on it (textattack/bert-base-uncased-<TASK>):
  1. calibrate the encoder on --calibrate training examples;
  2. plan the 12-block encoder (128 tokens) for each --settings entry, with the calibration (HE only
     where the kernels' approximations are valid on the data) and, with --uncalibrated, without it;
  3. emulate every plan on the dev set (--limit examples) and score it like the plaintext model.
Embeddings, pooler and classifier run in plaintext; the encoder runs as the plan places it. Each
example runs at its own length without padding (CipherFlow has no attention masks), with the plan
made for 128 tokens (the operators are the same at every length).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from typing import Dict, List

TASKS: Dict[str, Dict] = {
    "sst2": {"model": "textattack/bert-base-uncased-SST-2", "fields": ("sentence", None), "metric": "acc"},
    "mrpc": {"model": "textattack/bert-base-uncased-MRPC", "fields": ("sentence1", "sentence2"), "metric": "acc_f1"},
    "rte": {"model": "textattack/bert-base-uncased-RTE", "fields": ("sentence1", "sentence2"), "metric": "acc"},
    "qnli": {"model": "textattack/bert-base-uncased-QNLI", "fields": ("question", "sentence"), "metric": "acc"},
    "stsb": {"model": "textattack/bert-base-uncased-STS-B", "fields": ("sentence1", "sentence2"), "metric": "corr"},
}
DEFAULT_SETTINGS = "cpu:10Mbps/20ms:2pc-dealer,cpu:100Mbps/20ms:2pc-dealer,cpu:1Gbps/40ms:2pc,cpu:300Mbps/40ms:2pc"


def _scores(metric: str, preds: List[float], labels: List[float]) -> Dict[str, float]:
    if metric == "corr":
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        p, y = np.array(preds), np.array(labels)
        return {"pearson": float(pearsonr(p, y)[0]), "spearman": float(spearmanr(p, y)[0])}
    acc = sum(int(p) == int(y) for p, y in zip(preds, labels)) / len(labels)
    out = {"accuracy": acc}
    if metric == "acc_f1":
        tp = sum(int(p) == 1 and int(y) == 1 for p, y in zip(preds, labels))
        fp = sum(int(p) == 1 and int(y) == 0 for p, y in zip(preds, labels))
        fn = sum(int(p) == 0 and int(y) == 1 for p, y in zip(preds, labels))
        out["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return out


def run_task(task: str, settings: List[tuple], n_cal: int, limit: int, uncalibrated: bool, verbose=True) -> Dict:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import cipherflow as cf
    from cipherflow import runtime

    spec = TASKS[task]
    tok = AutoTokenizer.from_pretrained(spec["model"])
    # eager attention: explicit Q/K/V matmuls and softmax (SDPA would be one opaque operator)
    model = AutoModelForSequenceClassification.from_pretrained(spec["model"], attn_implementation="eager").eval().double()
    enc = model.bert.encoder
    data = load_dataset("nyu-mll/glue", task)
    a, b = spec["fields"]
    regression = spec["metric"] == "corr"

    def hidden(ex):
        ids = tok(ex[a], ex[b] if b else None, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            return model.bert.embeddings(input_ids=ids["input_ids"], token_type_ids=ids["token_type_ids"])

    def predict(h):
        with torch.no_grad():
            z = model.classifier(model.bert.pooler(h))[0]
        return float(z[0]) if regression else int(z.argmax())

    t0 = time.time()
    train = data["train"].select(range(min(n_cal, len(data["train"]))))
    cal = runtime.calibrate(enc, (hidden(ex) for ex in train), model=spec["model"])
    dev = data["validation"]
    dev = dev.select(range(min(limit, len(dev)))) if limit else dev
    xs = [hidden(ex) for ex in dev]
    labels = [float(ex["label"]) for ex in dev]
    with torch.no_grad():
        plain = [predict(enc(x)[0]) for x in xs]
    result = {"task": task, "model": spec["model"], "calibration_examples": cal.samples, "dev_examples": len(xs),
              "plaintext": _scores(spec["metric"], plain, labels), "plans": []}
    if verbose:
        print(f"[{task}] plaintext {result['plaintext']} on {len(xs)} dev examples "
              f"(calibrated on {cal.samples}, {time.time() - t0:.0f} s)", flush=True)
    graph = cf.models.from_torch(enc, torch.randn(1, 128, 768, dtype=torch.float64))
    for hw, net, sec in settings:
        for use_cal in ((True, False) if uncalibrated else (True,)):
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
                plan = cf.compile(graph, cf.Deployment.of(hw, net), security=sec, verbose=False,
                                  calibration=cal if use_cal else None)
            em = runtime.Emulator(plan, enc)
            preds, rel, oor = [], [], set()
            for x in xs:
                r = em(x)
                preds.append(predict(r.output))
                rel.append(r.rel_err if math.isfinite(r.rel_err) else float("inf"))
                oor |= set(r.out_of_range())
            agree = (max(abs(p - q) for p, q in zip(preds, plain)) if regression
                     else sum(p == q for p, q in zip(preds, plain)) / len(xs))
            row = {"hardware": hw, "network": net, "security": sec, "calibrated": use_cal,
                   "backends": f"{plan.he_backend}+{plan.mpc_backend}", "latency_s": plan.cost["total_ms"] / 1e3,
                   "he_operators": sorted({n.op_type for n in plan.operators() if n.domain == "HE"}),
                   "scores": _scores(spec["metric"], preds, labels),
                   ("max_abs_diff_vs_plaintext" if regression else "agreement_with_plaintext"): agree,
                   "median_rel_err": sorted(rel)[len(rel) // 2], "out_of_range_nodes": sorted(oor),
                   "plan_problems": em._checks["problems"] if em._checks else []}
            result["plans"].append(row)
            if verbose:
                print(f"[{task}] {hw} {net} {sec} {'calibrated' if use_cal else 'uncalibrated':12s} "
                      f"{row['backends']} {row['latency_s']:.0f} s | HE {row['he_operators']} | {row['scores']} "
                      f"| vs plaintext {agree:.3g} | {len(oor)} out-of-range nodes ({time.time() - t:.0f} s)",
                      flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", default="sst2,mrpc,rte,stsb")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS, help="hardware:BW/RTT:security, comma separated")
    ap.add_argument("--calibrate", type=int, default=2048, help="training examples to calibrate on")
    ap.add_argument("--limit", type=int, default=0, help="dev examples per task (0: all)")
    ap.add_argument("--uncalibrated", action="store_true", help="also plan without the calibration")
    ap.add_argument("--download-only", action="store_true", help="fetch the models and datasets, then exit")
    ap.add_argument("--out", default="glue_report.json")
    a = ap.parse_args(argv)
    tasks = [t for t in a.tasks.split(",") if t]
    if a.download_only:
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        for t in tasks:
            AutoTokenizer.from_pretrained(TASKS[t]["model"])
            AutoModelForSequenceClassification.from_pretrained(TASKS[t]["model"])
            load_dataset("nyu-mll/glue", t)
            print(f"[{t}] downloaded", flush=True)
        return 0
    settings = [tuple(s.split(":", 2)) for s in a.settings.split(",") if s]
    from ..selfcheck import machine_info
    report = {"machine": machine_info(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "args": vars(a), "tasks": []}
    for t in tasks:
        report["tasks"].append(run_task(t, settings, a.calibrate, a.limit, a.uncalibrated))
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    print(f"report: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

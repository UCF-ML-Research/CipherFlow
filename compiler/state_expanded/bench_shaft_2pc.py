"""Two-party SHAFT (CrypTen) per-operator measurement.

Unlike bench_shaft_{cpu,gpu}.py, which run CrypTen in a single process
(world_size=1, so there is no second party and no communication), this runs
the two parties as separate processes connected over localhost (gloo/TCP),
optionally plus a trusted third party (TTP) that deals the Beaver triples.

Per operator and party it records:
  ms             wall time of the operator (median of REPEATS, after a warm-up)
  rounds         CrypTen's communication-call count for one run
  logical_bytes  CrypTen's byte estimate for one run (elements x 8)
and, machine-wide, the bytes that crossed the loopback interface during one run
(TCP payload plus headers, both parties and the dealer): the wire traffic.

Providers: TFP = triples generated locally from a shared seed (online phase only,
no dealer traffic); TTP = a dealer process sends the triples (adds the offline
traffic). Offline traffic = TTP wire bytes - TFP wire bytes.

    CIPHERFLOW_SHAFT_ROOT=... python -m compiler.state_expanded.bench_shaft_2pc --device cuda --seq 128
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SHAFT_ROOT = Path(os.environ.get("CIPHERFLOW_SHAFT_ROOT",
                                 Path(__file__).resolve().parents[2] / "baseline" / "SHAFT"))
sys.path.insert(0, str(SHAFT_ROOT))

import torch  # noqa: E402
import crypten  # noqa: E402
import crypten.communicator as comm  # noqa: E402
from crypten.mpc.context import run_multiprocess  # noqa: E402

REPEATS = 3
LO = Path("/sys/class/net/lo/statistics/tx_bytes")


def _lo_bytes() -> int:
    return int(LO.read_text())


def _ops(B, S, H, F, heads):
    d = H // heads
    tok, ffn, sc = (B, S, H), (B, S, F), (B, heads, S, S)
    return {
        # name: (input specs [(shape, owner rank, scale)], function, shape_in, shape_out)
        "Attention_QK_MatMul": ([((B, heads, S, d), 1, .01), ((B, heads, S, d), 1, .01)],
                                lambda q, k: q.matmul(k.transpose(-1, -2)), (3, B, S, H), sc),
        "Attention_V_MatMul": ([(sc, 1, .01), ((B, heads, S, d), 1, .01)],
                               lambda a, v: a.matmul(v), sc, tok),
        "Linear_QKV": ([(tok, 1, .01), ((H, 3 * H), 0, .01), ((3 * H,), 0, .01)],
                       lambda x, w, b: x.matmul(w) + b, tok, (3, B, S, H)),
        "Out_Projection": ([(tok, 1, .01), ((H, H), 0, .01), ((H,), 0, .01)],
                           lambda x, w, b: x.matmul(w) + b, tok, tok),
        "FFN_Linear_1": ([(tok, 1, .01), ((H, F), 0, .01), ((F,), 0, .01)],
                         lambda x, w, b: x.matmul(w) + b, tok, ffn),
        "FFN_Linear_2": ([(ffn, 1, .01), ((F, H), 0, .01), ((H,), 0, .01)],
                         lambda x, w, b: x.matmul(w) + b, ffn, tok),
        "Softmax": ([(sc, 1, .01)], lambda x: x.softmax(dim=-1), sc, sc),
        "LayerNorm": ([(tok, 1, .1)], _layernorm, tok, tok),
        "GeLU": ([(ffn, 1, .1)], lambda x: x.gelu(), ffn, ffn),
        "Residual_Add": ([(tok, 1, 1.), (tok, 1, 1.)], lambda a, b: a + b, tok, tok),
    }


def _spec(op, sin, sout):
    """(input specs, function) for one operator at explicit shapes (MicroBench jobs)."""
    sin, sout = tuple(sin), tuple(sout)
    if op == "Attention_QK_MatMul":                 # packed [3,B,S,H] -> [B,h,S,S]
        _, B, S, H = sin
        h = sout[1]
        return [((B, h, S, H // h), 1, .01)] * 2, lambda q, k: q.matmul(k.transpose(-1, -2))
    if op == "Attention_V_MatMul":                  # [B,h,S,S] -> [B,S,H]
        B, h, S, _ = sin
        H = sout[-1]
        return [(sin, 1, .01), ((B, h, S, H // h), 1, .01)], lambda a, v: a.matmul(v)
    if op in ("Out_Projection", "FFN_Linear_1", "FFN_Linear_2", "Linear_QKV"):
        hi = sin[-1]
        ho = 3 * hi if op == "Linear_QKV" else sout[-1]     # Q/K/V: one matmul with weights [H, 3H]
        return [(sin, 1, .01), ((hi, ho), 0, .01), ((ho,), 0, .01)], lambda x, w, b: x.matmul(w) + b
    if op == "Softmax":
        return [(sin, 1, .01)], lambda x: x.softmax(dim=-1)
    if op == "LayerNorm":
        return [(sin, 1, .1)], _layernorm
    if op == "GeLU":
        return [(sin, 1, .1)], lambda x: x.gelu()
    if op == "Residual_Add":
        return [(sin, 1, 1.), (sin, 1, 1.)], lambda a, b: a + b
    raise ValueError(f"no SHAFT benchmark for {op}")


def _layernorm(x):
    # same CrypTen formulation as bench_shaft_{cpu,gpu}.py
    mean = x.mean(dim=-1, keepdim=True)
    c = x - mean
    var = (c * c).mean(dim=-1, keepdim=True)
    return c * var.add(1e-5).reciprocal().sqrt()


def _party(device, jobs):
    rank = comm.get().get_rank()
    dev = f"cuda:{rank}" if device == "cuda" else "cpu"
    if device == "cuda":
        torch.cuda.set_device(rank)
    torch.manual_seed(0)
    sync = (lambda: torch.cuda.synchronize(dev)) if device == "cuda" else (lambda: None)
    out = []
    for name, sin, sout in jobs:
        specs, fn = _spec(name, sin, sout)
        args = [crypten.cryptensor(torch.randn(*shape) * scale, src=src, device=dev)
                for shape, src, scale in specs]
        fn(*args)                                   # warm-up
        sync()
        times, wire = [], []
        for _ in range(REPEATS):
            comm.get().barrier()
            comm.get().reset_communication_stats()
            lo0 = _lo_bytes()
            t0 = time.perf_counter()
            fn(*args)
            sync()
            t1 = time.perf_counter()
            st = comm.get().get_communication_stats()
            comm.get().barrier()
            wire.append(_lo_bytes() - lo0)
            times.append((t1 - t0) * 1e3)
        out.append({"op": name, "shape_in": list(sin), "shape_out": list(sout), "rank": rank,
                    "ms": statistics.median(times), "ms_all": times,
                    "rounds": st["rounds"], "logical_bytes": st["bytes"],
                    "lo_bytes": statistics.median(wire)})
    return out


def bert_jobs(B=1, S=128, H=768, F=3072, heads=12, names=None):
    ops = _ops(B, S, H, F, heads)
    return [(n, ops[n][2], ops[n][3]) for n in (names or list(ops))]


def run(device="cpu", provider="TFP", jobs=None):
    """jobs: [(op_type, input_shape, output_shape)]; default: BERT-base at S=128."""
    jobs = [(n, list(i), list(o)) for n, i, o in (jobs or bert_jobs())]
    names = [n for n, _, _ in jobs]
    crypten.cfg.mpc.provider = provider            # read by the forked parties and the TTP check
    wrapped = run_multiprocess(world_size=2)(_party)
    res = wrapped(device, jobs)
    if res is None:
        raise RuntimeError("a CrypTen party failed; see the log above")
    # values sorted by rank; the TTP server (if any) returns None
    by_rank = {v[0]["rank"]: v for v in res if isinstance(v, list) and v}
    rows = []
    for i, name in enumerate(names):
        p0, p1 = by_rank[0][i], by_rank[1][i]
        rows.append({"op": name, "shape_in": p0["shape_in"], "shape_out": p0["shape_out"],
                     "device": device, "provider": provider,
                     "ms": max(p0["ms"], p1["ms"]), "ms_parties": [p0["ms"], p1["ms"]],
                     "rounds_parties": [p0["rounds"], p1["rounds"]],
                     "logical_bytes_parties": [p0["logical_bytes"], p1["logical_bytes"]],
                     "wire_bytes": max(p0["lo_bytes"], p1["lo_bytes"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--provider", default="TFP", choices=["TFP", "TTP"])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--ops", default="")
    ap.add_argument("--jobs", default=None, help="JSON file: [[op_type, input_shape, output_shape], ...]")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()
    jobs = (json.loads(Path(a.jobs).read_text()) if a.jobs else
            bert_jobs(S=a.seq, names=[o for o in a.ops.split(",") if o] or None))
    rows = run(a.device, a.provider, jobs)
    for r in rows:
        print(json.dumps(r), flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

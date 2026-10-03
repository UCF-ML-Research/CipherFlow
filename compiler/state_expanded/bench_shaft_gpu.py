"""Per-op SHAFT (CrypTen) GPU wallclock benchmark.

Measures real local compute wallclock on GPU for each operator in the
BERT block using CrypTen's encrypted tensor primitives in the same
comp-only configuration SHAFT uses (computation timing without
communication — see SHAFT/examples/text-classification/run_glue_private.py
comp mode).

Communication bytes/rounds are **not** measured here — they're
protocol-level properties that don't change between CPU and GPU
execution, so we reuse the aggregate SHAFT end-to-end measurement from
``baseline/SHAFT_communication.json`` (allocated per-op proportionally).

Why per-op and not end-to-end?
- The 12-block end-to-end SHAFT run allocates ~20 GB of GPU memory
  during communication buffering. Free memory on this shared GPU is
  9.4 GB.
- Per-op benchmarks for the operators we care about fit in 2-3 GB
  each and produce directly usable ``local_compute_ms`` for the
  compiler's profile.

Output: prints ``[shaft_stats] op=<name> shape_in=... shape_out=... ms=<t>``
lines matching the existing ``[mpc_stats]`` format.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SHAFT_ROOT = Path(os.environ.get("CIPHERFLOW_SHAFT_ROOT",
                               Path(__file__).resolve().parents[2] / "baseline" / "SHAFT"))
sys.path.insert(0, str(SHAFT_ROOT))

import torch
import crypten
import crypten.nn as cnn

DEVICE = "cuda"
REPEATS = 3
WARMUPS = 1


def _time_gpu(thunk):
    """Wallclock over ``REPEATS`` runs using CUDA events for accuracy."""
    for _ in range(WARMUPS):
        thunk()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    stops  = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
    for i in range(REPEATS):
        starts[i].record()
        thunk()
        stops[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(stops[i]) for i in range(REPEATS)]
    return sum(samples) / len(samples)


def _enc(x: torch.Tensor):
    """Encrypt a plaintext tensor via CrypTen on GPU."""
    c = crypten.cryptensor(x)
    # Move encrypted shares to GPU.
    c.share = c.share.to(DEVICE)
    return c


def bench_attention_qk(batch=1, seq=128, heads=12, hidden=768):
    q = torch.randn(batch, heads, seq, hidden // heads, device=DEVICE) * 0.01
    k = torch.randn(batch, heads, seq, hidden // heads, device=DEVICE) * 0.01
    eq = _enc(q); ek = _enc(k)
    def run():
        _ = eq.matmul(ek.transpose(-1, -2))
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=Attention_QK_MatMul shape_in=[3,{batch},{seq},{hidden}] shape_out=[{batch},{heads},{seq},{seq}] ms={ms:.2f}")


def bench_attention_v(batch=1, seq=128, heads=12, hidden=768):
    attn = torch.randn(batch, heads, seq, seq, device=DEVICE) * 0.01
    v    = torch.randn(batch, heads, seq, hidden // heads, device=DEVICE) * 0.01
    ea = _enc(attn); ev = _enc(v)
    def run():
        _ = ea.matmul(ev)
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=Attention_V_MatMul shape_in=[{batch},{heads},{seq},{seq}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_linear(name, batch, seq, hidden_in, hidden_out):
    x = torch.randn(batch, seq, hidden_in, device=DEVICE) * 0.01
    w = torch.randn(hidden_in, hidden_out, device=DEVICE) * 0.01
    b = torch.randn(hidden_out, device=DEVICE) * 0.01
    ex = _enc(x); ew = _enc(w); eb = _enc(b)
    def run():
        _ = ex.matmul(ew) + eb
    ms = _time_gpu(run)
    print(f"[shaft_stats] op={name} shape_in=[{batch},{seq},{hidden_in}] shape_out=[{batch},{seq},{hidden_out}] ms={ms:.2f}")


def bench_softmax(batch=1, heads=12, seq=128):
    x = torch.randn(batch, heads, seq, seq, device=DEVICE) * 0.01
    ex = _enc(x)
    def run():
        _ = ex.softmax(dim=-1)
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=Softmax shape_in=[{batch},{heads},{seq},{seq}] shape_out=[{batch},{heads},{seq},{seq}] ms={ms:.2f}")


def bench_layernorm(batch=1, seq=128, hidden=768):
    x = torch.randn(batch, seq, hidden, device=DEVICE) * 0.1
    ex = _enc(x)
    # CrypTen approximates LayerNorm via sqrt/reciprocal protocols.
    def run():
        mean = ex.mean(dim=-1, keepdim=True)
        c = ex - mean
        var = (c * c).mean(dim=-1, keepdim=True)
        inv = var.add(1e-5).reciprocal()
        _ = c * inv.sqrt()
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=LayerNorm shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_gelu(batch=1, seq=128, hidden=3072):
    x = torch.randn(batch, seq, hidden, device=DEVICE) * 0.1
    ex = _enc(x)
    def run():
        _ = ex.gelu()
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=GeLU shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_residual_add(batch=1, seq=128, hidden=768):
    a = torch.randn(batch, seq, hidden, device=DEVICE)
    b = torch.randn(batch, seq, hidden, device=DEVICE)
    ea = _enc(a); eb = _enc(b)
    def run():
        _ = ea + eb
    ms = _time_gpu(run)
    print(f"[shaft_stats] op=Residual_Add shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    torch.manual_seed(0)
    crypten.init()
    # Pin to GPU 0 (after CUDA_VISIBLE_DEVICES remap).
    torch.cuda.set_device(0)
    print(f"[init] device={DEVICE} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)}")

    # ----- seq=16 (per-block scope) -----
    print("\n== per-block scope (B=1, S=16) ==")
    bench_attention_qk(1, 16, 12, 768)
    bench_attention_v(1, 16, 12, 768)
    bench_linear("Out_Projection", 1, 16, 768, 768)
    bench_linear("FFN_Linear_1",   1, 16, 768, 3072)
    bench_linear("FFN_Linear_2",   1, 16, 3072, 768)
    bench_softmax(1, 12, 16)
    bench_layernorm(1, 16, 768)
    bench_gelu(1, 16, 3072)
    bench_residual_add(1, 16, 768)

    # ----- seq=128 (full-model scope) -----
    print("\n== full-model scope (B=1, S=128) ==")
    bench_attention_qk(1, 128, 12, 768)
    bench_attention_v(1, 128, 12, 768)
    bench_linear("Out_Projection", 1, 128, 768, 768)
    bench_linear("FFN_Linear_1",   1, 128, 768, 3072)
    bench_linear("FFN_Linear_2",   1, 128, 3072, 768)
    bench_softmax(1, 12, 128)
    bench_layernorm(1, 128, 768)
    bench_gelu(1, 128, 3072)
    bench_residual_add(1, 128, 768)


if __name__ == "__main__":
    main()

"""Per-op SHAFT (CrypTen) CPU wallclock benchmark.

CPU mirror of bench_shaft_gpu.py. Same operators, same shapes, same
CrypTen 2PC semantics; only DEVICE differs. Timing uses wallclock since
CUDA events are GPU-only.

Output: prints ``[shaft_stats] op=<name> shape_in=... shape_out=... ms=<t>``
lines that the assemble step parses into profile_cpu_full_model_shaft.json.

Why a CPU SHAFT profile?
The original CPU profile uses SCI-BOLT primitives (n_matrix_mul_iron etc.)
which are tuned for transformer non-linearities, not for dense FFN linear
matmul. Forcing every BERT operator through BOLT primitives gives a
deeply pessimistic Pure-MPC baseline (10^5-10^6 s). SHAFT/CrypTen has
proper 2PC matmul protocols and reports real BERT-base in 28.6s LAN /
66.46s WAN. Replacing the MPC half of the CPU profile with SHAFT lets
the placement-fair Pure-MPC column be computed against a state-of-the-art
2PC-MPC system, removing the apparent 10^5x ``slowdown'' artefact.
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

DEVICE = "cpu"
REPEATS = 3
WARMUPS = 1


def _time_cpu(thunk):
    for _ in range(WARMUPS):
        thunk()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        thunk()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def _enc(x: torch.Tensor):
    return crypten.cryptensor(x)


def bench_attention_qk(batch=1, seq=128, heads=12, hidden=768):
    q = torch.randn(batch, heads, seq, hidden // heads) * 0.01
    k = torch.randn(batch, heads, seq, hidden // heads) * 0.01
    eq = _enc(q); ek = _enc(k)
    def run():
        _ = eq.matmul(ek.transpose(-1, -2))
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=Attention_QK_MatMul shape_in=[3,{batch},{seq},{hidden}] shape_out=[{batch},{heads},{seq},{seq}] ms={ms:.2f}")


def bench_attention_v(batch=1, seq=128, heads=12, hidden=768):
    attn = torch.randn(batch, heads, seq, seq) * 0.01
    v    = torch.randn(batch, heads, seq, hidden // heads) * 0.01
    ea = _enc(attn); ev = _enc(v)
    def run():
        _ = ea.matmul(ev)
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=Attention_V_MatMul shape_in=[{batch},{heads},{seq},{seq}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_linear(name, batch, seq, hidden_in, hidden_out):
    x = torch.randn(batch, seq, hidden_in) * 0.01
    w = torch.randn(hidden_in, hidden_out) * 0.01
    b = torch.randn(hidden_out) * 0.01
    ex = _enc(x); ew = _enc(w); eb = _enc(b)
    def run():
        _ = ex.matmul(ew) + eb
    ms = _time_cpu(run)
    print(f"[shaft_stats] op={name} shape_in=[{batch},{seq},{hidden_in}] shape_out=[{batch},{seq},{hidden_out}] ms={ms:.2f}")


def bench_softmax(batch=1, heads=12, seq=128):
    x = torch.randn(batch, heads, seq, seq) * 0.01
    ex = _enc(x)
    def run():
        _ = ex.softmax(dim=-1)
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=Softmax shape_in=[{batch},{heads},{seq},{seq}] shape_out=[{batch},{heads},{seq},{seq}] ms={ms:.2f}")


def bench_layernorm(batch=1, seq=128, hidden=768):
    x = torch.randn(batch, seq, hidden) * 0.1
    ex = _enc(x)
    def run():
        mean = ex.mean(dim=-1, keepdim=True)
        c = ex - mean
        var = (c * c).mean(dim=-1, keepdim=True)
        inv = var.add(1e-5).reciprocal()
        _ = c * inv.sqrt()
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=LayerNorm shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_gelu(batch=1, seq=128, hidden=3072):
    x = torch.randn(batch, seq, hidden) * 0.1
    ex = _enc(x)
    def run():
        _ = ex.gelu()
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=GeLU shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def bench_residual_add(batch=1, seq=128, hidden=768):
    a = torch.randn(batch, seq, hidden)
    b = torch.randn(batch, seq, hidden)
    ea = _enc(a); eb = _enc(b)
    def run():
        _ = ea + eb
    ms = _time_cpu(run)
    print(f"[shaft_stats] op=Residual_Add shape_in=[{batch},{seq},{hidden}] shape_out=[{batch},{seq},{hidden}] ms={ms:.2f}")


def main():
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("SHAFT_THREADS", "16")))
    crypten.init()
    print(f"[init] device={DEVICE} torch={torch.__version__} "
          f"threads={torch.get_num_threads()}")

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

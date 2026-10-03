"""Real GPU wallclock measurements on this machine.

HE (NEXUS-CUDA) — ran directly against built ``NEXUS/cuda/build/bin/main``.
Some ops could not be measured on the available GPU at the full
N=65536 parameters because another user's workload held >15 GB; we
reduced to N=32768 where possible. MatMul and Bootstrap OOM'd even at
reduced N with 9.4 GB free; those two entries are derived from the real
CPU measurement scaled by documented NEXUS-CUDA speedup.

MPC (SHAFT / CrypTen) — ran via ``bench_shaft_gpu.py`` which measures
each operator's ``local_compute_ms`` on GPU with real CrypTen
encrypted-tensor primitives (arithmetic secret shares on-GPU) using
CUDA events for timing.

For MPC communication (bytes, rounds) we anchor to the measured
end-to-end SHAFT run (``baseline/baseline_res.txt`` TEST_ID 003:
10.48 GB / 1 496 rounds on 12-block BERT seq=128) by apportioning
across operators with the same relative shape as BOLT's measured
per-op bytes/rounds, then scaling the totals. This keeps aggregate
comm numbers aligned with a real SHAFT measurement while respecting
per-op relative structure we measured on BOLT.
"""

from __future__ import annotations

from typing import Dict, Tuple

Shape = Tuple[int, ...]

# -------- Real NEXUS-CUDA wallclocks (measured at N=32768 on RTX 3090) --------
# Reduced from the paper's N=65536 to fit 9.4 GB free GPU memory
# (shared workstation). Cost at N=65536 is ~2x these numbers; we keep
# the measured N=32768 numbers and flag in metadata.
NEXUS_CUDA_MS: Dict[str, float] = {
    # Measured:
    "GELU":      126.0,    # slot count 32768
    "LayerNorm":  90.0,    # 16 x 768
    "SoftMax":    22.0,    # 128 x 128
    # OOM at N=32768 with 9 GB free; derived from real CPU
    # measurement (200 s) / published NEXUS-CUDA matmul speedup factor
    # (~20x for 4096x768x64 matmul). Flagged as derived.
    "MatMul":   10000.0,
}
NEXUS_CUDA_MEASURED: Dict[str, bool] = {
    "GELU": True, "LayerNorm": True, "SoftMax": True, "MatMul": False,
}

# Bootstrap: OOM on GPU here too. Derived from measured CPU (50 558 ms)
# / NEXUS-CUDA published speedup (~30x for logN=15 bootstrap on A100).
NEXUS_CUDA_BOOTSTRAP_MS: float = 1685.0  # 50558/30
NEXUS_CUDA_BOOTSTRAP_MEASURED: bool = False


# -------- Real SHAFT/CrypTen per-op GPU wallclocks (ms) --------
# Captured with bench_shaft_gpu.py (CUDA-event timing, WARMUPS=1, REPEATS=3).
# device=cuda, torch=2.0.1+cu118 on NVIDIA GeForce RTX 3090 sm_86.
# Both seq=16 (per-block scope) and seq=128 (full-model scope) measured.
SHAFT_GPU_MS: Dict[Tuple[str, int], float] = {
    # (op_type, seq_len) -> local_compute_ms
    # seq=16
    ("Attention_QK_MatMul",  16):  8.51,
    ("Attention_V_MatMul",   16): 19.88,
    ("Out_Projection",       16): 15.05,
    ("FFN_Linear_1",         16): 45.62,
    ("FFN_Linear_2",         16): 39.87,
    ("Softmax",              16): 64.66,
    ("LayerNorm",            16): 85.51,
    ("GeLU",                 16): 12.24,
    ("Residual_Add",         16):  0.01,
    # seq=128
    ("Attention_QK_MatMul", 128):  11.65,
    ("Attention_V_MatMul",  128):  23.42,
    ("Out_Projection",      128):  29.31,
    ("FFN_Linear_1",        128): 111.99,
    ("FFN_Linear_2",        128): 117.19,
    ("Softmax",             128):  66.60,
    ("LayerNorm",           128):  86.23,
    ("GeLU",                128):  13.54,
    ("Residual_Add",        128):   0.01,
}


# -------- Measured end-to-end SHAFT comm (from baseline/baseline_res.txt #003) --------
# Used to scale per-op bytes/rounds away from BOLT's values and toward
# real SHAFT aggregate comm. Ratios are applied to BOLT per-op records.
SHAFT_TOTAL_BYTES_12BLOCK_S128: int = 11_252_364_410   # 10.48 GB
SHAFT_TOTAL_ROUNDS_12BLOCK_S128: int = 1_496

BOLT_TOTAL_BYTES_12BLOCK_S128: int = 1_889_785_549     # 1.76 GB
BOLT_TOTAL_ROUNDS_12BLOCK_S128: int = 1_199

SHAFT_OVER_BOLT_BYTES:   float = SHAFT_TOTAL_BYTES_12BLOCK_S128 / BOLT_TOTAL_BYTES_12BLOCK_S128
SHAFT_OVER_BOLT_ROUNDS:  float = SHAFT_TOTAL_ROUNDS_12BLOCK_S128 / BOLT_TOTAL_ROUNDS_12BLOCK_S128

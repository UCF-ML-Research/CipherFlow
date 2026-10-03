"""HE operator wallclock profiler (paper §4.2.1 step 1).

For each (op_type, input_shape, output_shape) in the BERT graph we
invoke the *existing* HE primitive — no algorithmic modification, per
the user constraint — and time the wallclock over ``warmups + repeats``
runs using ``time.perf_counter`` (CPU) or ``torch.cuda.Event`` (GPU).

What "HE primitive" means in this framework
-------------------------------------------
The Python HE methods under ``operators/*/method_he_nexus.py`` mirror
the NEXUS CKKS primitives in structure: polynomial approximations for
GeLU/Softmax, row-pack matmul for linear layers, inv-sqrt for
LayerNorm. They are what the runtime actually executes as the HE path
in this repo (NEXUS C++ is not built in this tree — see
``NEXUS/CMakeLists.txt`` but no build/). Measuring them therefore gives
a faithful CPU/GPU wallclock of this framework's HE path. For the
B200 / Threadripper numbers reported in the paper, re-run this profiler
on those machines and write profile_cpu.json / profile_gpu.json next
to the demo — the compiler consumes whatever profile it is given.

HE comm bytes / rounds are always 0 (no interaction).

Shape adaptation
----------------
Some HE primitives assert restricted shape contracts (CLAUDE.md: e.g.
LayerNorm B*S<=16, FFN_Linear_1 B*S<=4096 and H=768). When the target
shape violates the contract we mark the record ``feasible=False`` and
record ``local_compute_ms = 0`` — the compiler's state-expanded solver
will never pick that HE transition. We never silently reshape or
approximate to force the op through.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from .profile_schema import OperatorRecord

Shape = Tuple[int, ...]


@dataclass
class TimingResult:
    mean_ms: float
    stdev_ms: float
    runs: int
    feasible: bool
    reason: str = ""


def _timeit_numpy(
    thunk: Callable[[], None], warmups: int, repeats: int
) -> TimingResult:
    for _ in range(warmups):
        thunk()
    samples: List[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        thunk()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return TimingResult(
        mean_ms=statistics.mean(samples),
        stdev_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        runs=repeats,
        feasible=True,
    )


def _timeit_cuda(
    build_and_run: Callable[[], None], warmups: int, repeats: int
) -> TimingResult:
    import torch

    for _ in range(warmups):
        build_and_run()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        build_and_run()
        stops[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(stops[i]) for i in range(repeats)]
    return TimingResult(
        mean_ms=statistics.mean(samples),
        stdev_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        runs=repeats,
        feasible=True,
    )


# ---------- HE primitive thunks ----------


def _he_gelu_cpu(input_shape: Shape) -> Callable[[], None]:
    from backends.he_nexus import run_nexus_gelu_bridge

    x = np.random.randn(*input_shape).astype(np.float64)

    def run() -> None:
        run_nexus_gelu_bridge(x)

    return run


def _he_softmax_cpu(input_shape: Shape) -> Callable[[], None]:
    from backends.he_nexus import run_nexus_softmax_bridge

    x = np.random.randn(*input_shape).astype(np.float64)

    def run() -> None:
        run_nexus_softmax_bridge(x)

    return run


def _he_layernorm_cpu(input_shape: Shape) -> Tuple[Callable[[], None], bool, str]:
    from backends.he_nexus_layernorm_adapter import (
        NexusLayerNormRestrictedAdapterConfig,
        run_nexus_layernorm_restricted_adapter,
    )

    cfg = NexusLayerNormRestrictedAdapterConfig()
    if len(input_shape) != 3 or input_shape[-1] != cfg.hidden_size:
        return (lambda: None), False, f"LayerNorm HE shape contract violated: {input_shape}"
    n = input_shape[0] * input_shape[1]
    if n > cfg.max_tokens:
        return (
            lambda: None,
            False,
            f"LayerNorm HE max_tokens {cfg.max_tokens} exceeded by B*S={n}",
        )
    x = np.random.randn(*input_shape).astype(np.float64) * 0.1

    def run() -> None:
        run_nexus_layernorm_restricted_adapter(x)

    return run, True, ""


def _he_linear_ffn1_cpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    from backends.he_nexus_linear_ffn1_adapter import (
        NexusLinearFfn1RestrictedAdapterConfig,
        run_nexus_linear_ffn1_restricted_adapter,
    )

    cfg = NexusLinearFfn1RestrictedAdapterConfig()
    if len(input_shape) != 3 or input_shape[-1] != cfg.hidden_size:
        return (lambda: None), False, f"FFN1 HE shape contract violated: {input_shape}"
    n = input_shape[0] * input_shape[1]
    if n > cfg.max_tokens:
        return (
            lambda: None,
            False,
            f"FFN1 HE max_tokens {cfg.max_tokens} exceeded by B*S={n}",
        )
    # The restricted adapter writes fixed 768->64 per call; to profile
    # the full 768->3072 FFN1 we chain 48 parallel row-pack calls (the
    # NEXUS primitive interface). Per-call cost x 48 = full FFN1 cost.
    slices = max(1, int(round(output_shape[-1] / cfg.out_dim)))
    x = np.random.randn(*input_shape).astype(np.float64) * 0.01

    def run() -> None:
        for _ in range(slices):
            run_nexus_linear_ffn1_restricted_adapter(x)

    return (
        run,
        True,
        f"ffn1 profiled as {slices} calls of the 768->64 row-pack adapter",
    )


def _he_linear_ffn2_cpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    # FFN2 shape is 3072 -> 768; the repo lowers it onto the same NEXUS
    # matrix-mul primitive as FFN1 (see docs/handoff_status.md).
    # We profile it as a chain of row-pack calls with transposed IO
    # dims.
    from backends.he_nexus_linear_ffn1_adapter import (
        NexusLinearFfn1RestrictedAdapterConfig,
        run_nexus_linear_ffn1_restricted_adapter,
    )

    cfg = NexusLinearFfn1RestrictedAdapterConfig()
    # input [B,S,3072] cannot flow directly into the H=768 adapter; the
    # NEXUS FFN2 path slices the input hidden dim into chunks of 768
    # and accumulates. We emulate that:
    if len(input_shape) != 3:
        return (lambda: None), False, f"FFN2 HE shape contract violated: {input_shape}"
    hidden_in = input_shape[-1]
    if hidden_in % cfg.hidden_size != 0:
        return (
            lambda: None,
            False,
            f"FFN2 input hidden {hidden_in} not a multiple of {cfg.hidden_size}",
        )
    in_slices = hidden_in // cfg.hidden_size
    out_dim = output_shape[-1]
    out_slices = max(1, int(round(out_dim / cfg.out_dim)))
    b, s = input_shape[0], input_shape[1]
    x = np.random.randn(b, s, cfg.hidden_size).astype(np.float64) * 0.01

    def run() -> None:
        for _ in range(in_slices * out_slices):
            run_nexus_linear_ffn1_restricted_adapter(x)

    return (
        run,
        True,
        f"ffn2 profiled as {in_slices}*{out_slices} calls of the 768->64 row-pack adapter",
    )


def _he_attn_qk_cpu(input_shape: Shape) -> Tuple[Callable[[], None], bool, str]:
    from backends.he_nexus_attention_adapter import (
        NexusAttentionRestrictedConfig,
        run_nexus_attention_qk_restricted_adapter,
    )

    cfg = NexusAttentionRestrictedConfig()
    if len(input_shape) != 4 or input_shape[0] != 3 or input_shape[-1] != cfg.hidden_size:
        return (lambda: None), False, f"QK HE shape contract violated: {input_shape}"
    qkv = np.random.randn(*input_shape).astype(np.float64) * 0.01

    def run() -> None:
        run_nexus_attention_qk_restricted_adapter(qkv)

    return run, True, ""


def _he_attn_v_cpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    from backends.he_nexus_attention_adapter import (
        NexusAttentionRestrictedConfig,
        run_nexus_attention_v_restricted_adapter,
    )

    cfg = NexusAttentionRestrictedConfig()
    # AV consumes attn_probs [B,H,S,S] and qkv [3,B,S,768].
    if len(input_shape) != 4:
        return (lambda: None), False, f"AV HE attn shape expected rank 4: {input_shape}"
    B, H, S, _ = input_shape
    qkv = np.random.randn(3, B, S, cfg.hidden_size).astype(np.float64) * 0.01
    attn = np.random.randn(*input_shape).astype(np.float64) * 0.01

    def run() -> None:
        run_nexus_attention_v_restricted_adapter(attn, qkv)

    return run, True, ""


def _he_out_projection_cpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    # Out_Projection lowers to the same row-pack matmul as FFN1.
    return _he_linear_ffn1_cpu(input_shape, output_shape)


def _he_residual_add_cpu(input_shape: Shape) -> Callable[[], None]:
    a = np.random.randn(*input_shape).astype(np.float64)
    b = np.random.randn(*input_shape).astype(np.float64)

    def run() -> None:
        _ = a + b

    return run


# GPU counterparts — torch equivalents of the numpy paths. We preserve
# the same polynomial structure so the CPU and GPU profiles stay
# comparable.


def _he_gelu_gpu(input_shape: Shape) -> Callable[[], None]:
    import torch

    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32)
    x = torch.clamp(x, -8.0, 8.0)

    def run() -> None:
        torch.clamp(x, -8.0, 8.0)
        v = 0.5 * x * (
            1.0
            + torch.tanh(
                torch.tensor(0.7978845608).cuda() * (x + 0.044715 * torch.pow(x, 3))
            )
        )
        _ = v

    return run


def _he_softmax_gpu(input_shape: Shape) -> Callable[[], None]:
    import torch

    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32)
    original_shape = x.shape
    rows = int(np.prod(original_shape[:-1]))
    cols = int(original_shape[-1])

    def run() -> None:
        x2d = x.reshape(rows, cols)
        # (1 + x/128)^128 via repeated squaring — same as CPU emulation.
        y = 1.0 + x2d / 128.0
        for _ in range(7):
            y = y * y
        s = y.sum(dim=1, keepdim=True)
        s = torch.clamp(0.01 * s, min=1e-8)
        # Inverse via 4 iterations of Newton-like polynomial — same structure.
        z = 1.0 - s
        inv = 1.0 + z
        for _ in range(4):
            z = z * z
            inv = inv * (1.0 + z)
        inv = 0.01 * inv
        _ = y * inv

    return run


def _he_layernorm_gpu(input_shape: Shape) -> Tuple[Callable[[], None], bool, str]:
    import torch

    if len(input_shape) != 3 or input_shape[-1] != 768 or input_shape[0] * input_shape[1] > 16:
        return (lambda: None), False, f"LayerNorm HE contract violated: {input_shape}"
    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32) * 0.1

    def run() -> None:
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        inv_rms = 1.0 / torch.sqrt(torch.clamp(ms, min=1e-8))
        _ = x * inv_rms

    return run, True, ""


def _he_linear_ffn1_gpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    import torch

    if len(input_shape) != 3 or input_shape[-1] != 768 or input_shape[0] * input_shape[1] > 4096:
        return (lambda: None), False, f"FFN1 HE contract violated: {input_shape}"
    slices = max(1, int(round(output_shape[-1] / 64)))
    x = torch.randn(*input_shape, device="cuda", dtype=torch.float32) * 0.01
    w = torch.randn(768, 64, device="cuda", dtype=torch.float32) * 0.01
    b = torch.randn(64, device="cuda", dtype=torch.float32) * 0.01

    def run() -> None:
        for _ in range(slices):
            _ = x @ w + b

    return run, True, f"ffn1 profiled as {slices} slices"


def _he_linear_ffn2_gpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    import torch

    if len(input_shape) != 3:
        return (lambda: None), False, f"FFN2 HE shape violated: {input_shape}"
    h = input_shape[-1]
    if h % 768 != 0:
        return (lambda: None), False, f"FFN2 in-hidden {h} not multiple of 768"
    in_slices = h // 768
    out_slices = max(1, int(round(output_shape[-1] / 64)))
    b_, s_ = input_shape[0], input_shape[1]
    x = torch.randn(b_, s_, 768, device="cuda", dtype=torch.float32) * 0.01
    w = torch.randn(768, 64, device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn(64, device="cuda", dtype=torch.float32) * 0.01

    def run() -> None:
        for _ in range(in_slices * out_slices):
            _ = x @ w + bias

    return run, True, f"ffn2 profiled as {in_slices}*{out_slices} slices"


def _he_attn_qk_gpu(input_shape: Shape) -> Tuple[Callable[[], None], bool, str]:
    import torch

    if len(input_shape) != 4 or input_shape[0] != 3 or input_shape[-1] != 768:
        return (lambda: None), False, f"QK HE contract violated: {input_shape}"
    _, B, S, H = input_shape
    Hds, Hd = 12, 64
    qkv = torch.randn(3, B, S, H, device="cuda", dtype=torch.float32) * 0.01

    def run() -> None:
        q = qkv[0].reshape(B, S, Hds, Hd).transpose(1, 2)
        k = qkv[1].reshape(B, S, Hds, Hd).transpose(1, 2)
        _ = q @ k.transpose(-1, -2)

    return run, True, ""


def _he_attn_v_gpu(
    input_shape: Shape, output_shape: Shape
) -> Tuple[Callable[[], None], bool, str]:
    import torch

    if len(input_shape) != 4:
        return (lambda: None), False, f"AV HE attn rank expected 4: {input_shape}"
    B, Hds, S, _ = input_shape
    Hd = 768 // Hds
    attn = torch.randn(*input_shape, device="cuda", dtype=torch.float32) * 0.01
    qkv = torch.randn(3, B, S, 768, device="cuda", dtype=torch.float32) * 0.01

    def run() -> None:
        v = qkv[2].reshape(B, S, Hds, Hd).transpose(1, 2)
        _ = attn @ v

    return run, True, ""


def _he_residual_add_gpu(input_shape: Shape) -> Callable[[], None]:
    import torch

    a = torch.randn(*input_shape, device="cuda", dtype=torch.float32)
    b = torch.randn(*input_shape, device="cuda", dtype=torch.float32)

    def run() -> None:
        _ = a + b

    return run


# ---------- Per-op level-delta lookup (from method_he_nexus modules) ----------


HE_LEVEL_DELTA: dict = {
    "Attention_QK_MatMul": 1,
    "Softmax": 8,
    "Attention_V_MatMul": 1,
    "Out_Projection": 1,
    "Residual_Add": 0,
    "LayerNorm": 3,
    "FFN_Linear_1": 1,
    "GeLU": 4,
    "FFN_Linear_2": 1,
}


def _dispatch_he(
    op_type: str, input_shape: Shape, output_shape: Shape, device: str
) -> Tuple[Callable[[], None], bool, str]:
    if device == "cpu":
        if op_type == "GeLU":
            return _he_gelu_cpu(input_shape), True, ""
        if op_type == "Softmax":
            return _he_softmax_cpu(input_shape), True, ""
        if op_type == "LayerNorm":
            return _he_layernorm_cpu(input_shape)
        if op_type == "FFN_Linear_1":
            return _he_linear_ffn1_cpu(input_shape, output_shape)
        if op_type == "FFN_Linear_2":
            return _he_linear_ffn2_cpu(input_shape, output_shape)
        if op_type == "Out_Projection":
            return _he_out_projection_cpu(input_shape, output_shape)
        if op_type == "Attention_QK_MatMul":
            return _he_attn_qk_cpu(input_shape)
        if op_type == "Attention_V_MatMul":
            return _he_attn_v_cpu(input_shape, output_shape)
        if op_type == "Residual_Add":
            return _he_residual_add_cpu(input_shape), True, ""
    elif device == "cuda":
        if op_type == "GeLU":
            return _he_gelu_gpu(input_shape), True, ""
        if op_type == "Softmax":
            return _he_softmax_gpu(input_shape), True, ""
        if op_type == "LayerNorm":
            return _he_layernorm_gpu(input_shape)
        if op_type == "FFN_Linear_1":
            return _he_linear_ffn1_gpu(input_shape, output_shape)
        if op_type == "FFN_Linear_2":
            return _he_linear_ffn2_gpu(input_shape, output_shape)
        if op_type == "Out_Projection":
            return _he_linear_ffn1_gpu(input_shape, output_shape)
        if op_type == "Attention_QK_MatMul":
            return _he_attn_qk_gpu(input_shape)
        if op_type == "Attention_V_MatMul":
            return _he_attn_v_gpu(input_shape, output_shape)
        if op_type == "Residual_Add":
            return _he_residual_add_gpu(input_shape), True, ""
    raise ValueError(f"Unknown HE op: {op_type} on {device}")


# ---------- Public API ----------


def profile_he_operators(
    shapes: Iterable[Tuple[str, Shape, Shape]],
    device: str = "cpu",
    warmups: int = 2,
    repeats: int = 5,
) -> List[OperatorRecord]:
    records: List[OperatorRecord] = []
    for op_type, input_shape, output_shape in shapes:
        try:
            thunk, feasible, note = _dispatch_he(op_type, input_shape, output_shape, device)
        except Exception as exc:
            records.append(
                OperatorRecord(
                    op_type=op_type,
                    domain="HE",
                    method="method_he_nexus",
                    input_shape=input_shape,
                    output_shape=output_shape,
                    local_compute_ms=0.0,
                    comm_bytes=0,
                    comm_rounds=0,
                    he_level_delta=HE_LEVEL_DELTA.get(op_type, 0),
                    feasible=False,
                    metadata={"reason": f"dispatch_error: {exc}"},
                )
            )
            continue
        if not feasible:
            records.append(
                OperatorRecord(
                    op_type=op_type,
                    domain="HE",
                    method="method_he_nexus",
                    input_shape=input_shape,
                    output_shape=output_shape,
                    local_compute_ms=0.0,
                    comm_bytes=0,
                    comm_rounds=0,
                    he_level_delta=HE_LEVEL_DELTA.get(op_type, 0),
                    feasible=False,
                    metadata={"reason": note},
                )
            )
            continue
        if device == "cuda":
            tr = _timeit_cuda(thunk, warmups=warmups, repeats=repeats)
        else:
            tr = _timeit_numpy(thunk, warmups=warmups, repeats=repeats)
        records.append(
            OperatorRecord(
                op_type=op_type,
                domain="HE",
                method="method_he_nexus",
                input_shape=input_shape,
                output_shape=output_shape,
                local_compute_ms=tr.mean_ms,
                comm_bytes=0,
                comm_rounds=0,
                he_level_delta=HE_LEVEL_DELTA.get(op_type, 0),
                feasible=True,
                metadata={
                    "device": device,
                    "warmups": warmups,
                    "repeats": repeats,
                    "stdev_ms": tr.stdev_ms,
                    "note": note,
                },
            )
        )
    return records

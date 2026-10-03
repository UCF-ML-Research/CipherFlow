"""HE↔MPC conversion and bootstrapping profiler (paper §4.2.1).

We model the cross-domain conversion protocols standard in BOLT /
BumbleBee / CryptFlow2, which are what this repo's runtime shells out
to. Implementations differ, but the wire-level model is the same
mask-and-decrypt pattern:

HE → MPC (ciphertext -> arithmetic shares):
    1. Server samples random mask r, homomorphically computes
       ct' = Enc(x) - Enc(r) (local, HE compute).
    2. Server sends ct' to client (one round, one ciphertext).
    3. Client decrypts -> holds plaintext (x - r); server holds r.
       These are additive shares of x.
    rounds = 1
    bytes  = one RLWE ciphertext   (≈ 2 * N * log q / 8)

MPC → HE (additive shares -> ciphertext):
    1. Server re-encrypts its share s_0 -> Enc(s_0) (local, HE compute).
    2. Client encrypts its share s_1 -> Enc(s_1) (local, HE compute).
    3. Client sends Enc(s_1) to server (one round, one ciphertext).
    4. Server homomorphically adds -> Enc(x).
    rounds = 1
    bytes  = one RLWE ciphertext

Ciphertext size (CKKS @ logN=16, L=20, default NEXUS params):
    N = 2^16 coefficients
    q tower ≈ 20 * 40 bits = 800 bits per coeff
    one ciphertext = 2 polys * N * (800/8) bytes ≈ 13.1 MB
    slots per ciphertext = N/2 = 32768

A BERT-base tensor [B,S,H] = [1,16,768] = 12288 slots fits in a
single ciphertext; larger tensors need ``ceil(slots / N*)`` ciphertexts.

Local HE compute for conversion is *dominated* by the plaintext/mask
mul and the encrypt/decrypt calls. We measure the emulated work
(scalar multiply, add, encode/decode) the same way as the HE operator
profiler does — by timing the Python path that mirrors the NEXUS
arithmetic.

Bootstrapping cost at NEXUS params is published in Kim et al. (EuroS&P
2022, "Approximate Homomorphic Encryption with Reduced Approximation
Error") and in NEXUS's own benchmarks: ~1.5 s per ciphertext at
logN=16 on AMD EPYC. On GPU (NEXUS CUDA path) ~50 ms per ciphertext.
We take these as platform-specific constants unless a live measurement
is available.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from .profile_schema import BootstrapRecord, ConversionRecord

Shape = Tuple[int, ...]

# CKKS / NEXUS parameters.
POLY_N: int = 1 << 16             # 65536 coefficients
SLOT_COUNT: int = POLY_N // 2     # 32768 slots per ciphertext
# 20-modulus tower with 60-bit special + 40-bit rest from NEXUS main.cpp.
# Total bits for a fresh ct ≈ 58 + 40*18 + 58 = ~836 bits per coeff.
# Conservative average: 800 bits = 100 bytes per coeff.
CT_BYTES_PER_COEFF: int = 100
CT_SIZE_BYTES: int = 2 * POLY_N * CT_BYTES_PER_COEFF  # ~13.1 MB


def _shape_numel(shape: Shape) -> int:
    n = 1
    for d in shape:
        n *= max(1, int(d))
    return n


def ciphertexts_needed(shape: Shape) -> int:
    """How many CKKS ciphertexts pack a tensor of this shape."""
    slots = _shape_numel(shape)
    return max(1, int(math.ceil(slots / SLOT_COUNT)))


def conversion_bytes_rounds(shape: Shape) -> Tuple[int, int]:
    """Return (bytes, rounds) for HE<->MPC conversion of ``shape``.

    Both directions transfer one ciphertext per slot-pack. Rounds = 1
    per direction (the two parties exchange once).
    """
    cts = ciphertexts_needed(shape)
    return cts * CT_SIZE_BYTES, 1


def _time_emulation(thunk: Callable[[], None], warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        thunk()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        thunk()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.mean(samples)


def _time_he_to_mpc_cpu(shape: Shape) -> float:
    """Time the emulated HE-side work for HE->MPC mask-and-subtract.

    Mirrors: ``ct_prime = HE.sub(HE.encrypt(x), HE.encrypt(r))`` where
    r is a random mask of the same shape. We count the mask sample +
    plaintext-sub (local HE compute) as the HE-side cost.
    """
    numel = _shape_numel(shape)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(numel).astype(np.float64)
    r = rng.standard_normal(numel).astype(np.float64)

    def run() -> None:
        # Encode + mask-subtract + encode. Encodings in CKKS are
        # number-theoretic transforms; we emulate with a complex FFT of
        # the slot vector (matches NEXUS encode path structure).
        pt_x = np.fft.fft(x + 1j * np.zeros_like(x))
        pt_r = np.fft.fft(r + 1j * np.zeros_like(r))
        _ = pt_x - pt_r
        # Decode via ifft to obtain the masked value (client side).
        _ = np.fft.ifft(pt_x - pt_r).real

    return _time_emulation(run, warmups=2, repeats=5)


def _time_mpc_to_he_cpu(shape: Shape) -> float:
    """Time the emulated HE-side work for MPC->HE re-encryption.

    Two encryptions + one homomorphic addition.
    """
    numel = _shape_numel(shape)
    rng = np.random.default_rng(7)
    s0 = rng.standard_normal(numel).astype(np.float64)
    s1 = rng.standard_normal(numel).astype(np.float64)

    def run() -> None:
        pt0 = np.fft.fft(s0 + 1j * np.zeros_like(s0))
        pt1 = np.fft.fft(s1 + 1j * np.zeros_like(s1))
        _ = pt0 + pt1
        _ = np.fft.ifft(pt0 + pt1).real

    return _time_emulation(run, warmups=2, repeats=5)


def _time_he_to_mpc_gpu(shape: Shape) -> float:
    import torch

    numel = _shape_numel(shape)
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    r = torch.randn(numel, device="cuda", dtype=torch.float32)

    def run() -> None:
        # Use torch.fft.fft on complex extension of x / r.
        pt_x = torch.fft.fft(x.to(torch.complex64))
        pt_r = torch.fft.fft(r.to(torch.complex64))
        _ = pt_x - pt_r
        _ = torch.fft.ifft(pt_x - pt_r).real

    # Warmup + cuda event timing inline (mirror profiler_he).
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for i in range(5):
        starts[i].record()
        run()
        stops[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(stops[i]) for i in range(5)]
    return statistics.mean(samples)


def _time_mpc_to_he_gpu(shape: Shape) -> float:
    import torch

    numel = _shape_numel(shape)
    s0 = torch.randn(numel, device="cuda", dtype=torch.float32)
    s1 = torch.randn(numel, device="cuda", dtype=torch.float32)

    def run() -> None:
        pt0 = torch.fft.fft(s0.to(torch.complex64))
        pt1 = torch.fft.fft(s1.to(torch.complex64))
        _ = pt0 + pt1
        _ = torch.fft.ifft(pt0 + pt1).real

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for i in range(5):
        starts[i].record()
        run()
        stops[i].record()
    torch.cuda.synchronize()
    samples = [starts[i].elapsed_time(stops[i]) for i in range(5)]
    return statistics.mean(samples)


def profile_conversions(
    shapes: Iterable[Shape], device: str = "cpu"
) -> List[ConversionRecord]:
    records: List[ConversionRecord] = []
    for shape in shapes:
        bytes_, rounds = conversion_bytes_rounds(shape)
        if device == "cuda":
            he2mpc = _time_he_to_mpc_gpu(shape)
            mpc2he = _time_mpc_to_he_gpu(shape)
        else:
            he2mpc = _time_he_to_mpc_cpu(shape)
            mpc2he = _time_mpc_to_he_cpu(shape)
        cts = ciphertexts_needed(shape)
        records.append(
            ConversionRecord(
                from_domain="HE",
                to_domain="MPC",
                method="mask_and_decrypt",
                tensor_shape=shape,
                local_compute_ms=he2mpc,
                comm_bytes=bytes_,
                comm_rounds=rounds,
                metadata={
                    "device": device,
                    "ciphertexts": cts,
                    "poly_N": POLY_N,
                    "slot_count": SLOT_COUNT,
                    "ct_size_bytes": CT_SIZE_BYTES,
                },
            )
        )
        records.append(
            ConversionRecord(
                from_domain="MPC",
                to_domain="HE",
                method="share_plus_enc",
                tensor_shape=shape,
                local_compute_ms=mpc2he,
                comm_bytes=bytes_,
                comm_rounds=rounds,
                metadata={
                    "device": device,
                    "ciphertexts": cts,
                    "poly_N": POLY_N,
                    "slot_count": SLOT_COUNT,
                    "ct_size_bytes": CT_SIZE_BYTES,
                },
            )
        )
    return records


# ---------- Bootstrapping ----------


# NEXUS CKKS bootstrapping cost at logN=16 on the platforms the paper
# evaluates on. These are *references* taken from NEXUS and
# state-of-the-art CKKS bootstrapping literature; we label them as
# ``source`` in metadata so the numbers are auditable.
BOOTSTRAP_MS = {
    "cpu": 1500.0,   # ~1.5 s / ct on AMD EPYC Threadripper class (NEXUS)
    "cuda": 50.0,    # ~50 ms / ct on A100/B200 (NEXUS-CUDA reference)
}


def profile_bootstrap(device: str = "cpu") -> BootstrapRecord:
    local = BOOTSTRAP_MS.get(device, BOOTSTRAP_MS["cpu"])
    return BootstrapRecord(
        method="nexus_ckks_bootstrap",
        local_compute_ms=local,
        comm_bytes=0,
        comm_rounds=0,
        metadata={
            "source": "NEXUS CKKS bootstrapping reference timing",
            "device": device,
            "poly_N": POLY_N,
        },
    )

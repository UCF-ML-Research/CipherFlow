from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NexusLinearFfn1RestrictedAdapterConfig:
    """
    Restricted linear FFN1 adapter that mirrors NEXUS matrix-mul packing assumptions.

    Reused NEXUS internals (reference mapping):
    - he_compiler/NEXUS/src/matrix_mul.cpp
      - MMEvaluator::matrix_mul
      - MMEvaluator::enc_compress_ciphertext
      - MMEvaluator::expand_ciphertext
      - row-pack path in MM_test()

    Supported contract (restricted):
    - input x: [B, S, H]
    - H must equal 768
    - output dim O must equal 64
    - flattened token count N = B*S must satisfy 1 <= N <= 4096
    """

    hidden_size: int = 768
    out_dim: int = 64
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 1234


def _deterministic_params(hidden_size: int, out_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((hidden_size, out_dim)).astype(np.float64)
    b = rng.standard_normal((out_dim,)).astype(np.float64)
    return w, b


def _nexus_row_pack_weight(weight: np.ndarray, poly_modulus_degree: int) -> list[np.ndarray]:
    """
    Pack weight using the same row-major chunking style as NEXUS MM_test().

    NEXUS MM packs transpose(weight) rows into slot-sized chunks.
    """
    out_dim, hidden = weight.T.shape
    total = out_dim * hidden
    row_ct = np.zeros((poly_modulus_degree,), dtype=np.float64)
    packs: list[np.ndarray] = []

    for i in range(total):
        row = i // hidden
        col = i % hidden
        row_ct[i % poly_modulus_degree] = weight.T[row, col]
        if i % poly_modulus_degree == (poly_modulus_degree - 1):
            packs.append(row_ct.copy())

    if total % poly_modulus_degree != 0:
        packs.append(row_ct.copy())
    return packs


def _nexus_unpack_weight(
    row_packs: list[np.ndarray], hidden_size: int, out_dim: int, poly_modulus_degree: int
) -> np.ndarray:
    total = hidden_size * out_dim
    flat = np.zeros((total,), dtype=np.float64)
    idx = 0
    for pack in row_packs:
        take = min(poly_modulus_degree, total - idx)
        flat[idx : idx + take] = pack[:take]
        idx += take
        if idx >= total:
            break
    wt = flat.reshape(out_dim, hidden_size)
    return wt.T.copy()


def run_nexus_linear_ffn1_restricted_adapter(
    x: np.ndarray,
    cfg: NexusLinearFfn1RestrictedAdapterConfig | None = None,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or NexusLinearFfn1RestrictedAdapterConfig()
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(f"NEXUS FFN1 restricted adapter expects [B,S,H], got shape={x.shape}")
    bsz, seq, hidden = x.shape
    if hidden != cfg.hidden_size:
        raise ValueError(f"Restricted adapter supports H={cfg.hidden_size} only, got H={hidden}")
    n = bsz * seq
    if n <= 0 or n > cfg.max_tokens:
        raise ValueError(f"Restricted adapter supports 1 <= B*S <= {cfg.max_tokens}, got B*S={n}")

    if weight is None or bias is None:
        w, b = _deterministic_params(cfg.hidden_size, cfg.out_dim, cfg.weight_seed)
    else:
        w = np.asarray(weight, dtype=np.float64)
        b = np.asarray(bias, dtype=np.float64)
        if w.shape != (cfg.hidden_size, cfg.out_dim):
            raise ValueError(
                f"weight shape must be [{cfg.hidden_size},{cfg.out_dim}] for restricted adapter, got {w.shape}"
            )
        if b.shape != (cfg.out_dim,):
            raise ValueError(f"bias shape must be [{cfg.out_dim}], got {b.shape}")

    packed = _nexus_row_pack_weight(w, cfg.poly_modulus_degree)
    w_recovered = _nexus_unpack_weight(packed, cfg.hidden_size, cfg.out_dim, cfg.poly_modulus_degree)

    x2d = x.reshape(n, hidden)
    y2d = x2d @ w_recovered + b
    y = y2d.reshape(bsz, seq, cfg.out_dim)
    return y, w_recovered, b

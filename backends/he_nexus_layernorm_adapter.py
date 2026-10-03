from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NexusLayerNormRestrictedAdapterConfig:
    """
    Restricted LayerNorm adapter that mirrors current NEXUS layer-norm assumptions.

    NEXUS internals reference:
    - he_compiler/NEXUS/src/layer_norm.cpp
      - LNEvaluator::layer_norm(Ciphertext&, Ciphertext&, int len)
    - he_compiler/NEXUS/src/main.cpp
      - demo path uses a 768-feature vector packed with len=1024

    Supported contract (restricted):
    - input x: [B, S, H]
    - H must equal 768
    - flattened token count N = B*S must satisfy 1 <= N <= 16
    - affine weight/bias are not supported
    - normalization follows the current NEXUS execution style:
      x / sqrt(mean(x^2)) over the last dimension

    This is intentionally narrower than a general LayerNorm contract.
    """

    hidden_size: int = 768
    max_tokens: int = 16
    packed_len: int = 1024
    eps: float = 1e-8


def run_nexus_layernorm_restricted_adapter(
    x: np.ndarray,
    cfg: NexusLayerNormRestrictedAdapterConfig | None = None,
) -> np.ndarray:
    cfg = cfg or NexusLayerNormRestrictedAdapterConfig()
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 3:
        raise ValueError(f"NEXUS LayerNorm restricted adapter expects [B,S,H], got shape={x.shape}")
    bsz, seq, hidden = x.shape
    if hidden != cfg.hidden_size:
        raise ValueError(f"Restricted adapter supports H={cfg.hidden_size} only, got H={hidden}")
    n = bsz * seq
    if n <= 0 or n > cfg.max_tokens:
        raise ValueError(f"Restricted adapter supports 1 <= B*S <= {cfg.max_tokens}, got B*S={n}")
    if cfg.packed_len < cfg.hidden_size:
        raise ValueError(f"packed_len must be >= hidden_size, got {cfg.packed_len} < {cfg.hidden_size}")

    mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(np.maximum(mean_square, cfg.eps))
    return x * inv_rms

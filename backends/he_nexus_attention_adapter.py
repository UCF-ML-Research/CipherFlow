from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NexusAttentionRestrictedConfig:
    """
    Restricted attention-matmul adapter backed by NEXUS matrix-mul style assumptions.

    NEXUS internals reference:
    - he_compiler/NEXUS/src/matrix_mul.cpp
      - MMEvaluator::matrix_mul
      - MMEvaluator::enc_compress_ciphertext
      - MMEvaluator::expand_ciphertext
    - he_compiler/NEXUS/src/main.cpp MM_test() packing workflow

    Supported contract:
    - hidden_size = 768
    - num_heads = 12 (head_dim = 64)
    - 1 <= seq_len <= 128
    - 1 <= batch <= 8
    - packed qkv input layout [3,B,S,768]
    """

    hidden_size: int = 768
    num_heads: int = 12
    max_seq_len: int = 128
    max_batch: int = 8


def _to_bhsd_from_bsh(x: np.ndarray, heads: int) -> np.ndarray:
    bsz, seq, hidden = x.shape
    if hidden % heads != 0:
        raise ValueError(f"hidden size {hidden} not divisible by heads {heads}")
    head_dim = hidden // heads
    return x.reshape(bsz, seq, heads, head_dim).transpose(0, 2, 1, 3)


def _validate_restricted_qkv(qkv: np.ndarray, cfg: NexusAttentionRestrictedConfig) -> tuple[int, int]:
    if qkv.ndim != 4 or qkv.shape[0] != 3:
        raise ValueError(f"Restricted adapter requires packed qkv shape [3,B,S,H], got {qkv.shape}")
    _, bsz, seq, hidden = qkv.shape
    if hidden != cfg.hidden_size:
        raise ValueError(f"Restricted adapter supports hidden_size={cfg.hidden_size} only, got {hidden}")
    if bsz < 1 or bsz > cfg.max_batch:
        raise ValueError(f"Restricted adapter supports 1 <= B <= {cfg.max_batch}, got {bsz}")
    if seq < 1 or seq > cfg.max_seq_len:
        raise ValueError(f"Restricted adapter supports 1 <= S <= {cfg.max_seq_len}, got {seq}")
    return bsz, seq


def run_nexus_attention_qk_restricted_adapter(
    qkv_packed: np.ndarray,
    cfg: NexusAttentionRestrictedConfig | None = None,
) -> np.ndarray:
    cfg = cfg or NexusAttentionRestrictedConfig()
    qkv = np.asarray(qkv_packed, dtype=np.float64)
    _validate_restricted_qkv(qkv, cfg)
    q = _to_bhsd_from_bsh(qkv[0], cfg.num_heads)
    k = _to_bhsd_from_bsh(qkv[1], cfg.num_heads)
    return q @ np.swapaxes(k, -1, -2)


def run_nexus_attention_v_restricted_adapter(
    attn_probs: np.ndarray,
    qkv_packed: np.ndarray,
    cfg: NexusAttentionRestrictedConfig | None = None,
    return_canonical: bool = False,
) -> np.ndarray:
    cfg = cfg or NexusAttentionRestrictedConfig()
    qkv = np.asarray(qkv_packed, dtype=np.float64)
    bsz, seq = _validate_restricted_qkv(qkv, cfg)
    attn = np.asarray(attn_probs, dtype=np.float64)
    expected_attn_shape = (bsz, cfg.num_heads, seq, seq)
    if attn.shape != expected_attn_shape:
        raise ValueError(f"Restricted adapter requires attn_probs shape {expected_attn_shape}, got {attn.shape}")
    v = _to_bhsd_from_bsh(qkv[2], cfg.num_heads)
    context_bhsd = attn @ v
    if return_canonical:
        return context_bhsd
    return context_bhsd.transpose(0, 2, 1, 3).reshape(bsz, seq, cfg.hidden_size)

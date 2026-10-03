from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NexusHeGeluBridgeConfig:
    """
    NEXUS-based GeLU bridge configuration.

    Source reference:
    - he_compiler/NEXUS/src/gelu.cpp (GeLUEvaluator::gelu)
    - he_compiler/NEXUS/data/data_generation.py (gelu calibration function)

    Notes:
    - This bridge mirrors the target GeLU behavior in plaintext for framework integration.
    - It does not perform encrypted execution in this Python layer.
    """

    clamp_min: float = -8.0
    clamp_max: float = 8.0


@dataclass
class NexusHeSoftmaxBridgeConfig:
    """
    NEXUS-based Softmax bridge configuration.

    Source reference:
    - he_compiler/NEXUS/src/softmax.cpp (SoftmaxEvaluator::softmax)
    - he_compiler/NEXUS/src/ckks_evaluator.cpp:
      - CKKSEvaluator::exp
      - CKKSEvaluator::inverse

    Notes:
    - Uses NEXUS-style approximation:
      exp(x) ~= (1 + x / 128)^128 and inverse via iterative polynomial scheme.
    - Operates on plaintext tensors in this integration layer.
    """

    inverse_iterations: int = 4
    sum_scale_factor: float = 0.01
    eps: float = 1e-8


def _nexus_exp_approx(x: np.ndarray) -> np.ndarray:
    # Mirrors CKKSEvaluator::exp: (1 + x / 128)^128 via repeated squaring.
    y = 1.0 + (x / 128.0)
    for _ in range(7):
        y = y * y
    return y


def _nexus_inverse_approx(x: np.ndarray, iterations: int) -> np.ndarray:
    # Mirrors CKKSEvaluator::inverse iterative structure from NEXUS.
    y = 1.0 - x
    res = 1.0 + y
    for _ in range(iterations):
        y = y * y
        res = res * (1.0 + y)
    return res


def run_nexus_gelu_bridge(x: np.ndarray, cfg: NexusHeGeluBridgeConfig | None = None) -> np.ndarray:
    cfg = cfg or NexusHeGeluBridgeConfig()
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, cfg.clamp_min, cfg.clamp_max)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def run_nexus_softmax_bridge(x: np.ndarray, cfg: NexusHeSoftmaxBridgeConfig | None = None) -> np.ndarray:
    cfg = cfg or NexusHeSoftmaxBridgeConfig()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError(f"NEXUS softmax bridge expects ndim>=2 with last axis as row width, got shape={x.shape}")

    original_shape = x.shape
    rows = int(np.prod(original_shape[:-1]))
    cols = int(original_shape[-1])
    x2d = x.reshape(rows, cols)

    exp_x = _nexus_exp_approx(x2d)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)

    # Same normalization trick as NEXUS SoftmaxEvaluator::softmax.
    scaled_sum = np.maximum(cfg.sum_scale_factor * sum_exp, cfg.eps)
    inv_sum = _nexus_inverse_approx(scaled_sum, cfg.inverse_iterations)
    inv_sum = cfg.sum_scale_factor * inv_sum

    out = exp_x * inv_sum
    return out.reshape(original_shape)

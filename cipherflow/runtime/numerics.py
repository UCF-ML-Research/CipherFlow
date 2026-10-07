"""Numerical models of the backends, for executing a plan on real data (emulation).

HE (the OpenFHE CKKS kernels, native/openfhe/he_ops.cpp)
  * linear layers, attention products and residual adds: exact (CKKS noise is ~1e-9);
  * GeLU, LayerNorm and Softmax: the kernels' algorithms, with the same Chebyshev interpolations
    (function, interval, degree) evaluated as OpenFHE's EvalChebyshevFunction does (interpolation
    at Chebyshev nodes of the first kind; outside the interval the polynomial is evaluated as is);
  * bootstrapping, inside those kernels and where a plan inserts it: Gaussian noise of the measured
    size (one ciphertext of values in [-1, 1]; scaled with the largest magnitude above 1).
MPC
  * exact, then rounded to the backend's fixed point (fractional bits); the protocols' own
    approximations of non-linear functions are not modeled.
Conversions (the hardened protocol, native/openfhe/conv_hardened.cpp)
  * HE -> MPC rounds value + noise + a mask offset within +-1/2 step to the MPC fixed point;
  * MPC -> HE adds the error the wrap removal leaves (Gaussian, measured).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ..kernel_params import (BOOTSTRAP_FLAT, BOOTSTRAP_RANGE, CONVERSION_BOUND_BITS, HE_NUMERICS,  # noqa: F401
                             MPC_FRAC_BITS, MPC_TO_HE_SIGMA, HENumerics, bootstrap_amplitude)


# ---------------------------------------------------------------- building blocks

def gelu_exact(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x):
    return torch.erf(x) if isinstance(x, torch.Tensor) else np.vectorize(math.erf)(x)


@lru_cache(maxsize=None)
def _cheb(kind: str, a: float, b: float, degree: int, p1: float = 0.0, p2: float = 0.0):
    f: Callable = {
        "gelu": lambda v: 0.5 * v * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0))),
        "inv_sqrt": lambda v: 1.0 / np.sqrt(v * p1 + p2),          # 1 / sqrt(v * hi + eps)
        "exp_shift": lambda v: np.exp(v - p1),                      # exp(v - bound)
        "ratio": lambda v: p1 / v,                                  # umin / u
    }[kind]
    return np.polynomial.chebyshev.Chebyshev.interpolate(f, degree, domain=[a, b])


def chebyshev(x: torch.Tensor, kind: str, a: float, b: float, degree: int, p1: float = 0.0, p2: float = 0.0):
    poly = _cheb(kind, float(a), float(b), int(degree), float(p1), float(p2))
    with np.errstate(over="ignore", invalid="ignore"):      # outside [a, b] the polynomial may overflow
        return torch.from_numpy(poly(x.detach().cpu().double().numpy())).to(x.dtype)


def fixed_point(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    s = float(2 ** frac_bits)
    return torch.round(x * s) / s


def to_mpc(x: torch.Tensor, frac_bits: int, gen: torch.Generator) -> torch.Tensor:
    """Hardened HE -> MPC: the client rounds (value + noise + a mask offset within +-1/2 step)."""
    step = 2.0 ** -frac_bits
    off = (torch.rand(x.shape, generator=gen, dtype=torch.float64) - 0.5).to(x.dtype) * step
    return fixed_point(x + off, frac_bits)


def to_he(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """Hardened MPC -> HE: the wrap removal in HE leaves Gaussian error of `sigma`."""
    return x + (torch.randn(x.shape, generator=gen, dtype=torch.float64) * sigma).to(x.dtype)


def bootstrap(x: torch.Tensor, num: HENumerics, gen: torch.Generator, scale: float = 1.0) -> Tuple[torch.Tensor, bool]:
    """Refreshed values, and whether they were inside the bootstrapping range. The error is the
    bootstrap's base noise up to |x| = BOOTSTRAP_FLAT and grows ~cubically beyond (he_ops bootlevel);
    a scaled bootstrap (x / scale, bootstrap, * scale) multiplies the base noise by the scale. The
    values' mean counts with weight BOOTSTRAP_MEAN_WEIGHT (it sits in one coefficient)."""
    peak = (bootstrap_amplitude(x.abs().max(), x.mean().abs()) if x.numel() else 0.0) / scale
    if not math.isfinite(peak):                                  # already wrong: stays wrong
        return x, False
    sigma = num.bootstrap_sigma * scale * max(1.0, min(peak / BOOTSTRAP_FLAT, 1e30) ** 3)
    noise = torch.randn(x.shape, generator=gen, dtype=torch.float64) * sigma
    return x + noise.to(x.dtype), peak < BOOTSTRAP_RANGE


# ---------------------------------------------------------------- HE operators

def he_gelu(x, num: HENumerics, stats: dict):
    b = num.gelu_bound
    stats.update(input_min=float(x.min()), input_max=float(x.max()), interval=[-b, b],
                 outside=float(((x < -b) | (x > b)).double().mean()))
    return chebyshev(x, "gelu", -b, b, num.gelu_degree)


def cheb_series(x: torch.Tensor, coeffs) -> torch.Tensor:
    """sum c_k T_k(x) (numpy's convention), as the kernels evaluate it (outside [-1, 1] too)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return torch.from_numpy(np.polynomial.chebyshev.chebval(x.detach().cpu().double().numpy(),
                                                                np.asarray(coeffs))).to(x.dtype)


def he_gelu_wide(x, bound: int, num: HENumerics, gen, stats: dict):
    """The wide-range GeLU (native/openfhe/NONLINEAR.md): x h(f1(x / B)), bootstrapped between f1 and h."""
    from ..kernel_params import gelu_wide_coefficients
    c = gelu_wide_coefficients(bound)
    stats.update(input_min=float(x.min()), input_max=float(x.max()), interval=[-bound, bound], kernel="wide",
                 outside=float(((x < -bound) | (x > bound)).double().mean()), bootstraps=1)
    v, ok = bootstrap(cheb_series(x / bound, c["f1"]), num, gen)
    stats["bootstrap_in_range"] = ok
    return x * cheb_series(v, c["h"])


def he_softmax_wide(x, params: dict, num: HENumerics, gen, stats: dict):
    """The wide-range Softmax (native/openfhe/NONLINEAR.md) over the last axis, as he_ops softmax_wide
    computes it: exp((x - c) / 2^k), a rough normalisation, k rounds of square-and-normalise; the row
    sums bootstrapped in [-1, 1], the Newton iterates and the rows (scaled by sqrt(amp)) in the flat range."""
    from ..kernel_params import softmax_wide_constants
    K = softmax_wide_constants()
    c, lo, k, W = params["c"], params["lo"], int(params["k"]), params["W"]
    n = x.shape[-1]
    sig, amp = num.bootstrap_sigma, K["amp"]
    stats.update(input_min=float(x.min()), input_max=float(x.max()), interval=[lo, c], kernel="wide", rounds=k,
                 outside=float(((x < lo) | (x > c)).double().mean()),
                 rowmax_min=float(x.max(-1).values.min()))
    noise = lambda t, s: t + torch.randn(t.shape, generator=gen, dtype=torch.float64) * s
    u = (2 * (x - lo) / (c - lo) - 1).double()
    pexp = np.polynomial.chebyshev.Chebyshev.interpolate(lambda v: np.exp(W * (v - 1) / 2), K["exp_degree"])
    y = torch.from_numpy(pexp(u.detach().cpu().numpy()))
    low = 0
    for j in range(k + 1):
        last = j == k
        f = K["first"] if j == 0 else K["rounds"]
        if j == min(2, k):
            y = noise(y, sig)                                    # the rows' bootstrap (values <= sqrt(amp))
        s = y if j == 0 else y * y
        S = s.sum(-1, keepdim=True)
        a, b = f["a"], f["b"]                                    # rows shorter than n: sums inside too
        low += int((S < a).sum() + (S > b).sum())
        z = cheb_series(noise((2 * S - (a + b)) / (b - a), sig), f["coef"])   # sums bootstrapped in [-1, 1]
        if last:                                                 # Newton steps on the exact sums
            zeta = f["zeta"]
            z = z / zeta
            for _ in range(3):
                z = noise(z, sig)                                # the iterate may be bootstrapped
                z = z * (2 - zeta * S * z)
            z = z * zeta
        y = s * z * (1.0 if last else math.sqrt(amp))
    stats["sums_outside"] = low
    return y.to(x.dtype)


def he_layernorm_wide(x, weight, bias, params: dict, num: HENumerics, gen, stats: dict, eps: float = 1e-12):
    """The wide-range LayerNorm (native/openfhe/NONLINEAR.md): an initial 1/sqrt from the bootstrapped
    variances, Newton steps on the exact ones, the result refreshed before it multiplies the rows."""
    from ..kernel_params import layernorm_wide_constants
    f = layernorm_wide_constants()["fits"][repr(params["e0"])]
    vhi, e0, s, steps = params["vhi"], params["e0"], params["s"], int(params["steps"])
    sig = num.bootstrap_sigma
    noise = lambda t: t + torch.randn(t.shape, generator=gen, dtype=torch.float64) * sig
    c = x - x.mean(-1, keepdim=True)
    var = (c * c).mean(-1, keepdim=True)
    w = ((var + eps) / vhi).double()
    lo = e0 * vhi - eps
    stats.update(input_min=float(var.min()), input_max=float(var.max()), interval=[lo, vhi], quantity="variance",
                 kernel=params.get("kernel", "wide"), outside=float(((var < lo) | (var > vhi)).double().mean()))
    t = 2 * (w - e0) / (1 - e0) - 1
    amp = 1.0 if num.bootstrap_sigma < 1e-6 else 16.0             # fast: t bootstrapped 16x larger
    y = cheb_series(t + torch.randn(t.shape, generator=gen, dtype=torch.float64) * sig / amp, f["coef"]) / s
    for _ in range(steps):
        y = noise(y)
        y = y * (1.5 - 0.5 * s * s * w * y * y)
    y = noise(y)
    out = c * (y * s / math.sqrt(vhi)).to(c.dtype)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def he_layernorm(x, weight, bias, num: HENumerics, gen, stats: dict):
    lo, hi = num.ln_var_lo, num.ln_var_hi
    mu = x.mean(-1, keepdim=True)
    c = x - mu
    var = (c * c).mean(-1, keepdim=True)
    v, ok = bootstrap(var / hi, num, gen)                            # the kernel bootstraps the variance
    inv = chebyshev(v, "inv_sqrt", lo / hi, 1.0, num.ln_degree, hi, num.ln_eps)
    stats.update(input_min=float(var.min()), input_max=float(var.max()), interval=[lo, hi],
                 outside=float(((var < lo) | (var > hi)).double().mean()), quantity="variance",
                 bootstraps=1, bootstrap_in_range=ok)
    out = c * inv
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def he_softmax(x, num: HENumerics, gen, stats: dict):
    """Softmax over the last axis (keys) as the kernel computes it."""
    b, umin, S = num.sm_bound, num.sm_umin, x.shape[-1]
    ex = chebyshev(x, "exp_shift", -b, b, num.sm_exp_degree, b)
    u, ok1 = bootstrap(ex.sum(-1, keepdim=True) / S, num, gen)        # the kernel bootstraps u
    g = chebyshev(u, "ratio", umin, 1.0, num.sm_inv_degree, umin)
    boots, ok2 = 1, True
    if num.sm_bootstrap_inverse:
        g, ok2 = bootstrap(g, num, gen)
        boots += 1
    stats.update(input_min=float(x.min()), input_max=float(x.max()), interval=[-b, b],
                 outside=float(((x < -b) | (x > b)).double().mean()),
                 u_below_umin=float((u < umin).double().mean()), bootstraps=boots, bootstrap_in_range=ok1 and ok2)
    return ex * g / (S * umin)

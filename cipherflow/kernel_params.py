"""Parameters of the HE kernels' approximations, and the value ranges they are valid on.

Shared by the planner (value contracts, checked against a Calibration) and cipherflow.runtime
(numerical models). The OpenFHE kernels (native/openfhe/he_ops.cpp) approximate:
  GeLU       Chebyshev interpolation on [-gelu_bound, gelu_bound]
  LayerNorm  1/sqrt(variance) on [ln_var_lo, ln_var_hi] (variance scaled by 1/ln_var_hi)
  Softmax    exp(x - sm_bound) on [-sm_bound, sm_bound], then umin / u on [umin, 1], where u is the
             row mean of exp(x - sm_bound)
Outside these intervals the polynomials are evaluated anyway and the result is wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

BOOTSTRAP_RANGE = 64.0     # |value| < q0 / (2 * scale) = 2^56 / 2^50: beyond it bootstrapping fails
# he_ops bootlevel (fast): the bootstrapping error stays at its |x| <= 1 level up to |x| = 32 and grows
# ~ (|x| / 32)^3 beyond; values above are bootstrapped scaled (x / s, bootstrap, * s: one more level).
BOOTSTRAP_FLAT = 32.0
BOOTSTRAP_MARGIN = 2.0      # outputs on STS-B's dev set reach 2x a 2048-sentence calibration
# A vector's mean sits in one coefficient, which bootstrapping handles much less precisely (he_ops
# bootlevel: a constant 2.9 outside one block costs 1e-2, like random values of amplitude ~90): a
# mean m counts as an amplitude of BOOTSTRAP_MEAN_WEIGHT * |m|.
BOOTSTRAP_MEAN_WEIGHT = 30.0


def bootstrap_amplitude(absmax: float, absmean: float = 0.0) -> float:
    return max(float(absmax), BOOTSTRAP_MEAN_WEIGHT * float(absmean))


def bootstrap_scale(absmax: float, flat: float = BOOTSTRAP_FLAT, margin: float = BOOTSTRAP_MARGIN) -> int:
    """Power of two s that brings margin * absmax within the flat range (1: no scaling)."""
    need = margin * float(absmax) / flat
    return 1 if need <= 1 else 2 ** math.ceil(math.log2(need))


@dataclass(frozen=True)
class HENumerics:
    name: str
    gelu_bound: float = 8.0
    gelu_degree: int = 27
    ln_var_lo: float = 0.25
    ln_var_hi: float = 25.0
    ln_degree: int = 27
    ln_eps: float = 1e-5
    sm_bound: float = 6.0
    sm_exp_degree: int = 27
    sm_umin: float = 0.02
    sm_inv_degree: int = 27
    sm_bootstrap_inverse: bool = False   # precise mode bootstraps umin / u once more
    bootstrap_sigma: float = 1.5e-4      # per value (values in [-1, 1])
    modeled: bool = True                 # False: the backend's own approximations are unknown


# bootstrap_sigma: the measured max |error| over 32768 random values in [-1, 1] / 4.56 (expected max of
# 32768 Gaussian samples). Development machine (OpenFHE 1.4.2): fast 6.6e-4, precise 8.4e-8.
HE_NUMERICS: Dict[str, HENumerics] = {
    "openfhe-fast": HENumerics("openfhe-fast", bootstrap_sigma=1.45e-4),
    "openfhe-precise": HENumerics("openfhe-precise", sm_exp_degree=13, sm_bootstrap_inverse=True,
                                  bootstrap_sigma=1.8e-8),
    # NEXUS's polynomials are not reproduced: its HE operators are emulated like openfhe-fast
    "nexus": HENumerics("nexus", bootstrap_sigma=1.45e-4, modeled=False),
}
MPC_FRAC_BITS: Dict[str, int] = {"sci": 12, "shaft": 16}
# Hardened MPC -> HE conversion (native/openfhe/conv_hardened): removing the share wrap in HE adds
# noise of this RMS to the fixed-point value (measured, bound_bits 12, delta 6, both modes): the CKKS
# noise times 2^(bound_bits + delta), the same on both rings.
MPC_TO_HE_SIGMA: Dict[str, float] = {"sci": 4.0e-5, "shaft": 4.0e-5}
# The hardened protocol assumes |value| < 2^CONVERSION_BOUND_BITS: above it HE -> MPC hides less and
# MPC -> HE leaves the wrap in (wrong values).
CONVERSION_BOUND_BITS = 12

# (node, calibrated statistics of the node) -> None if the kernel is valid on them, else why not
ValueContract = Callable[[object, Dict[str, float]], Optional[str]]

# Margins on calibrated ranges, for every kernel (base and wide): inputs the calibration did not see
# can exceed its ranges, and a polynomial evaluated beyond its interval diverges (STS-B: a dev
# sentence with a LayerNorm variance of 27.5 where 256 training sentences reached 18.6 turned a
# whole prediction into NaN). Measured gaps between a 2048-sentence training calibration and the
# dev sets (SST-2, MRPC, RTE, STS-B; STS-B is the worst): GeLU |x| 1.09x, LayerNorm variances 1.92x
# above / 1.12x below, Softmax scores 0.08 (above) / 0.20 (below) of the calibrated span, row maxima
# 0.01 below. GeLU: |x| x GELU_MARGIN; LayerNorm: variances / LN_MARGIN_LO and x LN_MARGIN_HI;
# Softmax: SOFTMAX_PAD_HI / SOFTMAX_PAD_LO of the span above / below the scores, SOFTMAX_PAD_ROW
# below the smallest row maximum.
GELU_MARGIN = 1.5
LN_MARGIN_LO = 1.5
LN_MARGIN_HI = 2.5
SOFTMAX_PAD_HI = 0.2
SOFTMAX_PAD_LO = 0.5
SOFTMAX_PAD_ROW = 0.125


def _range(lo, hi) -> str:
    return "unknown (not in the calibration)" if lo is None or hi is None else f"[{lo:.3g}, {hi:.3g}]"


def unknown_ranges(backend: str) -> Dict[str, ValueContract]:
    """For a backend whose approximations are not known: with a calibration, its non-linear
    operators cannot be checked, so they are kept out of HE."""
    def unknown(node, s):
        return f"the input ranges {backend}'s approximation is valid on are not known, so they cannot be checked"
    return {"GeLU": unknown, "LayerNorm": unknown, "Softmax": unknown}


def value_contracts(p: HENumerics) -> Dict[str, ValueContract]:
    """Value contracts of the HE kernels with parameters `p` (see the module docstring)."""
    b = p.gelu_bound

    def gelu(node, s):
        lo, hi = s.get("in_min"), s.get("in_max")
        if lo is None or hi is None or GELU_MARGIN * max(abs(lo), abs(hi)) > b:
            return (f"inputs in {_range(lo, hi)}, the HE GeLU is valid on [{-b:g}, {b:g}] "
                    f"(margin {GELU_MARGIN:g})")
        return None

    def layernorm(node, s):
        lo, hi = s.get("var_min"), s.get("var_max")
        if lo is None or hi is None or lo / LN_MARGIN_LO < p.ln_var_lo or hi * LN_MARGIN_HI > p.ln_var_hi:
            return (f"variances in {_range(lo, hi)}, the HE LayerNorm is valid on [{p.ln_var_lo:g}, "
                    f"{p.ln_var_hi:g}] (margins {LN_MARGIN_LO:g}, {LN_MARGIN_HI:g})")
        return None

    def softmax(node, s):
        lo, hi, lme = s.get("in_min"), s.get("in_max"), s.get("lme_min")
        B = p.sm_bound
        if lo is None or hi is None:
            return f"scores in {_range(lo, hi)}"
        span = hi - lo
        if lo - SOFTMAX_PAD_LO * span < -B or hi + SOFTMAX_PAD_HI * span > B:
            return f"scores in {_range(lo, hi)}, the HE Softmax is valid on [{-B:g}, {B:g}] (with margins)"
        if lme is None:
            return "no row statistics (log-mean-exp) in the calibration"
        if math.exp(lme - SOFTMAX_PAD_ROW * span - B) < p.sm_umin:
            return (f"row means of exp(x - {B:g}) down to {math.exp(lme - B):.3g}, the HE Softmax needs >= "
                    f"{p.sm_umin:g} (with a margin)")
        return None

    return {"GeLU": gelu, "LayerNorm": layernorm, "Softmax": softmax}


# ----------------------------------------------------------------- wide-range kernels (NONLINEAR.md)
# GeLU(x) = x h(f1(x / B)) with f1, h fitted by native/openfhe/fit_gelu_wide.py for each B; one
# bootstrap between f1 and h. A node gets the smallest B >= GELU_WIDE_MARGIN x its calibrated max |x|.
GELU_WIDE_BOUNDS = (32, 64, 128, 256)
GELU_WIDE_MARGIN = GELU_MARGIN


def gelu_wide_bound(stats) -> Optional[int]:
    """B for a node with these calibrated statistics (None: no B covers them)."""
    lo, hi = (stats or {}).get("in_min"), (stats or {}).get("in_max")
    if lo is None or hi is None:
        return None
    need = GELU_WIDE_MARGIN * max(abs(lo), abs(hi))
    return next((b for b in GELU_WIDE_BOUNDS if b >= need), None)


def gelu_wide_contract(node, s) -> Optional[str]:
    if gelu_wide_bound(s) is None:
        return (f"inputs in {_range(s.get('in_min'), s.get('in_max'))}, the wide HE GeLU covers |x| <= "
                f"{GELU_WIDE_BOUNDS[-1] / GELU_WIDE_MARGIN:g} (bound {GELU_WIDE_BOUNDS[-1]}, margin {GELU_WIDE_MARGIN:g})")
    return None


def gelu_wide_params(node, s) -> Dict:
    return {"bound": gelu_wide_bound(s)}


_GELU_WIDE = None


def gelu_wide_coefficients(bound: int) -> Dict:
    """{"f1": [...], "h": [...], "max_err": ...}: numpy Chebyshev coefficients on [-1, 1]."""
    global _GELU_WIDE
    if _GELU_WIDE is None:
        import json
        from pathlib import Path
        _GELU_WIDE = json.loads((Path(__file__).parent / "kernel_data" / "gelu_wide.json").read_text())
    return _GELU_WIDE["kernels"][str(int(bound))]


# Softmax over wide score ranges (native/openfhe/fit_softmax_wide.py): scores shifted by c (above every
# score), y0 = exp((x - c) / 2^k), then k rounds of square-and-normalise. c, the exp interval and k
# come from the calibrated scores (with margins) and the smallest row maximum; the kernel is priced
# for k <= SOFTMAX_WIDE_KMAX rounds.

SOFTMAX_WIDE_KMAX = 4
_SOFTMAX_WIDE = None


def softmax_wide_constants() -> Dict:
    global _SOFTMAX_WIDE
    if _SOFTMAX_WIDE is None:
        import json
        from pathlib import Path
        _SOFTMAX_WIDE = json.loads((Path(__file__).parent / "kernel_data" / "softmax_wide.json").read_text())
    return _SOFTMAX_WIDE


def softmax_wide_design(stats) -> Optional[Dict]:
    """{"c", "lo", "k", "W"} for a node's calibrated scores (None: not enough statistics)."""
    s = stats or {}
    lo, hi = s.get("in_min"), s.get("in_max")
    rmin = s.get("rowmax_min", s.get("lme_min"))          # lme <= the row max: a safe lower bound
    if lo is None or hi is None or rmin is None:
        return None
    span = hi - lo
    c, lo2, rmin2 = hi + SOFTMAX_PAD_HI * span, lo - SOFTMAX_PAD_LO * span, rmin - SOFTMAX_PAD_ROW * span
    k = max(1, math.ceil(math.log2(max(c - rmin2, 1e-9) / softmax_wide_constants()["target"])))
    return {"c": c, "lo": lo2, "k": k, "W": (c - lo2) / 2 ** k}


def softmax_wide_contract(node, s) -> Optional[str]:
    d = softmax_wide_design(s)
    if d is None:
        return "no score statistics in the calibration"
    if d["k"] > SOFTMAX_WIDE_KMAX:
        return (f"scores in {_range(s.get('in_min'), s.get('in_max'))} need {d['k']} rounds, the wide HE Softmax "
                f"is priced for {SOFTMAX_WIDE_KMAX}")
    return None


def softmax_wide_params(node, s) -> Dict:
    return softmax_wide_design(s)


# LayerNorm over wide variance ranges (native/openfhe/fit_layernorm_wide.py): w = (var + eps) / vhi in
# [e0, 1], an initial 1/sqrt(w) from the bootstrapped variances, Newton steps on the exact ones.
# Calibrated variances get LN_MARGIN_LO / LN_MARGIN_HI; e0 is the largest fitted value
# below vlo / vhi. "wide" (e0 >= 1e-3, at most 2 Newton steps) and "wider" (e0 >= 1e-4, 6 steps) are
# priced separately.

LN_WIDE_MIN_E0 = {"wide": 1e-3, "wider": 1e-4}
_LN_WIDE = None


def layernorm_wide_constants() -> Dict:
    global _LN_WIDE
    if _LN_WIDE is None:
        import json
        from pathlib import Path
        _LN_WIDE = json.loads((Path(__file__).parent / "kernel_data" / "layernorm_wide.json").read_text())
    return _LN_WIDE


def layernorm_wide_design(stats, min_e0: float) -> Optional[Dict]:
    """{"vlo", "vhi", "e0", "s", "steps"} for a node's calibrated variances (None: not covered)."""
    lo, hi = (stats or {}).get("var_min"), (stats or {}).get("var_max")
    if lo is None or hi is None or lo <= 0:
        return None
    vlo, vhi = lo / LN_MARGIN_LO, hi * LN_MARGIN_HI
    fits = [f for f in layernorm_wide_constants()["fits"].values() if min_e0 <= f["e0"] <= vlo / vhi]
    if not fits:
        return None
    f = max(fits, key=lambda f: f["e0"])
    return {"vlo": vlo, "vhi": vhi, "e0": f["e0"], "s": f["s"], "steps": f["steps"]}


def _ln_contract(variant):
    def contract(node, s):
        if layernorm_wide_design(s, LN_WIDE_MIN_E0[variant]) is None:
            return (f"variances in {_range((s or {}).get('var_min'), (s or {}).get('var_max'))}, the {variant} HE "
                    f"LayerNorm covers a ratio of {1 / LN_WIDE_MIN_E0[variant] / (LN_MARGIN_LO * LN_MARGIN_HI):g}")
        return None
    return contract


def _ln_params(variant):
    return lambda node, s: layernorm_wide_design(s, LN_WIDE_MIN_E0[variant])


# op type -> [(variant, value contract, kernel parameters from the node's calibrated statistics)]
WIDE_KERNELS = {"GeLU": [("wide", gelu_wide_contract, gelu_wide_params)],
                "Softmax": [("wide", softmax_wide_contract, softmax_wide_params)],
                "LayerNorm": [("wide", _ln_contract("wide"), _ln_params("wide")),
                              ("wider", _ln_contract("wider"), _ln_params("wider"))]}

# Wide-range non-linear kernels (OpenFHE): design notes

Status: GeLU, Softmax and LayerNorm implemented and measured (`he_ops gelu_wide`, `softmax_wide`,
`layernorm_wide`, 2026-10-05/06); the summary is at the end. The current kernels (`he_ops.cpp`) approximate
on fixed ranges that a fine-tuned BERT-base exceeds, so calibrated plans keep GeLU, LayerNorm
and Softmax in MPC. This note compares the published approaches and fixes the design.

## What the kernels must cover

Calibrated on textattack/bert-base-uncased-SST-2 (64 sentences; more data widens the ranges):

| operator | current kernel | per layer (calibration) |
|---|---|---|
| GeLU input | [-8, 8] | every layer exceeds it: min -15.6 to -61.3, max 6.6 to 135.4 (layers 9-10: 84, 135) |
| LayerNorm variance | [0.25, 25] | after attention: [0.024, 2.1]; after FFN: [0.7, 82], layers 9-10 up to 1559 |
| Softmax scores | [-6, 6], row mean of exp(x-6) >= 0.02 | +-10 to +-14; layer 2: [-28, 29]; layer 10: up to 19 |

What wide-range kernels can save (calibrated SST-2 plans with one contract lifted, at the current
kernels' costs, so an upper bound):

| plan | none in HE | GeLU in HE | all three in HE |
|---|---|---|---|
| openfhe-fast + sci, 10 Mbps | 56,518 s | 30,717 s | 10,988 s (5.1x) |
| openfhe-fast + sci, 100 Mbps | 12,042 s | 9,465 s | 9,414 s |
| openfhe-fast + sci, 1 Gbps | 6,326 s | 5,974 s | 5,962 s |
| openfhe-fast + shaft, 10 Mbps | 8,861 s | 7,247 s | 6,803 s |

Budget: one bootstrap of one ciphertext costs 17.6 s (fast). The MPC operator costs as much as
bootstrapping the whole HE tensor this many times (SCI; SHAFT is 6-10x cheaper):

| | tensor | 1 Gbps | 100 Mbps | 10 Mbps |
|---|---|---|---|---|
| GeLU | [128, 3072], 12 ciphertexts | 0.2 | 1.0 | 9.1 |
| LayerNorm | [128, 768], 3 | 0.7 | 1.0 | 6.7 |
| Softmax | [12, 128, 128], 6 | 0.6 | 1.3 | 10.3 |

So a kernel that bootstraps its full tensor more than about twice only pays at low bandwidth.
Per-row statistics (128 LayerNorm variances; 1536 Softmax row sums) fit in one ciphertext:
iterate and bootstrap there, not on the full tensor.

## Published approaches

| | NEXUS (reference code) | THOR (CCS'25) | Cho et al. (CCS'24) |
|---|---|---|---|
| GeLU | 2 composite signs (deg-9 f/g, 2+2) at +-3.5, deg-12 poly between | gate erf(x/sqrt2) = f2(f1(x/64)), least-squares deg 31 and 27, depth 11, no sign | - |
| GeLU range | [-5, 5] (diverges beyond 6, our plaintext check) | [-64, 64], err 5.4e-3 (our check; 5.7e-3 with our bootstrap noise) | - |
| LayerNorm | no mean subtraction, linear initial guess + Newton 4 + Goldschmidt 2 | mean, eps, gamma/beta; inverse sqrt by Goldschmidt with aSOR on var / (1.05 max), T = 4-5 | - |
| LN variance range | ~[4, 800] (ours) | per LN type: [0.15, 10], [0.2, 150], layers 9-10 [0.75, 2500] (repo) | - |
| Softmax | (1+x/128)^128, no max; inverse needs sum of exp in (0, 200) | subtract fixed midpoint, exp(x/(d1 d2))^d1, then log d2 square-and-normalize, Goldschmidt inverse + aSOR | subtract fixed upper bound; exp(x/2^k), k normalize-and-square with inverse sqrt (only the last precise) |
| Softmax range | scores in [-3, 3] already diverge (ours) | [-44, 50] (MRPC), [-70, 70] layer 2 (repo); 3 bootstraps per 256 rows | [-M, 0], M up to 256; Alg. 4: log M main-thread levels per ciphertext |
| setting | SEAL, three parameter sets | Liberate.FHE, A100, 13 levels between bootstraps | HEaaN, 9 levels, CPU single thread |
| accuracy | - | GLUE about -1% (models likely fine-tuned with the approximations) | no model accuracy |

Notes: THOR's ranges are fixed per layer type and taken from data, as calibration does here. Its
GLUE numbers probably come from approximation-aware fine-tuning (v1 of the paper says so); ours
must hold for ordinarily fine-tuned models, checked by emulation. Sources: ePrint 2024/1881 and
github.com/crypto-starlab/THOR (BSD-3); arXiv 2410.11184.

## Design

Common: kernel parameters (ranges) are chosen per node from the calibration, with a margin, and
recorded in the plan; the emulator uses the same coefficients as the kernels.

**GeLU (implemented).** GeLU(x) = x (1 + g(x)) / 2 with g = f2(f1(x / B)) ~ erf(x / sqrt 2),
f1 of degree 59 and f2 of degree 27 in Chebyshev bases on [-1, 1], fitted jointly (minimax by
reweighted least squares, |f1| <= 0.95 so f2 stays in its domain; `fit_gelu_wide.py` writes
`cipherflow/kernel_data/gelu_wide.json` and `gelu_wide_coeffs.h`). The kernel takes x / B (the
scaling folds into the producer: the previous linear layer's weights, or the MPC -> HE conversion's
output scale) and computes (x / B) (B h)(f1(x / B)), h = (1 + f2) / 2. Levels: f1 uses 6, then one
bootstrap per ciphertext, then h and the product leave 2 levels (fast) / 1 (precise). B in
{32, 64, 128, 256}, the smallest >= 1.25 x the node's calibrated max |x|
(`cipherflow.kernel_params.gelu_wide_bound`); the plan records it (`node.kernel`).

Measured (`he_ops <mode> gelu_wide:F=..,B=..`, max |error| on uniform inputs in [-B, B]):

| B | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| fast (one ciphertext) | 2.9e-4 | 4.1e-4 | 4.2e-4 | 4.5e-4 |
| precise (one ciphertext) | 3.4e-7 | 1.7e-4 | 1.1e-6 | 1.9e-5 |

(the [-8, 8] kernel: 5.6e-4 fast). Whole [128, 3072] tensor (12 ciphertexts): fast 266.6 s, of
which 240.2 s bootstrapping; precise 485.8 s. These runs were slowed by other users (their
Out_Projection controls took 25.4 / 22.5 s against the profiles' 15.0 s; bootstrapping ~20 s per
ciphertext against 17.6 s), so they overstate the cost by roughly 10-15%: re-measure on a quiet
machine. Effect on calibrated SST-2 plans (SCI): 56,642 s -> 35,365 s at 10 Mbps with the fast
backend (every GeLU in HE, 1.6x), 56,608 s -> 47,394 s with the precise one (9 of 12); none at
100 Mbps and above, and none with SHAFT, as the budget above predicts. Emulated accuracy on 64
SST-2 dev sentences equals the plaintext model's.

**Bootstrapping limits found on the way** (`he_ops <mode> bootlevel`): a ciphertext with no level
left cannot be bootstrapped (OpenFHE throws), and the error grows with the amplitude:

| max abs error | |x| <= 1 | 32 | 60 | 100 | 256 | 1100 | 256 scaled by 1/8 | 1100 by 1/32 |
|---|---|---|---|---|---|---|---|---|
| fast | 8.2e-4 | 8.3e-4 | 4.1e-3 | 1.6e-2 | 0.28 | 21 | 7.2e-3 | 2.9e-2 |
| precise | 8.1e-8 | 9.4e-8 | 1.1e-7 | 9.3e-6 | 5.2e-2 | 21 | 1.5e-6 | 6.5e-6 |

A fine-tuned BERT's residual stream reaches 1060, so the planner now needs one level left to
bootstrap and, with a calibration, bootstraps values above 32 scaled (x / s, bootstrap, * s), which
needs one more. Kernels that end with no level left (fast Softmax, precise LayerNorm) cannot be
followed by a bootstrap or a conversion; their redesigns must leave levels.

**Softmax (implemented).** softmax(x) = softmax over exp((x - c) / 2^k) raised to the 2^k, so
(Cho et al.; THOR's square-and-normalize):

1. u = affine(x) in [-1, 1] (the map folds into the producer, like GeLU's 1 / B), c above every
   calibrated score (no row max is computed); y0 = exp((x - c) / 2^k) by a degree-13 Chebyshev
   interpolant (4 levels).
2. Normalise y0 once (rough inverse P1, degree 59 on [0.04, 154]), then k rounds: s = y^2, S = row
   sum, y = s / S. Rows are kept scaled by sqrt(32) so that y and s stay in the bootstrapping's
   flat range; the next normalisation removes any constant.
3. The 1536 row sums (12 heads x 128 queries) are packed into one ciphertext, mapped to [-1, 1]
   inside the packing mask (no level), inverted there, and broadcast back. Every inverse but the
   last is rough (P2, degree 27, relative error 0.17): its error is the same for all values of a row,
   so the next normalisation removes it. The last one adds 3 Newton steps on the packed sums
   *before* any bootstrap (built directly as zeta S in the packing), so its result is exact up to
   the rows' own errors.

k is the smallest with (c - smallest calibrated row max) / 2^k <= 3, which bounds the first sums
below by e^-3; with a 25% margin on the calibrated range, SST-2's layers need k = 3 (layers 2, 7,
10: 4). The planner prices the kernel at k = 4 (`SOFTMAX_WIDE_KMAX`).

Implementation lessons (`he_ops softmax_wide:...,debug=1` prints every stage):
- Slots of the packed ciphertext that hold no sum must stay small: with the offset of the [-1, 1]
  map added everywhere, the inverse and the Newton steps grew them past 32, and because
  bootstrapping mixes all slots, the whole ciphertext came back undecodable. The offset is now
  added only where sums are.
- The inverse's level bounds the rows': refreshing the one packed ciphertext before broadcasting
  saves bootstrapping the six row ciphertexts in every round.
- Newton steps need 3 levels on both factors to leave their result bootstrappable.

Measured, whole [12, 128, 128] tensor (6 row ciphertexts + the packed one):

| | k | time | bootstraps | max abs error | levels left |
|---|---|---|---|---|---|
| fast | 4 (priced) | 482.7 s | 22 | 3.1e-3 | 6 |
| fast | 3 | 322.1 s | 15 | 1.1e-3 | 4 |
| precise | 4 (priced) | 982.3 s | 24 | 9.5e-6 | 5 |
| precise | 3 | 899.5 s | 22 | 4.2e-6 | 5 |

(inputs: rows with maxima uniform in [rmin, hi] and values uniform below; the fast error is the
fast bootstrap's noise amplified by the rounds after it). As with GeLU, other users' jobs slowed
these runs (Out_Projection controls 24.0 / 21.2 s against 15.0 s; ~20 s per fast bootstrap against
17.6 s): re-measure on a quiet machine. Against SCI's MPC Softmax (1086 s at
10 Mbps, 134 s at 100 Mbps) it pays only at low bandwidth.

Calibrated SST-2 plans (SCI, 10 Mbps; the wide kernels add nothing at 100 Mbps and above, or with
SHAFT):

| backend | neither | wide GeLU | wide Softmax | both |
|---|---|---|---|---|
| openfhe-fast | 56,625 s | 35,349 s | 49,577 s | 28,300 s (2.0x) |
| openfhe-precise | 56,593 s | 47,379 s | 55,787 s | 46,573 s (1.2x) |

With both (fast: every GeLU and Softmax in HE), emulation on 64 SST-2 dev sentences (calibrated on
256 training sentences) agrees with the plaintext model on all 64, with no value outside a
kernel's range.

**LayerNorm (implemented).** Mean subtraction and the per-token variance as in the base kernel
(H var in one ciphertext); then w = (var + eps) / vhi in [e0, 1] gives two copies, each one constant
product (no extra level): t in [-1, 1], bootstrapped, for the initial y0 = P(t) / s ~ 1 / sqrt(s^2 w)
(relative-error fit of degree 59, `fit_layernorm_wide.py`, one per e0 in 1e-1 ... 1e-4), and
0.5 s^2 w, never bootstrapped, for Newton steps y <- y (1.5 - 0.5 s^2 w y^2), which converge to the
exact inverse square root whatever the noise of y0 (1, 2, 3 or 6 steps as e0 shrinks). s keeps the
bootstrapped values in the flat range. The output is c * y, i.e. the normalised values times
sqrt(vhi) / s, which folds into gamma with the next linear layer. The kernel needs 6 levels on its
input (the exact variances keep 3 for the Newton steps). Variances come from the calibration with a
factor 1.5 on both sides; "wide" covers e0 >= 1e-3 (2 Newton steps), "wider" e0 >= 1e-4 (6 steps,
SST-2's layers 9-10 after the FFN), priced separately.

Implementation lesson: the per-token statistics are the same in every 128-slot block, and a vector
repeated like that is bootstrapped much less precisely (`he_ops bootlevel`, "repeated every 128
slots": error 7e-4 up to |x| = 4, 1.8e-3 at 8, 1.3e-2 at 18; random values stay at 7e-4 up to 32),
because its energy sits in few large coefficients. The kernel's inverse square roots (up to ~19)
lost 6% in one refresh. Only block 0 now holds them (the mask folds into the constant products),
and the refreshed result is broadcast to every block by rotations.

A second lesson, for every kernel: bootstrapping is much less precise for vectors with a large
mean, which sits in one coefficient (`bootlevel`, "block 0 in [1.5, 19], constant elsewhere": a
constant 1 costs 2.4e-6 (precise), 2.9 costs 1.2e-2, 10 costs 0.4, in both modes). The unused blocks
of the Newton iterate held P(0) / s ~ 2.9: its first bootstrap now subtracts that constant and adds
it back, and the steps' 1.5 lives in block 0 only, so the steps zero the unused blocks. After the
final refresh the unused blocks are masked again before the broadcast (their bootstrapping noise
would add up over 256 blocks). The emulator now counts a mean m as an amplitude of 30 |m|
(`BOOTSTRAP_MEAN_WEIGHT`), and calibrations record each node's largest |mean| (`out_absmean`) for
the planner's scaled bootstraps. Bootstrapping is fine at 1-7 levels left and after constant
products (`bootlevel`).

Measured ([128, 768], 3 ciphertexts; variances log-uniform inside the interval):

| variances | e0 | Newton | fast: time, bootstraps, max error | precise: time, bootstraps, max error |
|---|---|---|---|---|
| [0.016, 3.15] (attention LN) | 3e-3 | 2 | 81.5 s, 4, 3.8e-4 | 159.1 s, 4, 3.4e-8 |
| [0.47, 123] (FFN LN) | 3e-3 | 2 | 81.9 s, 4, 5.9e-4 | 158.6 s, 4, 1.2e-7 |
| [0.47, 2340] (layers 9-10) | 1e-4 | 6 | 156.1 s, 8, 1.4e-3 | 307.2 s, 8, 2.1e-7 |

(errors in units of the normalised output; the base fast kernel's: 2e-2). Controls: Out_Projection
23.5 s (fast) / 21.1 s (precise) against 15.0 s, so these costs are high too. "wide" is priced
from the dearer 2-step run, "wider" from the 6-step one.

## Summary

Calibrated SST-2 plans (calibration on 256 training sentences; GeLU / Softmax / LayerNorm nodes
in HE in brackets):

| plan | none | GeLU | Softmax | LayerNorm | all three |
|---|---|---|---|---|---|
| openfhe-fast + sci, 10 Mbps | 56,619 s | 35,343 s | 49,570 s | 49,482 s | 21,157 s (12/12/24), 2.7x |
| openfhe-precise + sci, 10 Mbps | 56,591 s | 47,377 s | 55,785 s | 52,994 s | 43,131 s (9/12/18), 1.3x |
| openfhe-fast + sci, 100 Mbps | 12,143 s | | | 12,123 s | 12,123 s (0/0/1) |
| openfhe-fast + sci, 1 Gbps | 6,427 s | | | 6,416 s | 6,416 s (0/0/1) |
| openfhe-fast + shaft, 10 Mbps | 8,861 s | | | | 8,861 s (all MPC) |

With all three, the emulated predictions on 64 SST-2 dev sentences equal the plaintext model's
(fast and precise), with no value outside a kernel's range. The kernels pay where the bootstrap
budget said they would: SCI at low bandwidth. The upper bound at the base kernels' costs was
10,988 s (5.1x); the wide kernels cost more (and their measured costs are inflated by the busy
machine: re-measure on a quiet one).

## Robustness (2026-10-06)

**Kernels on a second machine.** An EPYC node with ECC memory (OpenFHE 1.4.2.6, whose fast
bootstrapping gives 9.4 bits against the first machine's 10.5) ran GeLU, Softmax (k 3, 4) and
LayerNorm (wide, wider), fast and precise, three times each: all 42 results decode. Fast errors are
about twice the first machine's (GeLU 6.6-8.3e-4, Softmax 2e-3 / 5.7-8.1e-3, LayerNorm wide 4-10e-4),
as the bootstrapping precision predicts, except one LayerNorm-wider run at 5.9e-2: with vhi / vlo ~
1e4 the initial guess at the smallest variances came out of Newton's basin under the noisier
bootstrapping (the plaintext model with that noise agrees: e0 <= 3e-4 is unreliable). The fast
kernel now bootstraps t sixteen times larger (it uses 1/32 of the flat range otherwise) and scales
it back, one more level. The node was shared, so its timings are not usable either (Out_Projection
control 10.2-11.3 s, against a quiet 15.0 s on the first machine).

Results that fail in one process and not the next (every bootstrap of a process failing, or none)
point to the machine: faulty memory corrupts bootstrapping's tables. Check a machine with
`cipherflow selfcheck` (with `--memtest`) before trusting kernel results on it.

**Inputs the calibration did not see.** GLUE with the three wide kernels (calibrated on 256
training sentences): SST-2, MRPC, RTE at 10 Mbps equal the plaintext model on every dev example;
STS-B at 10 Mbps gave NaN on 26 of 1500 dev pairs: a polynomial evaluated beyond its interval
diverges. The first cause was a base LayerNorm kernel chosen without margin (calibrated variances
up to 18.6, dev up to 44, kernel to 25), the second wide kernels whose margins were too small.
How far dev sets go beyond a training calibration (worst node per operator type):

| calibration | LN variances above / below | GeLU |x| | Softmax scores above / below (of the span) | outputs |x| |
|---|---|---|---|---|
| SST-2, MRPC, RTE, 256 sentences | <= 1.13x / 1.17x | <= 1.05x | <= 0.11 / 0.09 | <= 1.29x |
| STS-B, 256 | 2.34x / 1.20x | 1.32x | 0.17 / 0.44 | 2.12x |
| STS-B, 2048 | 1.92x / 1.12x | 1.09x | 0.08 / 0.20 | 2.04x |

Margins are now set for STS-B with 2048 sentences, for base and wide kernels alike
(`cipherflow.kernel_params`): GeLU |x| x1.5; variances /1.5 and x2.5; scores +0.2 / -0.5 of the span,
row maxima -0.125; scaled bootstraps x2. `python -m cipherflow.runtime.glue` calibrates on 2048
training sentences by default. With them, the STS-B plan at 10 Mbps (SCI, fast) runs all 1500 dev
pairs without a value outside a kernel's range: Pearson 0.880460 against the plaintext model's
0.880462 (largest prediction difference 0.013). On SST-2 they cost a little (calibration on 2048):
23,489 s fast (12 GeLU / 11 Softmax / 20 LayerNorm in HE, against 21,157 s with the earlier
margins) and 44,310 s precise, still 2.4x and 1.3x below 56,619 s / 56,591 s; the emulated
predictions on 64 dev sentences equal the plaintext model's. Margins are statistical: a deployment
whose inputs drift further still breaks; the emulator reports such values.

Open: the planner's own bootstraps of residual-stream values assume random-like data; values with
structure (an outlier dimension the same for every token) may bootstrap worse than the model says,
to be checked with real encrypted execution of a layer.

Each kernel: `he_ops` measurement (cost, levels, error on calibrated-range inputs) -> profile
records -> emulator numerics -> calibrated GLUE emulation -> one-layer real-encryption check.

# OpenFHE 1.4.2 CKKS bootstrapping: parameter search (2026-10-03)

`native/openfhe/boot_spike.cpp`; CPU, 32 OpenMP threads (Threadripper PRO 3955WX). Full slots unless noted.
Timings are for one ciphertext; "levels" = levels available after bootstrapping.

## Works at 128-bit security (HEStd_128_classic), N = 2^16, 32768 slots, SPARSE_ENCAPSULATED secret

| levels | scale / first mod | level budget | log2 q | bootstrap | peak RAM | precision |
|---|---|---|---|---|---|---|
| 5 | 59 / 60 | {3,3} | 1241 | 15.3 s | 12.0 GB | 21 bits |
| 7 | 50 / 57 | {3,3} | 1158 | 17.9 s | 13.0 GB | 13 bits |
| 7 | 50 / 57 | {4,4} | 1258 | 14.6 s | 14.6 GB | 13 bits |
| 9 | 50 / 57 | {3,3} | 1258 | 21.3 s | 14.4 GB | 13 bits |
| 9 | 45 / 52 | {3,3} | 1133 | 20.2 s | 14.0 GB | 9 bits |

Does not fit 128-bit at N = 2^16: 11 levels at scale 50; 7 levels at scale 55.

## Does not work

- N = 2^16, uniform ternary secret: no configuration with bootstrapping fits 128-bit security
  (works only without the security check: 13 bits, 26.2 s).
- N = 2^17, uniform ternary, full or 2^14-2^15 slots: decryption fails (< 5 bits). The bootstrap's
  modular-reduction range is K = 512 for composite degree 1; at N = 2^17 a uniform secret's
  |I| reaches ~6 sigma = ~510, so coefficients fall outside the range. 8 sparse slots work (29.8 s, 18 bits).
  Sparse / encapsulated secret, FLEXIBLEAUTOEXT, larger correction factors and 2 iterations also failed at 2^17.
- N = 2^17 composite scaling: 80-98-bit scales either fail encoding ("scaling factor too small",
  100+ GB RAM) or exceed the 128-bit modulus budget.

## For comparison

NEXUS CPU bootstrapping demo: N = 2^15, 8192 sparse slots, ~1430-bit modulus (128-bit security allows
~881 bits at 2^15, so these parameters are not secure), 51.6 s per ciphertext.

## Precision tuning at the recommended point (N = 2^16, SPARSE_ENCAPSULATED, scale 50 / first 57, budget {3,3})

| variant | levels | bootstrap | max abs error | precision |
|---|---|---|---|---|
| one pass (default correction factor) | 9 | 21.5 s | 6.8e-4 | 13 bits |
| two passes (iterative), precision hint 13 | 8 | 39.1 s | 7.7e-8 | 26 bits |
| two passes, precision hint 10 | 8 | 40.6 s | 7.0e-7 | 23 bits |
| one pass, correction factor 9 / 11 | 9 | ~20 s | 2.7e-3 / 9.8e-3 | 11 / 9 bits |
| one pass, first mod 60, correction 10 | 9 | 21.3 s | 5.0e-3 | 10 bits |

For reference (published): HEaaN FGb 9 levels / ~22 bits / 17 s; Lattigo N16QP1546H192H32 9 levels / 26.6 bits;
NEXUS (paper) 21 levels with bootstrapping error 4.32e-5 (~14.5 bits).

# HE <-> MPC conversion (hardened), OpenFHE backend

`conv_hardened.cpp` implements and measures the conversion between CKKS ciphertexts (server holds
the ciphertext, client holds the secret key) and additive shares in Z_2^ell (SCI: ell = 37 with 12
fractional bits; SHAFT: ell = 64 with 16). Parameters: those of `he_ops.cpp` (N = 2^16, scale 2^50,
first modulus 57 bits).

```
conv_hardened <numel> [repeats] [ell] [frac] [levels] [bound_bits] [kappa] [log2_values] [delta] [degree]
defaults:                  2        37    12     8         12         40        22          6       31
```

## HE -> MPC

Server, holding `ct` of values x (|x| < 2^b, b = `bound_bits`):

1. Keep the towers the masked values need (`Compress`): log2 Q >= M + log2(scale) + 3 with
   M = b + log2_values + kappa. With the defaults M = 74 and 3 towers are sent, so the ciphertext
   must have at least **2 levels left** (the planner enforces this; it bootstraps first otherwise).
2. Draw, for every complex slot, a mask r: real part R / 2^frac + o with R uniform in [0, 2^(M+frac))
   and o uniform in [-1/2, 1/2) / 2^frac; imaginary part uniform in [-2^M, 2^M). Encode it exactly
   (quadruple-precision FFT, integer coefficients) and add it to the ciphertext.
3. Re-randomise: add a fresh encryption of zero. Send. Its share: -R mod 2^ell.

Client: decrypt exactly (c0 + c1 s over the kept towers, CRT to big integers, centred) and decode
in quadruple precision (doubles hold 53 bits; the masked values need M + frac). Its share:
round(2^frac * Re(slot)) mod 2^ell. The shares sum to round(2^frac (x + e) + 2^frac o): within one
fixed-point step of x.

**What the client learns.** All N/2 complex slots, i.e. the whole decrypted polynomial, are
masked: per value, the client sees value + noise + mask with a mask 2^(log2_values + kappa) times
wider than |value + noise|, so its view over 2^log2_values values is within statistical distance
2^-kappa of a view independent of the values and of the noise. This hides x and the computation's
noise (which depends on the server's weights; IND-CPA^D-style leakage). The rounding offset o
hides the low bits of the noise. The fresh encryption of zero hides the ciphertext's other
component. **Assumption:** |x + noise| < 2^b; values above the bound weaken the hiding (calibration,
`cipherflow.runtime.calibrate`, records the actual ranges).

## MPC -> HE

Shares of X = round(2^frac x) are uniform in Z_2^ell, so encrypting one and adding the other in HE
gives X + w 2^ell with a wrap w that is 1 about half the time (the previous protocol ignored it).

1. Both parties reduce their shares modulo 2^ell' with ell' = b + frac + delta (local, exact).
2. Client: encrypts c' / 2^ell' (in [0, 1)) with L + `wrap_levels` levels and sends it.
3. Server: adds s' / 2^ell'. Now t = X / 2^ell' + w, w in {0, 1, 2}, |X / 2^ell'| < 2^-delta.
   It computes y = sin(2 pi t) (Chebyshev interpolation of degree `degree` on [0, 2]) and
   g = arcsin(y) / (2 pi) by the series y + y^3/6 + 3 y^5/40 + 5 y^7/112 (exact near the integers
   up to (2 pi 2^-delta)^9), multiplied by 2^(ell' - frac): x, at L levels.

Nothing is revealed: the client only encrypts, the server only computes on ciphertexts.
**Precision:** the CKKS noise of the evaluation (about 2^-30) is multiplied by 2^(b + delta); with
the defaults, noise RMS 4.0e-5 on both rings (max ~3e-4 over 10^5 values), below the HE kernels'
own errors (LayerNorm ~2e-2) but a few steps of SHAFT's fixed point. The series' truncation adds
~1e-6. `cipherflow.runtime` emulates the noise (`MPC_TO_HE_SIGMA`).

## The bound b

Both directions assume |x| < 2^b. Calibrated on SST-2 (64 sentences), a fine-tuned BERT-base
reaches 1060 in its residual stream (layer 10's FFN output and residual add; layer 9: 665), so
b = 10 is too small; b = 12 is the default. Raising b does not change the costs (M = 74 still fits
in 3 towers; the wrap removal keeps its depth) but the MPC -> HE noise doubles with each bit of
b + delta (17 with b = 10, delta = 7; 18 with b = 12, delta = 6). `compile(..., calibration=...)` warns when calibrated values crossing a conversion
reach 2^b (the bound is read from the profile's conversion records), and the emulator reports such
conversions (`conversion_in_bound: false`). A first set of measurements with b = 10, delta = 7
(`raw/2026-10-05/bound10/`) has the same costs and half the noise.

## Costs (per ciphertext of 32768 values)

| | previous (slot mask, not secure) | hardened |
|---|---|---|
| HE -> MPC | 1 tower, ~1.05 MB, ~0.15 s | 3 towers, 3.15 MB, server ~0.15 s + client ~0.28 s (quad decoding); needs 2 levels left |
| MPC -> HE | L + 1 towers, ~9 MB, ~0.15 s | L + 13 towers, 22.0 MB (fast) / 21.0 MB (precise), client ~0.2 s, server ~2.8 s (fast) / ~2.5 s (precise) for the wrap removal |

The measured values are in `compiler/state_expanded/profiles/raw/2026-10-05/` and in the
`cpu-openfhe-*` profiles (conversion records with `"protocol": "hardened"`).

## Not covered

- The NEXUS profiles' conversions (SEAL, `native/conversion`) still use the slot mask; plans that
  use them warn that conversion costs are underestimated.
- Malicious parties (the protocols are for semi-honest server and client).
- The bound b is a parameter, not checked inside the protocol (see above for the checks).

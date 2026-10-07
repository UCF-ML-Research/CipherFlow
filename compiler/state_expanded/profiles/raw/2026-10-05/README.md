# Raw measurements, 2026-10-05: hardened HE<->MPC conversion (OpenFHE)

Machine and OpenFHE parameters as in `../2026-10-04/` (Threadripper PRO 3955WX; OpenFHE 1.4.2, CKKS,
N = 2^16, scale 2^50, first modulus 57 bits). The protocol is described in
`native/openfhe/CONVERSION.md`. Each configuration ran as one process with load < 4 at its start;
every run checks its results against the plaintext values (`max_abs_err`).

| file | how |
|---|---|
| `openfhe_conversion_hardened_{sci,shaft}_{fast,precise}.jsonl` | `native/openfhe/build/conv_hardened <numel> 2 <37 12 / 64 16> <8 / 7> 12 40 22 6 31` for numel 98304, 196608, 294912, 393216 |
| `bound10/` | the same with bound_bits 10, delta 7 (superseded, see below); no `noise_rms` field |

Fields: `server_ms`, `client_ms` (best of 2 repeats, all ciphertexts), `bytes` (serialized
ciphertexts sent), `max_abs_err` (against the real values; for MPC -> HE it includes the
fixed-point rounding of the shares, 2^-13 on SCI's ring); HE -> MPC: `towers_sent`,
`min_levels_left` (levels the ciphertext needs), `mask_bits`, `kappa`, `bound_bits`, `log2_values`;
MPC -> HE: `noise_rms` (against the fixed-point value the shares hold: the conversion's own noise),
`levels_left` (of the result), `wrap_levels`, `reduced_ring_bits`, `delta`, `degree`.

Per ciphertext of 32768 values (means over the four sizes; error: largest):

| MPC | mode | direction | server s | client s | MB | max abs err | noise RMS |
|---|---|---|---|---|---|---|---|
| sci | fast | HE -> MPC | 0.15 | 0.28 | 3.15 | 2.4e-4 | |
| sci | fast | MPC -> HE | 2.80 | 0.20 | 22.03 | 4.5e-4 | 4.0e-5 |
| sci | precise | HE -> MPC | 0.16 | 0.28 | 3.15 | 2.4e-4 | |
| sci | precise | MPC -> HE | 2.57 | 0.19 | 20.98 | 3.6e-4 | 4.0e-5 |
| shaft | fast | HE -> MPC | 0.15 | 0.28 | 3.15 | 1.5e-5 | |
| shaft | fast | MPC -> HE | 2.77 | 0.20 | 22.03 | 3.7e-4 | 4.0e-5 |
| shaft | precise | HE -> MPC | 0.15 | 0.28 | 3.15 | 1.5e-5 | |
| shaft | precise | MPC -> HE | 2.50 | 0.19 | 20.98 | 2.9e-4 | 4.0e-5 |

HE -> MPC errors are one fixed-point step (2^-12, 2^-16): the client rounds value + noise + a
mask offset within half a step. The MPC -> HE noise is the same on both rings: the CKKS noise of
the wrap removal times 2^(bound_bits + delta) = 2^18.

## Bound 10 -> 12

The first set (`bound10/`, bound_bits 10, delta 7) has the same costs and half the noise (RMS
2.0e-5, measured separately with the `noise_rms` field added afterwards). Calibrating a fine-tuned
BERT-base (textattack/bert-base-uncased-SST-2, 64 training sentences) showed values up to 1060 in
its residual stream (layer 10's FFN output 1032 and the residual add after it), above 2^10, so the
default became bound_bits 12 with delta 6 (M = 74 still fits in 3 towers; the wrap removal keeps
its depth). With these profiles, the calibrated SST-2 plans have no conversion above the bound and
agree with the plaintext model on 64 dev sentences (`python -m cipherflow.runtime.glue`).

## Wide-range GeLU and bootstrapping limits (`native/openfhe/NONLINEAR.md`)

| file | how |
|---|---|
| `openfhe_ops_{fast,precise}_gelu_wide.jsonl` | `native/openfhe/build/he_ops <mode> gelu_wide:F=3072,B=64 out_proj` (Out_Projection: control) |
| `openfhe_bootlevel_{fast,precise}.jsonl` | `he_ops <mode> bootlevel`: bootstrapping at 0/1/2 levels left, at amplitudes 32-1100, and scaled (x / s, bootstrap, * s) |

The profiles price GeLU.wide from the first file (`build_measured.py`: he_ops op `GeLU_wide` ->
profile op type `GeLU.wide`, levels needed 6 + 1, output level 2 / 1). Both runs started at load
< 3 but other users' jobs started during them (load 15 at the end): the Out_Projection controls
took 25.4 s (fast) and 22.5 s (precise) against 15.0 s in the profiles, and bootstrapping ~20.0 s
(fast) per ciphertext against 17.6 s. The GeLU.wide costs (fast 266.6 s, precise 485.8 s) are
therefore ~10-15% too high; an earlier run under the same conditions took 270.5 / 489.1 s. The cost
does not depend on the bound; per-bound errors (one ciphertext): fast 2.9e-4 (B 32), 4.1e-4
(64), 4.2e-4 (128), 4.5e-4 (256); precise 3.4e-7, 1.7e-4, 1.1e-6, 1.9e-5.

bootlevel: at 0 levels left EvalBootstrap throws (DropLastElement); at 1 or 2 it works. The error is
flat up to |x| = 32 (fast; precise up to ~60) and grows ~cubically beyond (fast: 4.1e-3 at 60,
1.6e-2 at 100, 0.28 at 256, 21 at 1100); scaled bootstrapping keeps it at s times the base error.
The OpenFHE profiles' bootstrap records carry `min_level_in` 1 and `flat_range` 32 from this.

## Wide-range Softmax (`native/openfhe/NONLINEAR.md`)

| file | how |
|---|---|
| `openfhe_ops_{fast,precise}_softmax_wide.jsonl` | `he_ops <mode> softmax_wide:heads=12,lo=-28.2,hi=28.6,rmin=0 softmax_wide:heads=12,lo=-10.1,hi=11.2,rmin=0 out_proj` |

The first line (k = 4 rounds, scores as in BERT-base layer 2) prices `Softmax.wide`
(`build_measured.py`: `SOFTMAX_WIDE_PRICED`), the second (k = 3) is for reference. fast: 482.7 /
322.1 s, 22 / 15 bootstraps, max error 3.1e-3 / 1.1e-3; precise: 982.3 / 899.5 s, 24 / 22
bootstraps, 9.5e-6 / 4.2e-6. Runs started at load < 6, but the Out_Projection controls (24.0 s
fast, 21.2 s precise, against 15.0 s) show the machine was busy: the costs are too high by ~15%
(bootstraps) to ~60% (the rest).

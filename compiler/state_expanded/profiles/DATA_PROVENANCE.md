# Data provenance for `profile_{cpu,gpu}_resnet20.json`

These profiles are **synthetic** — they are NOT measured runs of any
ResNet-20 inference. They are scaled estimates derived from the
measured BERT-base profile (`profile_{cpu,gpu}_full_model.json`,
which is real BOLT/SCI MPC + real NEXUS CKKS HE) so that the
state-expanded compiler can be evaluated on a non-Transformer
topology in a placement-fair way.

The goal of the ResNet-20 evaluation is to demonstrate that
**BlockFold + state-expanded planning generalises structurally to
CNN topologies** — the residual-block SESE shape, the per-edge
domain choice, and the level-budget reasoning all carry over. The
goal is **not** to claim absolute wallclock numbers competitive with
published ResNet HE/MPC papers (which use specialised primitives we
do not model here).

If the paper reviewers ask for measured ResNet wallclock, this
profile would be replaced with a real one built by the same
methodology used for the BERT BOLT profile in
`compiler/state_expanded/profiler_*.py`.

## Reference shapes used for scaling

All scalings reference the BERT-base full-model profile at
`(B=1, S=128, H=768, FFN=3072)`:

| Reference | Shape | Cost | Source |
|---|---|---|---|
| HE Conv2D ref | 16x32x32 (16 in, 16 out, k=3) | 3000 ms (CPU) | hand-set, ~10 GFLOPs effective on NEXUS CKKS |
| HE ReLU ref | up to 16384 elements | 5000 ms (CPU) | hand-set, degree-6 polynomial CKKS |
| MPC matmul ref | `FFN_Linear_1@MPC` row in BERT profile | 146870 ms / 8.65 GB / 200 rounds for 16x768 -> 16x3072 | measured BOLT/SCI |
| MPC GeLU ref | `GeLU@MPC` row in BERT profile | 1325.18 ms / 315 MB / 408 rounds for 16x3072 = 49152 elements | measured BOLT/SCI |
| Conversion ref | `[1,128,768]` HE<->MPC | 10.5 ms / 39 MB / 1 round (CPU) | derived in BERT conversion table |
| Bootstrap | `nexus_ckks_bootstrap` | 51608.5 ms (CPU) / 1685 ms (GPU) | reused from BERT profile |

## Per-operator derivations

### Conv2D@HE
- `local_compute_ms = 3000 ms * (FLOPs / FLOPs_ref) / gpu_speedup`
- FLOPs proxy = `out_ch * out_h * out_w * in_ch * 9` (3x3 kernel).
  All 1x1 skip projections are conservatively over-counted as 3x3
  in the lookup since they share `(in_shape, out_shape)` with the
  stride-2 main conv they pair with — this gives a slightly
  pessimistic HE cost for the projections, which the planner can
  still place anywhere.
- `comm_bytes = 0`, `comm_rounds = 0`.
- `he_level_delta = 1` (one mult level).
- **Feasibility**: `feasible = (max(in_ch, out_ch) <= 64) AND
  (max(in_h, out_h) <= 32)`. Mirrors the NEXUS-style restriction
  documented in `CLAUDE.md` (small channel/seq packing only). Every
  Conv2D in ResNet-20 satisfies this.

### Conv2D@MPC
- Lowered to im2col matmul (the BOLT pattern reused for FFN_Linear_2).
- `ratio = FLOPs / (16 * 3072 * 768)` against the FFN1 reference.
- `local_compute_ms = 146870 * ratio / gpu_speedup`.
- `comm_bytes = 8.65e9 * ratio`.
- `comm_rounds = max(1, 200 * sqrt(ratio))` clamped at 50; rounds
  are structural in the protocol, but the smaller convs need fewer
  passes than a full 16x3072 matmul.
- `he_level_delta = 0`.

### ReLU@HE
- `local_compute_ms = 5000 / gpu_speedup` for any input volume <= 16384.
- `comm_bytes = 0`, `comm_rounds = 0`.
- `he_level_delta = 4` (degree-6 polynomial multiplicative depth).
- **Feasibility**: `feasible = (input_volume <= 16384)`. ResNet-20's
  largest map is `1*16*32*32 = 16384` exactly, so all ReLUs are
  feasible. Anything larger is marked infeasible to force MPC.

### ReLU@MPC
- DReLU protocol: `local = 0.0027 ms/elem` (~1/10 BOLT GeLU per-elem,
  consistent with the simpler comparison primitive ReLU uses vs.
  GeLU's polynomial+sign).
- `comm_bytes ≈ 641 B/elem` (~1/10 BOLT GeLU per-elem).
- `comm_rounds = 4` (DReLU has fixed depth).
- `he_level_delta = 0`.

### Residual_Add
- HE: `local = 0.05 ms` constant (matches BERT profile ciphertext-add).
- MPC: `local = 0`, no comm (share-only addition).

### AvgPool
- HE: `local = 10 ms / gpu_speedup`, no comm. Rotation-and-sum tree
  estimate.
- MPC: `local = 5 ms / gpu_speedup`, no comm. Share sum + truncation.

### FC (64 -> 10)
- HE: `local = 1 ms / gpu_speedup`, single packed dot product.
- MPC: `local = 0.5 ms / gpu_speedup`, 5 KB / 1 round (tiny matmul).

## Conversion records

For each tensor shape that actually appears as a graph edge in
ResNet-20 (`(1,16,32,32)`, `(1,32,16,16)`, `(1,64,8,8)`,
`(1,64,1,1)`):

```
local_compute_ms = 10.5 * (numel / 98304) / gpu_speedup_he
comm_bytes       = 39321600 * (numel / 98304)
comm_rounds      = 1
```

Both directions (`HE -> MPC` and `MPC -> HE`) use the same numbers
since the BERT reference shows them roughly symmetric.

## GPU multipliers

We adopt the same form as the BERT GPU profile:

- HE local compute: divide CPU value by **50** (NEXUS-CUDA published
  speedup is 30-60x; we use 50x as a conservative midpoint).
- MPC local compute: divide CPU value by **5** (matches the
  SHAFT/CrypTen GPU profile's typical observation).
- `comm_bytes`, `comm_rounds`: **unchanged** (the network is the
  bottleneck, not the kernel).
- Bootstrap: GPU value (`1685 ms`) reused verbatim from the BERT GPU
  profile.

## What this profile does NOT model

- Real cryptographic noise budget for arbitrary CKKS parameter
  choices (we use a single scalar `he_level_delta` per op type).
- Exact BOLT-style packing optimisations for small spatial
  dimensions (the synthetic Conv2D@MPC rounds estimate is coarse).
- Bootstrapping under the real per-op level cost (we share one
  global bootstrap record).
- 1x1 vs 3x3 distinction for the skip projections (counted as 3x3
  for HE, conservatively).

These omissions are deliberate to keep the profile auditable and to
isolate the planner's structural behaviour from primitive-level
detail. The BERT-base profile is the calibration reference; if those
real numbers are accepted, the scaling rules above produce
ResNet-20 numbers within roughly the same fidelity envelope.

## Reproduction

To regenerate both profiles from this provenance:

```bash
python -m compiler.state_expanded.profiles.build_resnet20_profile
```

Then run the sweep:

```bash
python -m experiments.paper_revision.run_resnet20_sweep
```

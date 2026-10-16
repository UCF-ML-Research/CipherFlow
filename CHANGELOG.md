# Changelog

## Unreleased

- The README and roadmap figures are single SVG files (`docs/assets/cipherflow-ecosystem.svg`,
  `cipherflow-roadmap.svg`) that follow the viewer's light or dark color scheme, on their own background,
  so they read the same anywhere they are opened; `*-light.svg` stay light for the white website.
- Roadmap figure (`docs/assets/cipherflow-roadmap.svg`, `make_roadmap_svg.py`) in `ROADMAP.md` and on the website:
  discriminative Transformers (done), generative models (next), a private inference runtime (then); below, the
  next stage's two steps (backend operators, then the models) and a LLaMA-style decoder block marked by which
  operators exist today. `ROADMAP.md` describes the stages, with details, and section 7 follows the two steps.
- Website (`docs/index.html`, GitHub Pages; white, one light theme): the thesis as a live panel (one BERT-base block re-planned as the
  bandwidth and the hardware change, with the latency against static placements), the figure, the demo,
  quick start, backends and roadmap. The plan visualization moves to `docs/planner/`; both are built by
  `python -m demo.web.build_page`. The terminal recordings move to `docs/assets/` (built by
  `demo/make_terminal_gifs.py`).

## v0.4.0 (2026-10-08)

A Transformer written in PyTorch in, the plan operator by operator out: plans name each operator's place
in the PyTorch code, the CLI plans PyTorch files and can measure the network between the two parties,
and the demo, the README and a roadmap are organized around this input and output.

- README: the logo and subtitle with links (demo, roadmap, API); the figure shows each layer's concrete
  inputs and outputs (PyTorch model, deployment given or measured, `cf.compile`, `plan.json`, backends,
  `cipherflow.runtime`); sections regrouped (about, demo, quick start, models, backends,
  running a plan, status, layout, measuring costs). `cipherflow/README.md` drops its status table
  ([ROADMAP.md](ROADMAP.md) lists what is done and planned); its sections are `API` and `Plan format`.
- `ROADMAP.md`: what is done and what is planned, along the demo's path (model in, plan out, running it).
- Plans say where each operator is in the PyTorch code: `from_torch` records it per operator
  (`"blocks.0.attn.{q,k,v}"`, `"blocks.0.attn: softmax"`, `"blocks.0.mlp.fc1"`), and plan nodes carry
  it (`node.source`, also in `plan.json`).
- `plan.steps()` / `plan.explain(steps=True)` / `--steps`: the plan in execution order, every operator
  with its domain, CKKS levels, kernel and PyTorch location, and the conversions and bootstraps before it.
- `cipherflow compile --torch FILE.py:NAME --input-shape B,S,H [--no-qkv]` plans a PyTorch encoder
  from the command line.
- The network can be measured instead of given: `cipherflow netbench --serve` on the server (another
  terminal or machine), then `cipherflow compile ... --ip SERVER[:PORT]` (alias `--peer`) on the client
  measures the RTT (median of one-byte round trips) and the bandwidth each way (bulk TCP transfer,
  timed at the receiver), plans with the slower direction, and records the measurement in the plan
  (`deployment.network_measured`). `cipherflow netbench --ip SERVER` only measures, and prints the
  equivalent `--network` value.
- The demo is its own folder, `demo/` (was `cipherflow/demo/`, inside the package): the input models
  (`demo/transformer.py`), the demo as a script (`python -m demo.plan_pytorch_model`), and the page
  builder (`python -m demo.web.build_data`, `python -m demo.web.build_page`, which now writes
  `docs/index.html` directly); `demo/README.md` describes its input and output.
- Demo: its input is PyTorch code (`demo/transformer.py`, BERT-base and ViT-B/16, traced
  with `from_torch`), shown on the page; its output adds the plan operator by operator (per block, with
  the conversions and bootstraps before each operator) and a JSON download. The traced BERT keeps the
  first block's residual (an `Input` node, as `cf.models.bert()` with Q/K/V projections), which the
  paper's BERT graph leaves out: plans change by up to 1.8% at some settings; ViT plans are unchanged.
  The reproduce command now uses the paper's profiles (`--profile paper-<hw> --security 2pc-dealer`).

## v0.3.0 (2026-10-07)

Plans checked and emulated on real data; calibration and value contracts; a hardened HE<->MPC
conversion; wide-range GeLU, Softmax and LayerNorm kernels for OpenFHE, chosen per node.

- V is a separate edge: the Q/K/V projections feed QK^T with Q and K (`[2,B,S,H]`) and Attn*V with V
  (`[B,S,H]`), so a plan that runs QK^T and Attn*V in different domains pays for converting V
  (it appears in two-party SCI plans at 300 Mbps and above). The planner handles nested regions
  (V's region inside the attention residual) and prices each branch's conversions and bootstraps
  at the join on that branch's tensor. The built-in BERT graph starts with an `Input` node (the
  first block's residual). The paper's graphs are unchanged.
- A conversion record of the same number of values is exact for any shape.
- Plan actions record the edge they apply to (`action.edge`: producer and consumer; consumer `"*"`
  for a region's exit, which applies to the producer's output for all its consumers).
- `cipherflow.runtime` (needs PyTorch): `check_plan` follows every edge through the plan's actions
  (domains and levels); `emulate(plan, module, x)` runs a traced PyTorch model on real data as the
  plan places it, with the backends' numerical models (OpenFHE kernels' Chebyshev interpolations and
  measured bootstrapping noise; MPC fixed point; conversions), and reports the error against the
  plaintext model and the approximation ranges the data falls outside of.
- Value contracts: `cipherflow.runtime.calibrate(model, inputs)` records each operator's value
  statistics on real inputs (`cf.Calibration`, JSON); `compile(..., calibration=cal)` keeps operators
  whose inputs fall outside an HE kernel's approximation out of HE (`cipherflow.kernel_params`), and
  HE non-linear operators of backends whose approximation ranges are unknown (NEXUS). Without a
  calibration, plans that use HE GeLU / LayerNorm / Softmax carry a warning. On a fine-tuned
  BERT-base (SST-2) the OpenFHE kernels' ranges do not hold (scores [-33, 30] vs [-6, 6], LayerNorm
  variances up to 1442 vs 25, GeLU inputs [-61, 129] vs [-8, 8]): calibrated plans keep the
  non-linear operators in MPC and match the plaintext model; uncalibrated ones do not.
- `python -m cipherflow.runtime.glue`: GLUE accuracy of calibrated (and uncalibrated) plans by
  emulation (BERT-base fine-tuned per task).
- Hardened HE<->MPC conversion for the OpenFHE backend (`native/openfhe/conv_hardened.cpp`,
  `CONVERSION.md`): HE -> MPC masks every complex slot with a statistically hiding mask (kappa = 40)
  and decodes exactly in quadruple precision, which needs 2 levels left on the ciphertext (3 towers
  sent); MPC -> HE removes the share wrap in HE (the previous protocol left it: wrong values about
  half the time with uniform shares). The planner bootstraps before an HE -> MPC conversion when
  fewer levels are left (`min_levels_left` in the profile's conversion records); `check_plan` and the
  emulator follow. Profiles whose conversions are not hardened (NEXUS/SEAL) carry a warning.
  The protocol assumes |x| < 2^12 (a fine-tuned BERT-base reaches ~1060 in its residual stream);
  with a calibration, `compile` warns when values crossing a conversion reach the bound, and the
  emulator reports them. The `cpu-openfhe-*` profiles price the hardened protocol (MPC -> HE:
  ~2.8 s and 22 MB per ciphertext; HE -> MPC: ~0.43 s and 3.15 MB). Calibrated BERT-base plans on
  SCI become 6-28% slower and stay 10-80x faster than all-MPC.
- Kernel variants chosen per node from a calibration: a backend can offer further HE kernels for an
  operator (`HEBackend.variants`, priced as `"<op>.<variant>"` profile records); `compile(...,
  calibration=)` gives each node the first kernel whose range covers its calibrated inputs, and the
  plan records it (`node.kernel`, e.g. `{"kernel": "wide", "bound": 64}`), which the emulator follows.
- Wide-range GeLU for the OpenFHE backend (`native/openfhe/NONLINEAR.md`): GeLU(x) = x h(f1(x / B))
  with f1 (degree 59) and h (degree 27) fitted jointly (`native/openfhe/fit_gelu_wide.py`), one
  bootstrap per ciphertext between them, B in {32, 64, 128, 256} per node. Max error 2.9-4.9e-4
  (fast) on [-B, B], same as the [-8, 8] kernel. A fine-tuned BERT-base's GeLU inputs ([-61, 135])
  now fit; at 10 Mbps (SCI) calibrated plans run every GeLU in HE.
- Wide-range Softmax for the OpenFHE backend (`native/openfhe/NONLINEAR.md`,
  `native/openfhe/fit_softmax_wide.py`): scores shifted by a calibrated bound c (no row max),
  exp((x - c) / 2^k), then k rounds of square-and-normalise; the 1536 row sums are inverted in one
  packed ciphertext (rough inverses in the middle rounds, Newton steps on the unbootstrapped sums in
  the last); k (3-4 on SST-2) and c per node from the calibration (which now records each Softmax
  node's smallest row maximum). Max error 1.1-3.1e-3 (fast), ~1e-5 (precise). With both wide
  kernels, calibrated SST-2 plans on SCI at 10 Mbps run every GeLU and Softmax in HE (56,625 s ->
  28,300 s, fast) and the emulated predictions equal the plaintext model's.
- Wide-range LayerNorm for the OpenFHE backend (`native/openfhe/fit_layernorm_wide.py`): an initial
  1/sqrt from the bootstrapped variances, Newton steps on the exact ones; two variants per node from
  the calibration ("wide": variance ratios to ~400, 2 steps; "wider": to ~4400, 6 steps). Max error
  4e-4-1.4e-3 (fast; the base kernel: 2e-2), ~1e-7 (precise). With the three wide kernels,
  calibrated SST-2 plans on SCI at 10 Mbps run every GeLU, Softmax and LayerNorm in HE (56,619 s ->
  21,157 s, fast; 56,591 s -> 43,131 s, precise) and the emulated predictions equal the plaintext
  model's.
- Margins on every kernel's calibrated ranges, set from measured gaps between training
  calibrations and GLUE dev sets (STS-B is the worst: LayerNorm variances 1.9x above a 2048-sentence
  calibration): GeLU |x| x1.5, LayerNorm variances /1.5 and x2.5, Softmax scores +0.2 / -0.5 of
  their span, scaled bootstraps x2. Base kernels had none, which turned 26 of STS-B's 1500 dev
  predictions into NaN at 10 Mbps; `cipherflow.runtime.glue` now calibrates on 2048 sentences. With
  the margins, STS-B's plan runs every dev pair within range (Pearson equal to the plaintext
  model's); SST-2's fast plan at 10 Mbps costs 23,489 s (21,157 s before).
- The fast wide LayerNorm bootstraps its normalised variances 16x larger (more of the flat range), so
  that variance ratios near 1e4 converge on a 9.4-bit build too.
- Bootstrapping is much less precise for vectors with a large mean or a structure repeated in every
  128-slot block (`he_ops bootlevel`); the emulator counts a mean m as an amplitude of 30 |m|, and
  calibrations record each node's largest |mean| (`out_absmean`) for the scaled bootstraps.
- Bootstrapping limits (`he_ops bootlevel`): OpenFHE cannot bootstrap a ciphertext with no level
  left, and its error grows ~cubically above |x| = 32 (wrong beyond ~100). The OpenFHE profiles say
  so (`min_level_in`, `flat_range`): the planner, static baselines and `check_plan` no longer
  bootstrap at level 0 and, with a calibration, bootstrap larger values scaled (x / s, bootstrap,
  * s; one more level). The emulator models the bootstrapping error accordingly (flat, then cubic;
  s times larger when scaled). Kernels that end with no level left (fast Softmax, precise
  LayerNorm) can therefore not be followed by a bootstrap or a conversion; plans avoid them.
- With `he="auto"`, plans that convert with an unhardened protocol (NEXUS profiles) rank after all
  others (they were chosen on underpriced conversions); plans of equal cost go to the backend pair
  with fewer warnings.
- Static baselines (`compile(..., safe=True)`) name the edge of every conversion and bootstrap,
  bootstrap HE values before converting them when the conversion needs more levels, and place the
  model input in its consumers' domain; the static policies place `Linear_QKV` with the other
  linear operators.

## v0.2.0 (2026-10-04)

Planning with several HE and MPC backends, on costs measured end to end.

**Backends**
- Several MPC backends: `mpc="sci"` (SCI/BOLT, two parties) and `mpc="shaft"` (SHAFT/CrypTen, two
  parties + trusted dealer), chosen by `security=` (`"2pc"` default, `"2pc-dealer"`, `"any"`);
  `cf.match` and `cipherflow backends` report which backend pairs fit a model and why.
- Several HE backends: `he="openfhe-fast"` / `"openfhe-precise"` (OpenFHE CKKS, N=2^16, 128-bit,
  one- or two-pass bootstrapping, all operators measured on one parameter set; `native/openfhe`)
  and `he="nexus"` (NEXUS reference kernels scaled to each shape). `min_precision=` filters by
  bootstrapping precision. Plans record `he_backend`, `mpc_backend`, `security`, `precision_bits`
  and the alternatives considered.
- Bootstrapping priced per ciphertext; methods that bootstrap internally set their output level.
- HE<->MPC conversions measured with real encryption (SEAL for `nexus`, OpenFHE for `openfhe-*`).

**Profiles**
- Measured built-in profiles named `<hardware>-<he>-<mpc>`: `cpu-openfhe-{fast,precise}-{sci,shaft}`
  (everything measured), `cpu-nexus-{sci,shaft}`, `gpu-nexus-shaft` (measured MPC and conversions,
  reference-scaled HE). `cpu` / `gpu` select among them. The paper's profiles are `paper-cpu` /
  `paper-gpu`. Raw measurements and the builder are in `compiler/state_expanded/profiles/`.
- `profile="auto"` measures OpenFHE, SCI and SHAFT on the local CPU.

**Models**
- Q/K/V projections are an operator (`Linear_QKV`), in `cf.models.bert` / `vit` and `from_torch`;
  every measured profile prices them. v0.1 left them out (three matmuls per block). The paper's
  graphs are `cf.models.bert_paper()` / `vit_paper()` (or `qkv_projection=False`), to be planned
  with `paper-cpu` / `paper-gpu`; `cipherflow compile --model bert_paper`.

**Tools**
- `cipherflow selfcheck`: checks that a machine computes the OpenFHE kernels correctly (keys,
  repeated bootstraps, parallel vs serial bit for bit, the BERT operators in sequence, optional
  memtester), with a JSON report.

**Changed**
- `cf.models.bert()` / `vit()` now include the Q/K/V projections; plans with the paper's profiles
  need the paper's graphs (an explicit error says so).
- `profile="cpu"` / `"gpu"` in `load_profile` are the paper's profiles, as in v0.1; in `compile`,
  the hardware name selects the measured profiles.

### Known limitations
- The value tensor V is not a separate edge: when QK^T and Attn*V run in different domains, the
  conversion of V is not priced.
- HE<->MPC conversion masks in the slot domain: its costs are the protocol's, its security is not
  yet hardened (coefficient masking with noise flooding).
- Plans are not executed yet (no runtime); kernels are measured on random inputs.
- Chebyshev input ranges (GeLU, LayerNorm, Softmax) are assumed, not calibrated on real activations.
- Bootstrapping precision depends on the machine and OpenFHE build (one pass: 10.5 bits on a
  Threadripper PRO 3955WX with OpenFHE 1.4.2, 9.4 bits on an EPYC 75F3 with OpenFHE 1.4.2.6).
- OpenFHE kernels pack 128-token blocks (other sequence lengths stay in MPC); SHAFT's offline
  (dealer) phase is not costed; SCI and SHAFT count rounds differently.

## v0.1.1

Explicit limits (cost sources and `strict`, method contracts, graph checks) and the PyTorch front
end for encoder stacks.

## v0.1

The planner of the NeurIPS 2026 paper (state-expanded planning, BlockFold), the paper's profiles,
`plan.json`, CLI and interactive demo.

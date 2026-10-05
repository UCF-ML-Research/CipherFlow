# Changelog

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

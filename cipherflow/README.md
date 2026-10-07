# cipherflow

Deployment-aware planning for hybrid HE-MPC secure inference (NeurIPS 2026).
Given a model graph, a cost profile measured on the target hardware, and the
network conditions, CipherFlow decides for every operator whether it runs in
homomorphic encryption (HE) or secure multi-party computation (MPC), and where to
insert HE↔MPC conversions and bootstraps.

v0.1 is a **planner library**: it outputs a plan (`plan.json`); executing the plan
is left to the HE/MPC backends.

## Install

```bash
git clone https://github.com/UCF-ML-Research/CipherFlow.git
cd CipherFlow
pip install -e .
```

## Python

```python
import cipherflow as cf

graph = cf.models.bert(num_blocks=12, seq=128)          # or cf.models.vit() (pre-LN ViT-B/16),
                                                        # cf.models.vit_paper() (the paper's ViT graph), cf.models.from_torch(...)
dep   = cf.Deployment.of("cpu", "10Mbps/1ms")           # built-in hardware + network
plan  = cf.compile(graph, dep)                          # ~3.5 s for BERT-base
print(plan.explain())
plan.to_json("plan.json")
```

### Planning for this machine (automatic MicroBench)

```python
dep  = cf.Deployment.local("100Mbps/20ms", device="cpu")   # the target is the machine running this code
plan = cf.compile(graph, dep)                               # profile="auto": measures only what is missing
```

`profile="auto"` works out which cost entries the graph needs (each operator shape in
HE and MPC, both conversion directions for each edge shape, and bootstrapping), loads
this machine's cache, measures only the missing entries with the installed backends,
and saves them. Costs depend on shapes, not depth, so a 24-block model reuses a
12-block model's entries. HE costs for new shapes are scaled from cached NEXUS
reference kernels, so only MPC operators need a run per new shape.

- Cache: `$CIPHERFLOW_CACHE` or `~/.cache/cipherflow/<machine>-<device>/`.
- Backends: set `CIPHERFLOW_NEXUS_BUILD` and `CIPHERFLOW_SCI_BIN` if the NEXUS and
  SCI builds are not next to this repository.
- `profile="auto"` refuses a deployment that is not this machine; measure on the
  target and pass its profile file instead. Automatic measurement is wired for CPU;
  on GPU pass a measured profile file.

`cf.compile(graph, deployment, profile=..., mpc="auto", security="2pc", methods="auto", safe=True, strict=False)`:

| argument | meaning |
|---|---|
| `profile` | hardware name (`"cpu"`, `"gpu"`: its built-in profiles, one per HE/MPC backend pair), a built-in profile (`"cpu-nexus-sci"`, `"cpu-nexus-shaft"`, `"gpu-nexus-shaft"`; the paper's: `"paper-cpu"`, `"paper-gpu"`), a profile JSON path, a `LatencyProfile`, or `"auto"` (measured on this machine) |
| `he` | HE backend: `"openfhe-fast"`, `"openfhe-precise"`, `"nexus"`, or `"auto"` |
| `min_precision` | minimum HE bootstrapping precision in bits (backends without an established precision are excluded) |
| `mpc` | MPC backend: `"sci"`, `"shaft"`, or `"auto"` (plan with each one that fits, return the fastest) |
| `security` | `"2pc"` (two parties, no third party; default), `"2pc-dealer"` (also a trusted dealer for preprocessing, as in SHAFT), `"any"` |
| `methods` | `"auto"` for the built-in NEXUS-CKKS + MPC-backend methods, or a list of `cf.backends.Method` |
| `safe` | also evaluate the static placements and return the cheapest plan |
| `strict` | refuse cost entries that are not in the profile instead of scaling them (see below) |

`cf.match(graph, deployment, mpc=..., security=...)` reports which MPC backends can plan the
model there (operator coverage, contracts, security model, profile coverage) without planning.
The plan records the chosen backend (`plan.mpc_backend`, `plan.security`), each node's
`backend`, and the other backends' costs or exclusion reasons (`plan.alternatives`).

| backend | security model | devices | automatic MicroBench |
|---|---|---|---|
| `sci` (SCI/BOLT) | `2pc` | CPU | CPU |
| `shaft` (SHAFT/CrypTen) | `2pc-dealer` | CPU, GPU | not yet (use the built-in profiles) |

The paper's numbers use its own graphs and profiles: `cf.models.bert_paper()` / `vit_paper()`
(no Q/K/V projections) with `profile="paper-cpu"` or `"paper-gpu"` and `security="2pc-dealer"` (SHAFT). HE backends are listed by `cipherflow backends`; a plan records
`he_backend`, `precision_bits` and `mpc_backend`, and notes when the HE costs are reference-scaled.
Bootstrapping is priced per ciphertext when the profile's bootstrap record declares how many values
one bootstrap refreshes (`metadata["slots"]`; the paper profiles price it per call), and an HE method
that bootstraps internally declares its output level (`Method.level_out`, or `metadata["level_out"]`
in the profile).

### Models from PyTorch

```python
graph = cf.models.from_torch(encoder, torch.randn(1, 128, 768))   # forward takes hidden states [B,S,H]
```

The front end traces the module with torch.fx and maps it onto the operator set the
way the built-in graphs are written: self-attention whose Q/K/V are linear projections
of one tensor (or `nn.MultiheadAttention`) becomes the Q/K/V projections (`Linear_QKV`,
one matmul with the concatenated weights) → QKᵀ → Softmax → Attn·V, as in `cf.models.bert`
(`qkv_projection=False` leaves the projections out, as the paper's graphs and profiles do;
v0.1 always left them out, so its latencies miss three matmuls per block), linear layers become
Out-Proj / FFN-1 / FFN-2, plus LayerNorm, GeLU and residual Add. Shape-only operations
(view, transpose, split, dropout, ...) and products or sums with plaintext constants
are dropped. When the input feeds several operators it becomes an `Input` node, whose
domain in the plan says whether the client encrypts (HE) or secret-shares (MPC) it.
A hand-written ViT traces to exactly `cf.models.vit()`, and Hugging Face `BertEncoder`
to the same plan as `cf.models.bert()`. For Hugging Face models pass the encoder
(`model.encoder`), not the full model: embeddings, attention masks and task heads are
outside the operator set. Anything else raises `cf.UnsupportedModel` listing every
unsupported operation.

### What fails, and how

| situation | behaviour |
|---|---|
| a shape is not in the profile (e.g. 197 tokens with the built-in S=128 profiles) | cost scaled from the nearest profiled shape by element count; `CostExtrapolationWarning`, listed in `plan.warnings`, nodes marked `cost_source="scaled"`; `strict=True` raises `CostExtrapolationError` |
| no MPC backend satisfies `mpc`, `security`, the hardware and the profile | `cf.UnsupportedBackend`, with the reason for each backend |
| a node violates a method's contract (NEXUS: hidden 768, 12 heads × 64, packed Q/K/V) | the node is kept out of that domain (noted in `plan.warnings`); `UnsupportedOperator` if no domain can run it |
| the graph is not a chain with two-branch residual regions (several inputs/outputs, cross-attention, nested residuals, ...) | `cf.UnsupportedGraph` before planning; `cf.check_graph(graph)` runs the same check |
| an operator type has no method | `cf.UnsupportedOperator` |

## CLI

```bash
cipherflow profiles                                     # built-in profiles
cipherflow backends                                     # MPC backends and their security models
cipherflow backends --model vit --hardware cpu --security 2pc-dealer   # which backends fit
cipherflow compile --model bert --hardware cpu --network 10Mbps/1ms -o plan.json
cipherflow compile --model bert --hardware gpu --network 10Mbps/1ms --mpc shaft --security 2pc-dealer
cipherflow profile --model bert --device cpu --dry-run  # what this machine still needs to measure
cipherflow profile --model bert --device cpu            # measure it (MicroBench)
cipherflow compile --model bert --hardware local --device cpu --network 100Mbps/20ms
cipherflow explain plan.json
```

## Plan format (`cipherflow.plan/v1`)

`nodes` lists each operator with its `domain` and, for HE, `level_in` / `level_out`;
`actions` lists conversions (`src` → `dst`, tensor shape) and bootstraps, each attached
to the operator it precedes; `cost` gives total latency, its breakdown, and online
HE–MPC communication. Each node's `cost_source` is `exact` (profile record) or `scaled`
(nearest profiled shape); `warnings` lists scaled entries and contract exclusions.

## Status

| | |
|---|---|
| planner (StateExp + BlockFold), cost model, profiles, plan.json, CLI | done |
| explicit limits: cost-source report and `strict`, method contracts, graph check | done (v0.1.1) |
| several MPC backends (`mpc=`, `security=`, `cf.match`): `sci`, `shaft` | done (v0.2.0) |
| several HE backends (`he=`, `min_precision=`), per-ciphertext bootstrapping, methods that bootstrap internally | done (v0.2.0) |
| measured profiles: SCI and SHAFT two-party operators, HE<->MPC conversion with SEAL | done (v0.2.0) |
| OpenFHE HE backend: ten operators, bootstrapping and conversions on one 128-bit CKKS parameter set, measured | done (v0.2.0) |
| Q/K/V projections as an operator (`Linear_QKV`), measured on every backend; paper graphs kept as `bert_paper` / `vit_paper` | done (v0.2.0) |
| plan checks and emulation on real data (`cipherflow.runtime`: `check_plan`, `emulate`, GLUE by emulation) | done (v0.3.0) |
| calibration and value contracts (`runtime.calibrate`, `compile(..., calibration=)`), margins from measured dev-set gaps | done (v0.3.0) |
| hardened HE<->MPC conversion (OpenFHE): statistically hiding masks, exact decoding, wrap-free MPC -> HE | done (v0.3.0) |
| wide-range OpenFHE kernels (GeLU, Softmax, LayerNorm), chosen per node from the calibration | done (v0.3.0) |
| bootstrapping's level and amplitude limits in the planner, static baselines and emulator | done (v0.3.0) |
| NEXUS on one executable parameter set; hardened NEXUS (SEAL) conversion; SPU (Cheetah, 2PC) backend | planned |
| `from_torch` for encoder stacks (hand-written, `nn.MultiheadAttention`, HF `BertEncoder`/`ViTEncoder`) | done (v0.1.1) |
| automatic MicroBench (`profile="auto"`, cached, incremental) | done on CPU (NEXUS-CKKS, SCI/BOLT); GPU: pass a measured profile |
| decoder LLMs, `from_onnx`, embeddings / task heads | planned (see the roadmap) |
| encrypted execution of a plan (runtime adapter) | planned |

## Demo

`python -m cipherflow.demo.build_data` precomputes plans over a grid of models,
hardware profiles, bandwidths and RTTs; `python -m cipherflow.demo.build_page`
embeds them into `cipherflow/demo/cipherflow_demo.html`.

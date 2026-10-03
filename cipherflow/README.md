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
cd operator_execution_framework
pip install -e .
```

## Python

```python
import cipherflow as cf

graph = cf.models.bert(num_blocks=12, seq=128)          # or cf.models.vit() (pre-LN ViT-B/16),
                                                        # cf.models.vit_paper() (the paper's ViT graph), cf.models.from_torch(...)
dep   = cf.Deployment.of("gpu", "10Mbps/1ms")           # built-in hardware profile + network
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

`cf.compile(graph, deployment, profile=..., methods="auto", safe=True, strict=False)`:

| argument | meaning |
|---|---|
| `profile` | built-in name (`"cpu"`, `"gpu"`), a profile JSON path, a `LatencyProfile`, or `"auto"` (cached profile of this machine) |
| `methods` | `"auto"` for the built-in NEXUS-CKKS / SCI-SHAFT methods, or a list of `cf.backends.Method` |
| `safe` | also evaluate the static placements and return the cheapest plan |
| `strict` | refuse cost entries that are not in the profile instead of scaling them (see below) |

### Models from PyTorch

```python
graph = cf.models.from_torch(encoder, torch.randn(1, 128, 768))   # forward takes hidden states [B,S,H]
```

The front end traces the module with torch.fx and maps it onto the operator set the
way the built-in graphs are written: self-attention whose Q/K/V are linear projections
of one tensor (or `nn.MultiheadAttention`) becomes QKᵀ → Softmax → Attn·V (the Q/K/V
projections are folded into QKᵀ, as in `cf.models.bert`; the v0.1 profiles price QKᵀ
itself, so the three projection matmuls are not in the latency), linear layers become
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
| a node violates a method's contract (NEXUS: hidden 768, 12 heads × 64, packed Q/K/V) | the node is kept out of that domain (noted in `plan.warnings`); `UnsupportedOperator` if no domain can run it |
| the graph is not a chain with two-branch residual regions (several inputs/outputs, cross-attention, nested residuals, ...) | `cf.UnsupportedGraph` before planning; `cf.check_graph(graph)` runs the same check |
| an operator type has no method | `cf.UnsupportedOperator` |

## CLI

```bash
cipherflow profiles                                     # built-in profiles
cipherflow compile --model bert --hardware gpu --network 10Mbps/1ms -o plan.json
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
| `from_torch` for encoder stacks (hand-written, `nn.MultiheadAttention`, HF `BertEncoder`/`ViTEncoder`) | done (v0.1.1) |
| automatic MicroBench (`profile="auto"`, cached, incremental) | done on CPU (NEXUS-CKKS, SCI/BOLT); GPU: pass a measured profile |
| decoder LLMs, `from_onnx`, embeddings / task heads | planned (see the roadmap) |
| runtime adapter (execute a plan) | v2 |

## Demo

`python -m cipherflow.demo.build_data` precomputes plans over a grid of models,
hardware profiles, bandwidths and RTTs; `python -m cipherflow.demo.build_page`
embeds them into `cipherflow/demo/cipherflow_demo.html`.

# CipherFlow

**Deployment-aware planning for hybrid HE-MPC secure inference.**
Code for *CipherFlow: Hardware-Aware Compiler Framework for Low-Latency Hybrid Secure Inference* (NeurIPS 2026).

Given a model graph, a cost profile measured on the target hardware, and the network
conditions, CipherFlow decides for every operator whether it runs in homomorphic
encryption (HE) or secure multi-party computation (MPC), and where to insert HE↔MPC
conversions and bootstraps, minimizing end-to-end latency.

**Interactive demo:** https://hd-zhang-hd.github.io/CipherFlow/

## Where CipherFlow fits

CipherFlow is a **planning layer** between model graphs and HE/MPC backends. Primitive
compilers optimize *within* HE or *within* MPC; CipherFlow decides *which* primitive runs
each operator, where to convert between them, and how to keep CKKS ciphertext levels
feasible, using costs measured on the target deployment.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cipherflow-ecosystem-dark.svg">
  <img alt="CipherFlow sits between the model layer and the HE/MPC backend layer. Inputs: the model graph, the deployment (hardware, bandwidth, RTT), and backend methods with measured costs. Output: plan.json, consumed by backends and runtimes." src="docs/assets/cipherflow-ecosystem-light.svg" width="100%">
</picture>

| | interface | contents |
|---|---|---|
| ① input | model graph `G = (V, E)` | operator types, tensor shapes, residual branches (`cipherflow.models`, torch.fx front end) |
| ② input | deployment `D = (HW, BW, RTT)` | target hardware (CPU or GPU), network bandwidth and round-trip time |
| ③ input | backend methods + profile | per operator and domain: level consumption δ, shape contract, and measured cost `(c_loc, b, r)`; HE↔MPC conversions and bootstrapping |
| ④ output | `plan.json` (`cipherflow.plan/v1`) | each operator's domain and CKKS level; conversions and bootstraps; latency, breakdown, and online communication |

## Install

```bash
git clone https://github.com/HD-ZHANG-HD/CipherFlow.git
cd CipherFlow
pip install -e .
```

## Quick start

```python
import cipherflow as cf

graph = cf.models.bert(num_blocks=12, seq=128)
plan  = cf.compile(graph, cf.Deployment.of("gpu", "10Mbps/1ms"))
print(plan.explain())
plan.to_json("plan.json")
```

```bash
cipherflow compile --model bert --hardware gpu --network 10Mbps/1ms -o plan.json
cipherflow explain plan.json
```

See [`cipherflow/README.md`](cipherflow/README.md) for the API, the plan format, and the current status.

## Supported models (v0.1.1)

| | |
|---|---|
| **supported** | BERT-base-shaped encoders (hidden 768, 12 heads): built-in `cf.models.bert()` / `vit()`, or your own PyTorch encoder stack via `cf.models.from_torch` (hand-written blocks, `nn.MultiheadAttention`, Hugging Face `BertEncoder` / `ViTEncoder`). Costs are exact for the shapes in the profile (built-in: B=1, S=128) and for any shape measured with `profile="auto"` (CPU). |
| **estimated (warning)** | other sequence lengths with a built-in profile: costs are scaled from the nearest profiled shape; `strict=True` refuses instead |
| **kept out of HE** | hidden size ≠ 768 or heads ≠ 12: the NEXUS methods do not apply, so those operators are planned in MPC |
| **not supported (explicit error)** | decoder LLMs and generation (causal mask, KV cache, RoPE, RMSNorm, SwiGLU), CNNs, encoder-decoder models, embeddings and task heads, attention masks, ONNX |

Anything outside this set raises `UnsupportedModel`, `UnsupportedGraph` or `UnsupportedOperator`
naming the cause; CipherFlow does not plan a model with parts of it dropped.

## Repository layout

| path | contents |
|---|---|
| `cipherflow/` | public API: `compile`, `Deployment`, `Plan`, PyTorch front end, CLI, demo builder |
| `compiler/state_expanded/` | planner: state-expanded planning, BlockFold (SESE), cost model, MicroBench profilers, measured profiles |
| `ir/` | operator graph types |
| `backends/` | HE (NEXUS) adapters used by the MicroBench profilers |
| `docs/` | the interactive demo served by GitHub Pages, and the README figure (`docs/assets/`) |

Profiling HE/MPC operators requires the NEXUS, SCI (EzPC) and SHAFT builds; point
`CIPHERFLOW_NEXUS_BUILD`, `CIPHERFLOW_SCI_BIN` and `CIPHERFLOW_SHAFT_ROOT` at them.
Planning with the shipped profiles needs none of these.

## Citation

```bibtex
@inproceedings{zhang2026cipherflow,
  title     = {CipherFlow: Hardware-Aware Compiler Framework for Low-Latency Hybrid Secure Inference},
  author    = {Zhang, Hedong and Zheng, Mengxin and Lou, Qian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

# User guide

How to install CipherFlow, plan a model, and check the plan on real data. [How CipherFlow works](architecture.md)
explains the planner; the [API reference](../cipherflow/README.md#api) lists every function and argument.

- [Installation](#installation)
- [Planning a model](#planning-a-model)
- [Measuring the network between the two parties](#measuring-the-network-between-the-two-parties)
- [Reading the plan](#reading-the-plan)
- [Supported models](#supported-models)
- [Running a plan on real data](#running-a-plan-on-real-data)
- [Reproducing the paper](#reproducing-the-paper)

## Installation

From source, which also gives you the demo, the tutorials and the native backends:

```bash
git clone https://github.com/UCF-ML-Research/CipherFlow.git
cd CipherFlow
pip install -e ".[torch]"
```

Or with pip from GitHub, for the library alone:

```bash
pip install "cipherflow[torch] @ git+https://github.com/UCF-ML-Research/CipherFlow.git"
```

- CipherFlow needs Python 3.10 or later.
- PyTorch, the `[torch]` extra, is needed for `from_torch`, the demo and `cipherflow.runtime`. Leave it out to
  plan only the built-in models.
- For Hugging Face encoders, also install `"transformers<5"` (see the
  [FAQ](FAQ.md#which-versions-of-hugging-face-transformers-work)).

Planning with the shipped profiles needs none of the HE or MPC backends. To measure costs on your own machine,
see [Measuring costs on your machine](backends.md#measuring-costs-on-your-machine).

## Planning a model

### Python

```python
import torch
import cipherflow as cf
from demo.transformer import bert                      # any encoder written in PyTorch (run from the repo root)

graph = cf.models.from_torch(bert(), torch.randn(1, 128, 768))      # or cf.models.bert(num_blocks=12, seq=128)
plan  = cf.compile(graph, cf.Deployment.of("cpu", "10Mbps/1ms"))   # picks the backends that fit
print(plan.explain(steps=True))                        # the plan, operator by operator
plan.to_json("plan.json")
```

`cf.compile` plans with every pair of HE and MPC backends that fits the deployment.
[Choosing a backend pair](backends.md#choosing-a-backend-pair) explains how it chooses, and how to ask for a
pair.

### Command line

```bash
cipherflow compile --torch demo/transformer.py:bert --input-shape 1,128,768 --network 10Mbps/1ms --steps -o plan.json
cipherflow compile --model bert --hardware cpu --network 10Mbps/1ms -o plan.json   # a built-in graph
cipherflow explain plan.json --steps
cipherflow backends                                                     # HE and MPC backends
cipherflow backends --model bert --hardware cpu --security 2pc-dealer   # which backend pairs fit
```

`--torch FILE.py:NAME` also takes `package.module:NAME`. `NAME` is an `nn.Module`, or a function that builds
one. The [API reference](../cipherflow/README.md#cli) lists every command.

## Measuring the network between the two parties

```bash
cipherflow netbench --serve                                          # on the server
cipherflow compile --torch demo/transformer.py:bert --ip SERVER_IP --steps     # on the client: measure, then plan
cipherflow netbench --ip SERVER_IP                                   # or only measure (prints the --network value)
```

The client measures two things:
- the round-trip time, as the median of one-byte ping-pongs;
- the bandwidth in each direction, by bulk TCP transfer.

It plans with the slower direction and records the measurement in the plan. The server listens on TCP port
7787; to use another, start it with `--port PORT` and give the client `--ip SERVER_IP:PORT`.

## Reading the plan

`plan.explain(steps=True)` (`--steps` on the command line) lists the plan in execution order. Here is BERT-base
on a CPU server at 10 Mbps / 1 ms:

```
     #  operator               domain CKKS level PyTorch
     0  Input                  HE     8 -> 8     input x (encrypted by the client)
     1  Linear_QKV             HE     8 -> 7     blocks.0.attn.{q,k,v}
     2  Attention_QK_MatMul    HE     7 -> 3     blocks.0.attn: matmul
          convert HE -> MPC  [1, 12, 128, 128]  (#2 -> #3)  17.72 s
     3  Softmax                MPC               blocks.0.attn: softmax
          convert MPC -> HE  [1, 12, 128, 128]  (#3 -> #4)  124.15 s
     4  Attention_V_MatMul     HE     7 -> 3     blocks.0.attn: matmul
```

Each row is one operator:
- its domain, HE or MPC;
- in HE, the CKKS level before and after it;
- where it is in the PyTorch code.

The indented lines are the conversions and bootstraps that run before the operator below them, with their
cost. `plan.json` holds the same plan for other tools (see the
[plan format](../cipherflow/README.md#plan-format)).

## Supported models

| | |
|---|---|
| **supported** | BERT-base-shaped encoders (hidden 768, 12 heads): built-in `cf.models.bert()` / `vit()`, or your own PyTorch encoder stack via `cf.models.from_torch` (hand-written blocks, `nn.MultiheadAttention`, Hugging Face `BertEncoder` / `ViTEncoder`). Costs are exact for the shapes in the profile (built-in: B=1, S=128) and for any shape measured with `profile="auto"` (CPU). |
| **estimated (warning)** | other sequence lengths with a built-in profile: costs are scaled from the nearest profiled shape; `strict=True` refuses instead |
| **kept out of HE** | hidden size ≠ 768 or heads ≠ 12: the NEXUS methods do not apply, so those operators are planned in MPC |
| **not supported (explicit error)** | decoder LLMs and generation (causal mask, KV cache, RoPE, RMSNorm, SwiGLU), CNNs, encoder-decoder models, embeddings and task heads, attention masks, ONNX |

Anything outside this set raises `UnsupportedModel`, `UnsupportedGraph` or `UnsupportedOperator`, naming the
cause. CipherFlow never plans a model with parts of it dropped. Operations outside the set can be added as the
operators of an extension (see the [tutorial](tutorials/backend-primitives.md)). [ROADMAP.md](../ROADMAP.md)
lists what comes next, decoder LLMs first.

The OpenFHE kernels take one sequence of exactly 128 tokens. Other lengths, including ViT's 197 tokens, are
kept out of HE with the `openfhe-*` backends (see the
[FAQ](FAQ.md#my-input-is-not-one-sequence-of-128-tokens-what-happens)).

## Running a plan on real data

```python
import cipherflow as cf
graph = cf.models.from_torch(encoder, x)                 # x: [1, 128, 768] hidden states
plan  = cf.compile(graph, cf.Deployment.of("cpu", "10Mbps/20ms"))
res   = cf.runtime.emulate(plan, encoder, x)             # needs PyTorch
print(res.summary())     # error vs the plaintext model; plan checks; inputs outside the approximations
```

`emulate` first checks the plan: every edge's conversions and bootstraps must take the value to the consumer's
domain and level. It then runs the model in float64 with each operator in its planned domain:
- in HE, the OpenFHE kernels' polynomial approximations and measured bootstrapping noise;
- in MPC, fixed point;
- at each conversion, its rounding.

It runs at plaintext speed, so it can evaluate accuracy on datasets. It does not run the cryptography.

The HE kernels approximate GeLU, LayerNorm and Softmax on fixed input ranges, and real models exceed them.
Calibrate on real inputs and plan with the calibration, so that operators whose inputs fall outside an
approximation stay out of HE:

```python
cal  = cf.runtime.calibrate(encoder, [hidden_states_of(s) for s in training_sentences])
plan = cf.compile(graph, cf.Deployment.of("cpu", "1Gbps/40ms"), calibration=cal)
```

`python -m cipherflow.runtime.glue --tasks sst2,mrpc,rte,stsb` measures the GLUE accuracy of calibrated plans
this way.

## Reproducing the paper

```python
# the paper's graph (no Q/K/V projections) and profiles; SHAFT assumes a trusted dealer
plan = cf.compile(cf.models.bert_paper(12), cf.Deployment.of("gpu", "10Mbps/1ms"),
                  profile="paper-gpu", security="2pc-dealer")
```

The built-in graphs and `from_torch` include the Q/K/V projections as an operator (`Linear_QKV`,
[B,S,H] → packed [3,B,S,H]), priced in every measured profile. The graphs of the NeurIPS 2026 evaluation
leave them out: three 768×768 matmuls per block, 10–30% of a block's cost. To reproduce the paper, plan
`cf.models.bert_paper()` / `vit_paper()` (or `qkv_projection=False`) with the `paper-cpu` / `paper-gpu`
profiles, which have no cost for the projections.

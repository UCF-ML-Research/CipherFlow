# CipherFlow FAQ

Short answers to the questions people ask when they start using CipherFlow. Each answer links to the full
documentation.

- [The basics](#the-basics)
- [Installing](#installing)
- [Models](#models)
- [Deployments and backends](#deployments-and-backends)
- [Reading a plan](#reading-a-plan)
- [Accuracy and security](#accuracy-and-security)
- [Extending CipherFlow](#extending-cipherflow)
- [Reproducing the paper](#reproducing-the-paper)
- [Troubleshooting](#troubleshooting)

## The basics

### What does CipherFlow do?

It plans two-party private inference for a model written in PyTorch. For every operator it decides:
- whether the operator runs in homomorphic encryption (HE) or secure multi-party computation (MPC);
- where values convert between HE and MPC;
- where ciphertexts are bootstrapped.

It picks the plan with the lowest estimated latency for your hardware and network, and returns it as
`plan.json`.

### Does it run my model under encryption?

Not yet. CipherFlow plans and checks plans, and it emulates them on real data: `cf.runtime.emulate` runs each
operator's numerical model, without the cryptography. Running a plan under encryption, with the client and
the server as separate processes, is on the [roadmap](../ROADMAP.md) (section 6).

### What do the client and the server hold?

- **The client** holds the input and the secret keys.
- **The server** holds the model's weights.

HE operators run on the server, on ciphertexts the client sent. MPC operators run on secret shares, with both
parties computing together. The plan's first row says whether the client encrypts its input (HE) or
secret-shares it (MPC).

### How is CipherFlow different from HE or MPC compilers such as Orion, HEIR, SPU or EzPC?

Those compilers work inside one primitive. Orion and HEIR lower a model to HE; SPU and EzPC/CrypTFlow lower it
to an MPC protocol. They also run the result.

CipherFlow works one level above them:
- It decides which primitive each operator uses, and prices each choice with costs measured on the backends.
- A compiler's kernels can become a CipherFlow backend (see
  [Adding new backend primitives](../README.md#adding-new-backend-primitives)).

Hybrid systems such as BumbleBee, BOLT and BLB fix the split between HE and MPC by rule. CipherFlow chooses it
per operator, for each deployment.

## Installing

### How do I install it?

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
- Leave out `[torch]` if you only plan the built-in models (`cf.models.bert()`, `cf.models.vit()`).
- The website's [Get started](https://claude.ai/artifact/DZBuQxQxf73muUsyQ19ZtS#start) page gives the
  commands for each setup.

### Do I need OpenFHE, SEAL or the MPC libraries?

Not to plan. The repository ships cost profiles measured on reference machines, so planning needs only Python,
NumPy, and PyTorch for PyTorch models.

You need the backends only to:
- measure costs on your own machine (see [below](#can-i-measure-costs-on-my-own-machine));
- check a machine with `cipherflow selfcheck`.

### Which versions of Hugging Face transformers work?

transformers 4.x:
- Up to 4.53, Hugging Face encoders trace as they are.
- From 4.54, build them with eager attention.
- transformers 5 does not trace yet: torch.fx fails inside its BERT code.

```python
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

encoder = BertEncoder(BertConfig(attn_implementation="eager"))     # BERT-base
```

We tested 4.43 with PyTorch 2.0, and 4.50, 4.53, 4.54, 4.56 and 4.57 with PyTorch 2.14.

## Models

### Which models can I plan?

Encoders shaped like BERT-base, with hidden size 768 and 12 heads:
- the built-in `cf.models.bert()` and `cf.models.vit()`;
- your own encoder written in PyTorch, including `nn.MultiheadAttention` blocks;
- Hugging Face `BertEncoder` and `ViTEncoder`.

[Supported models](user-guide.md#supported-models) lists what is estimated and what is refused.

### How do I plan my own PyTorch model?

```python
import torch
import cipherflow as cf

encoder = ...                                                      # your encoder, an nn.Module
graph = cf.models.from_torch(encoder, torch.zeros(1, 128, 768))    # forward takes hidden states [B, S, H]
plan = cf.compile(graph, cf.Deployment.of("cpu", "10Mbps/1ms"))
print(plan.explain(steps=True))
```

From the command line: `cipherflow compile --torch model.py:encoder --input-shape 1,128,768 --steps`, where
`model.py` defines `encoder` (an `nn.Module`, or a function that returns one).

The forward takes hidden states. The client computes the embeddings, and any task head, in plaintext.

### Why does `from_torch` say my model is unsupported?

`cf.UnsupportedModel` names every operation outside CipherFlow's operator set. The set is self-attention with
linear Q/K/V, linear layers, LayerNorm, GeLU, Softmax, residual adds and shape-only operations.

Common causes:
- **The whole Hugging Face model instead of its encoder.** Pass `model.encoder`; embeddings and attention
  masks are outside the set.
- **`torch.nn.TransformerEncoder`.** It does not trace with torch.fx. Write the block out, as in
  [`demo/transformer.py`](../demo/transformer.py).
- **Decoders.** Causal masks, RoPE, RMSNorm, SwiGLU and KV caches are not supported yet.

To plan an operation CipherFlow does not know, register it as an operator of an extension (see the
[tutorial](tutorials/backend-primitives.md#step-2c-a-new-operator)).

### My input is not one sequence of 128 tokens. What happens?

Two things:
- **Costs are scaled.** The built-in profiles were measured for one sequence of 128 tokens. Other shapes are
  priced by scaling the nearest measured shape, with a `CostExtrapolationWarning`. `strict=True` refuses
  instead.
- **The OpenFHE kernels do not apply.** They take exactly one sequence of 128 tokens, so with the `openfhe-*`
  backends every operator runs in MPC. With SCI at low bandwidth such a plan is very slow: BERT-base's Q/K/V
  and FFN layers each send over 100 GB in MPC.

What you can do:
- Plan with `he="nexus"`, whose kernels take other sequence lengths. Its conversions are not hardened, and
  the plan warns about that.
- Measure your shapes on your own machine with `profile="auto"`.

### Why is my ViT plan on CPU all MPC?

ViT-B/16 has 197 tokens, so the OpenFHE kernels cannot run it (see the previous answer). `he="auto"` then
returns the OpenFHE pair's all-MPC plan, which is very slow at low bandwidth.

The NEXUS pair plans most of ViT in HE and is far faster. Its conversions are priced with an unhardened
protocol, though, so `"auto"` ranks it last (see
[why CipherFlow skips a cheaper pair](#why-did-cipherflow-not-pick-the-cheapest-pair-that-explain-lists)).
Pass `he="nexus"` to get that plan; it carries a warning.

### Can I plan a decoder LLM such as GPT-2 or LLaMA?

Not yet. Causal masks, RoPE, RMSNorm, SwiGLU, KV caches and generation are outside the operator set, and
`from_torch` raises an error that names them. Generative models are the next stage of the
[roadmap](../ROADMAP.md).

## Deployments and backends

### What do `"cpu"` and `"gpu"` mean?

They select the built-in cost profiles for that kind of server, one per pair of backends:
- The CPU profiles were measured on an AMD Threadripper PRO 3955WX.
- The GPU profile is for two RTX 3090s: SHAFT was measured there, and the NEXUS-CUDA costs are scaled from
  reference kernels.

`cipherflow profiles` lists them. Costs differ across machines, and the best plan can change with them, so
measure your own machine where you can.

### Which security models are supported?

Two parties and semi-honest (passive) adversaries, as in the backends' protocols:

| `security=` | parties | MPC backend |
|---|---|---|
| `"2pc"` (default) | client and server, no third party | `sci` (SCI/BOLT) |
| `"2pc-dealer"` | also a trusted dealer, who supplies Beaver triples | `shaft` (SHAFT); the dealer's offline phase is not costed |

Malicious adversaries are not covered.

### Why does a GPU deployment need `security="2pc-dealer"`?

The only GPU profile uses SHAFT, which assumes a trusted dealer. With the default `"2pc"`, `compile` raises
`UnsupportedBackend` and says so.

### How do I give or measure the network?

Give it as bandwidth / round-trip time, e.g. `"10Mbps/1ms"` or `"1Gbps/40ms"`. Or measure it between the two
machines:

```bash
cipherflow netbench --serve                         # on the server
cipherflow compile --model bert --ip SERVER_IP      # on the client: measure, then plan
```

The client measures the round-trip time and the bandwidth in each direction, and plans with the slower
direction. In Python: `cf.Deployment.of("cpu", netbench.measure("SERVER_IP").network())`, after
`from cipherflow import netbench`.

### Can I measure costs on my own machine?

On CPU, yes. `cf.compile(graph, cf.Deployment.local("100Mbps/20ms"))` (or
`cipherflow compile --hardware local`) does three things:
- works out which cost entries your model needs;
- measures the missing ones with the installed backends;
- caches them.

`cipherflow profile --model bert --dry-run` shows what is missing.

This runs the backends, so they must be built first (see
[Measuring costs on your machine](backends.md#measuring-costs-on-your-machine)):
- OpenFHE 1.4 for the `openfhe-*` kernels.
- The SCI/BOLT bridge programs for MPC (`CIPHERFLOW_SCI_BIN`). Their sources are not in this repository yet.

MicroBench does not run on GPU yet.

### Why did CipherFlow not pick the cheapest pair that `explain()` lists?

With `he="auto"` and `mpc="auto"`, `compile` plans with every pair of backends that fits and returns the
fastest plan, with one exception:
- Plans whose HE↔MPC conversions are priced with an unhardened protocol rank after all others. Today these
  are the NEXUS profiles.
- `plan.alternatives` marks such pairs with `unhardened_conversions`.
- To use one anyway, ask for it, e.g. `he="nexus"`.

## Reading a plan

### What does `8 -> 7` in the CKKS level column mean?

A CKKS ciphertext supports a fixed number of multiplications, counted in levels. The column shows the level
before and after the operator: `Linear_QKV ... 8 -> 7` uses one level.

When a value runs low on levels, the plan either bootstraps it, which refreshes its levels, or converts it to
MPC and back.

### What are the lines between operators?

The conversions and bootstraps that run before the operator below them:
- `convert HE -> MPC [shape]` turns ciphertexts into secret shares; `MPC -> HE` does the reverse.
- `bootstrap at level 1` refreshes a ciphertext.

The cost of each is at the end of its line.

### Why does Softmax run in MPC, or the whole model in HE?

The plan follows the costs:
- **Low bandwidth:** MPC's communication dominates, so plans keep more operators in HE.
- **High bandwidth:** MPC gets cheaper, and operators move to it.
- **Nonlinear operators** such as Softmax are expensive in HE (polynomial approximations and bootstraps), so
  they are usually the first to move.

The [website](https://claude.ai/artifact/DZBuQxQxf73muUsyQ19ZtS)'s live plan shows the placement change as
you move the bandwidth.

### How accurate is the estimated latency?

The estimate is the sum of measured costs: every operator, conversion and bootstrap in the plan.

The paper checks that sum against real execution. A chain of five SCI/BOLT operators measured 2.3 s on one
machine, against 2.5 s predicted (6–9% error). On shaped networks the estimate stayed conservative.

Costs measured on another machine do not transfer exactly.

### Does the latency include the offline phase?

No. It covers the online phase: operators, conversions and bootstraps. SHAFT's offline phase, the dealer's
Beaver triples, is not costed.

### What is in `plan.json`?

- each operator's domain, CKKS levels and kernel, and where it is in your PyTorch code;
- the conversions and bootstraps, each attached to the operator it precedes;
- the latency, its breakdown, and the online HE–MPC communication.

`cipherflow explain plan.json --steps` prints it. See [Plan format](../cipherflow/README.md#plan-format).

## Accuracy and security

### Will the planned model give the same results as in PyTorch?

Not exactly, for three reasons:
- HE kernels approximate GeLU, LayerNorm and Softmax with polynomials on fixed input ranges.
- MPC computes in fixed point.
- Conversions round.

Check a plan with `cf.runtime.emulate(plan, encoder, x)`, which reports the error against the plaintext model.
Then calibrate on real inputs (`cf.runtime.calibrate`) and plan with the calibration, so that operators whose
values leave an approximation's range stay out of HE. See
[Running a plan on real data](user-guide.md#running-a-plan-on-real-data).

### What does each party learn?

Both parties know the model's architecture and the plan. Beyond that, the guarantees are those of the
backends' protocols, under the security model you choose.

### Are the HE↔MPC conversions secure?

For the OpenFHE backend, conversions use a hardened protocol for values below 2^12:
- statistically hiding masks on every slot;
- exact decoding;
- an MPC → HE direction without wrap-around.

See [`native/openfhe/CONVERSION.md`](../native/openfhe/CONVERSION.md). The NEXUS profiles still price an
unhardened mask, and plans that use them warn about it.

### My machine gives wrong or unstable results. How can I check it?

`cipherflow selfcheck` runs the deterministic OpenFHE kernels in fresh processes. It checks that:
- every key works;
- every bootstrap is correct;
- the same input always gives a bit-identical result;
- threads agree with a serial run.

`--memtest` also tests the memory. See [Checking a machine](backends.md#checking-a-machine).

## Extending CipherFlow

### Can I plug in my own HE or MPC primitive?

Yes. There are three ways, depending on what you have:

| You have | Use |
|---|---|
| a kernel for one operator | `cf.Extension.primitive` |
| a library's basic operations | `cf.PrimitiveBackend` |
| a new operator | `cf.Extension.operator` |

The [tutorial](tutorials/backend-primitives.md) walks through all three.

### Does CipherFlow check my primitive's costs or security?

No. It plans with the costs and CKKS levels you declare. It does not check an extension's correctness,
precision or security, and extensions are planned, not executed. The [roadmap](../ROADMAP.md#extensions)
lists these limits.

## Reproducing the paper

### How do I reproduce the paper's numbers?

Plan the paper's graphs with the paper's profiles:

```python
plan = cf.compile(cf.models.bert_paper(12), cf.Deployment.of("gpu", "10Mbps/1ms"),
                  profile="paper-gpu", security="2pc-dealer")
```

See [Reproducing the paper](user-guide.md#reproducing-the-paper).

### Why do my numbers differ from the paper's?

- **The graphs differ.** The default graphs include the Q/K/V projections: three 768×768 matrix products per
  block, 10–30% of a block's cost. The paper's graphs leave them out.
- **The profiles differ.** The default profiles are newer measurements.

To compare with the paper, use `cf.models.bert_paper()` / `vit_paper()` with `paper-cpu` / `paper-gpu`.

## Troubleshooting

### `UnsupportedBackend: no backend pair can be used`

The message gives the reason for each pair. The usual causes:
- a GPU deployment with the default `security="2pc"` (pass `"2pc-dealer"`);
- `he=` or `mpc=` names a pair that has no profile for your hardware.

### `CostExtrapolationWarning`

A shape in your model is not in the profile, so its cost was scaled from the nearest measured shape. The plan
marks those nodes `cost_source="scaled"`, and `strict=True` turns the warning into an error.

### `torch.fx could not trace ...` or `KeyError: None` with a Hugging Face model

- Pass the encoder (`model.encoder`) with hidden states, not the whole model.
- With transformers 4.54 or later, build it with `attn_implementation="eager"`. Without it you get
  `KeyError: None`.
- transformers 5 does not trace yet.

### `cipherflow netbench --serve` fails with "Address already in use"

Another program, often an earlier `netbench --serve`, holds port 7787. Stop it, or use another port on both
sides: `--serve --port 7800` on the server, and `--ip SERVER_IP:7800` on the client.

### The client cannot reach the server

The server listens on all interfaces, on TCP port 7787. Open that port in the server's firewall, or choose
another with `--port`.

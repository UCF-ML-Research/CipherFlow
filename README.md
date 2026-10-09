<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cipherflow-logo-dark.png">
    <img alt="CipherFlow" src="docs/assets/cipherflow-logo-light.png" width="520">
  </picture>
</p>

<h3 align="center">A Hardware-Aware Compiler Framework for Efficient Secure Inference</h3>

<p align="center">
  <a href="demo"><b>Demo</b></a> |
  <a href="#quick-start"><b>Quick start</b></a> |
  <a href="cipherflow/README.md#api"><b>API</b></a> |
  <a href="cipherflow/README.md#plan-format"><b>Plan format</b></a> |
  <a href="ROADMAP.md"><b>Roadmap</b></a> |
  <a href="https://claude.ai/artifact/Jp7U6k8hTuWgWgk2PRcKZF"><b>Website</b></a>
  <!-- | <a href="PAPER_URL"><b>Paper (NeurIPS 2026)</b></a> -->
</p>

## About

**CipherFlow** is a hardware-aware compiler framework for secure inference (NeurIPS 2026). It turns a model
written in PyTorch into a plan for two-party private inference that mixes homomorphic encryption (HE) and
secure multi-party computation (MPC). Other approaches optimize within one primitive, or fix the split with
linear layers in HE and non-linear ones in MPC. CipherFlow instead places every operator for the deployment
at hand. It searches jointly over each operator's domain, the HE↔MPC conversions, and the CKKS levels and
bootstraps, using state-expanded planning with residual blocks folded (BlockFold), and it prices every choice
with costs measured on the target hardware and network. Its inputs are a model, the hardware, and the
network, which is either given or measured between the two parties. It returns a plan that says, for every
operator, whether it runs in HE or MPC and where to convert or bootstrap, with the expected latency and
communication. It also checks that plan and emulates it on real inputs.

CipherFlow plans against an abstraction of the backends. A backend is a set of
*methods*, each one operator in one domain with the CKKS levels it consumes, the shapes and value ranges it
accepts, and its measured cost. The backend also supplies the measured cost of its HE↔MPC conversions and
bootstrapping. A library plugs in by registering its methods and a cost profile, which MicroBench can
measure on your machine. For HE, the supported backends are OpenFHE (CKKS, with fast and precise
bootstrapping) and NEXUS (SEAL on CPU, Phantom on CUDA). For MPC, they are SCI/BOLT (two parties) and
SHAFT/CrypTen (two parties and a trusted dealer, CPU and GPU). SPU is planned ([ROADMAP.md](ROADMAP.md)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cipherflow-ecosystem-dark.svg">
  <img alt="CipherFlow in the private-inference stack. Model layer: a PyTorch model, traced by cf.models.from_torch into an operator graph (operators, tensor shapes, residual edges). Deployment: hardware profile or this machine, network given or measured with --ip, security model. Planning layer (CipherFlow): MicroBench, state expansion over domains and CKKS levels, BlockFold; cf.compile returns plan.json. Backend layer: HE (OpenFHE, NEXUS) and MPC (SCI, SHAFT) report methods and measured costs and receive the plan; HE-MPC conversions and bootstrapping. Execution: cipherflow.runtime checks the plan and emulates it on real inputs; encrypted two-party execution is planned." src="docs/assets/cipherflow-ecosystem-light.svg" width="100%">
</picture>

| | interface | contents |
|---|---|---|
| ① input | operator graph `G = (V, E)` | operators, tensor shapes and residual edges, traced from a PyTorch `nn.Module` by `cf.models.from_torch` (or built in: `cf.models.bert()`, `vit()`) |
| ② input | deployment `D = (HW, BW, RTT)` | hardware profile (`cpu`, `gpu`, or measured by MicroBench); network (`--network 10Mbps/1ms`, or measured with `--ip SERVER`); security model (`2pc`, `2pc-dealer`) |
| ③ input | backend methods + measured costs | per operator and domain: time, bytes, rounds, CKKS levels, contracts; costs of HE↔MPC conversions and bootstrapping |
| optional input | calibration | value ranges on real inputs (`cf.runtime.calibrate`), so operators stay out of HE where an approximation would not hold |
| ④ output | `plan.json` | per operator: HE or MPC, CKKS levels, kernel; per edge: conversions and bootstraps; latency and communication |

## Demo

A Transformer written in PyTorch goes in. The plan comes out operator by operator: HE or MPC, CKKS levels,
the conversions and bootstraps before each operator, and where each operator is in the code.

![Terminal: python -m demo.plan_pytorch_model --model bert --blocks 1 --network 10Mbps/1ms prints the plan's latency and placement, then every operator with HE or MPC, its CKKS levels, its place in the PyTorch code, and the conversions and bootstraps before it](demo/assets/demo_network_given.gif)

```bash
python -m demo.plan_pytorch_model --model bert --network 10Mbps/1ms -o plan.json   # the network given
python -m demo.plan_pytorch_model --model bert --ip SERVER_IP -o plan.json         # measured to the other party
```

- [`demo/`](demo) has the input models and the two-terminal setup for measuring the network
  (`cipherflow netbench --serve` on the server).
- The [interactive demo](https://ucf-ml-research.github.io/CipherFlow/) shows the PyTorch code, a deployment
  picker, the plan as a graph and as a table for each block, and its latency against static placements.

## Installation

```bash
git clone https://github.com/UCF-ML-Research/CipherFlow.git
cd CipherFlow
pip install -e .            # PyTorch is needed for from_torch, the demo and cipherflow.runtime
```

Planning with the shipped profiles needs none of the HE or MPC backends. To measure costs on your own
machine, see [Measuring costs on your machine](#measuring-costs-on-your-machine).

## Quick start

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

### Command line

```bash
cipherflow compile --torch demo/transformer.py:bert --input-shape 1,128,768 --network 10Mbps/1ms --steps -o plan.json
cipherflow compile --model bert --hardware cpu --network 10Mbps/1ms -o plan.json   # a built-in graph
cipherflow explain plan.json --steps
cipherflow backends                                                     # HE and MPC backends
cipherflow backends --model bert --hardware cpu --security 2pc-dealer   # which backend pairs fit
```

### Measuring the network between the two parties

```bash
cipherflow netbench --serve                                          # on the server
cipherflow compile --torch demo/transformer.py:bert --ip SERVER_IP --steps     # on the client: measure, then plan
cipherflow netbench --ip SERVER_IP                                   # or only measure (prints the --network value)
```

The client measures two things:
- the round-trip time, as the median of one-byte ping-pongs;
- the bandwidth in each direction, by bulk TCP transfer.

It plans with the slower direction and records the measurement in the plan.

### Reproducing the paper

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

See [`cipherflow/README.md`](cipherflow/README.md) for the full [API](cipherflow/README.md#api) and the
[plan format](cipherflow/README.md#plan-format).

## Supported models

| | |
|---|---|
| **supported** | BERT-base-shaped encoders (hidden 768, 12 heads): built-in `cf.models.bert()` / `vit()`, or your own PyTorch encoder stack via `cf.models.from_torch` (hand-written blocks, `nn.MultiheadAttention`, Hugging Face `BertEncoder` / `ViTEncoder`). Costs are exact for the shapes in the profile (built-in: B=1, S=128) and for any shape measured with `profile="auto"` (CPU). |
| **estimated (warning)** | other sequence lengths with a built-in profile: costs are scaled from the nearest profiled shape; `strict=True` refuses instead |
| **kept out of HE** | hidden size ≠ 768 or heads ≠ 12: the NEXUS methods do not apply, so those operators are planned in MPC |
| **not supported (explicit error)** | decoder LLMs and generation (causal mask, KV cache, RoPE, RMSNorm, SwiGLU), CNNs, encoder-decoder models, embeddings and task heads, attention masks, ONNX |

Anything outside this set raises `UnsupportedModel`, `UnsupportedGraph` or `UnsupportedOperator`, naming the
cause. CipherFlow never plans a model with parts of it dropped. [ROADMAP.md](ROADMAP.md) lists what comes
next, decoder LLMs first.

## Backends and cost profiles

A plan uses one HE backend and one MPC backend, chosen with `he=` and `mpc=`:
- `"auto"` plans with every pair that fits and returns the fastest; `plan.alternatives` lists the others and
  why any were not used.
- `security=` sets which MPC security models are acceptable.
- `min_precision=` sets the minimum HE bootstrapping precision in bits.

### HE backends

| HE backend | status | security | bootstrapping (one ciphertext, 32768 values, CPU) |
|---|---|---|---|
| `openfhe-fast` | measured: all ten operators run on one CKKS parameter set ([`native/openfhe`](native/openfhe)) | 128-bit | 8 levels after, 10.5 bits, 17.6 s |
| `openfhe-precise` | measured, two-pass bootstrapping | 128-bit | 7 levels after, 23.5 bits, 34.8 s |
| `nexus` (SEAL / Phantom) | costs are NEXUS reference kernels scaled to each shape; its operators use three parameter sets, so they do not run as one pipeline | unverified | precision not established |

Precision is -log2 of the largest bootstrapping error on random values in [-1, 1].

The OpenFHE kernels:
- pack 128-token blocks; other sequence lengths are kept out of HE;
- approximate GeLU, LayerNorm and Softmax with Chebyshev polynomials on fixed input ranges, checked against
  a calibration (see below);
- convert between HE and MPC with a hardened protocol for values below 2^12: statistically hiding masks on
  every slot, exact decoding, and a wrap-free MPC → HE direction (see
  [`native/openfhe/CONVERSION.md`](native/openfhe/CONVERSION.md)).

An HE → MPC conversion needs 2 levels left on the ciphertext. An MPC → HE conversion costs about 2.8 s of
server time and 22 MB per ciphertext of 32768 values, for the wrap removal.

### MPC backends

| MPC backend | security model | devices |
|---|---|---|
| `sci` (SCI/BOLT, EzPC) | `2pc`: two parties, no third party | CPU |
| `shaft` (SHAFT, CrypTen) | `2pc-dealer`: Beaver triples from a trusted dealer; the offline phase is not costed | CPU, GPU |

`security="2pc"` (the default) allows only `sci`. `"2pc-dealer"` also allows `shaft`. `"any"` allows
everything, including profiles from unregistered backends. If no pair fits, `compile` raises
`UnsupportedBackend` with the reason for each one.

### Built-in profiles

Built-in profiles are named `<hardware>-<he>-<mpc>`:
- `cpu-openfhe-{fast,precise}-{sci,shaft}` are measured end to end on one machine: HE operators and
  bootstrapping with OpenFHE, MPC operators in two-party runs, and conversions with real encryption.
- `cpu-nexus-sci`, `cpu-nexus-shaft` and `gpu-nexus-shaft` have measured MPC and conversions, but NEXUS
  reference-scaled HE.
- `paper-cpu` / `paper-gpu` are the profiles used in the paper.

The raw measurements and the script that builds the profiles are in `compiler/state_expanded/profiles/`
(`raw/`, `build_measured.py`).

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

## Status and roadmap

Planning, cost profiles, plan checks and emulation on real data work today. Encrypted end-to-end execution
of a plan is the next milestone. [ROADMAP.md](ROADMAP.md) lists what is done and what is planned, and
[CHANGELOG.md](CHANGELOG.md) lists what changed in each release.

Known limitations:

- Plans are emulated (`cipherflow.runtime`: the backends' numerical models on real data). They are not yet
  executed under encryption end to end.
- The OpenFHE base kernels approximate GeLU, LayerNorm and Softmax on fixed input ranges that fine-tuned BERT
  exceeds. Their wide-range versions (`native/openfhe/NONLINEAR.md`) are chosen per node from a calibration and
  pay off at low bandwidth (SCI, 10 Mbps).
- The wide-range kernels' costs were measured on a busy machine and are too high.
- The planner's own bootstraps of residual-stream values assume random-like data; structured values bootstrap
  worse.
- HE↔MPC conversions are hardened for the OpenFHE backend only. The NEXUS (SEAL) profiles still price the
  slot-domain mask, and plans warn about it.
- Bootstrapping precision depends on the machine and the OpenFHE build; check yours with `cipherflow selfcheck`.

## Repository layout

| path | contents |
|---|---|
| `cipherflow/` | public API: `compile`, `Deployment`, `Plan`, PyTorch front end, runtime (checks, emulation, calibration), network measurement, CLI |
| `compiler/state_expanded/` | planner: state-expanded planning, BlockFold (SESE), cost model, MicroBench profilers, measured profiles |
| `ir/` | operator graph types |
| `backends/` | HE (NEXUS) adapters used by the MicroBench profilers |
| `native/` | measured kernels: OpenFHE HE operators, bootstrapping and conversions (`native/openfhe`), SEAL conversions (`native/conversion`) |
| `demo/` | the demo: PyTorch Transformers in, plans operator by operator out (script, web page builder, terminal recordings) |
| `docs/` | the demo page served by GitHub Pages (built by `demo/web`), the logo and the README figure (`docs/assets/`) |

## Measuring costs on your machine

Measuring costs on your own machine (`profile="auto"`, `cipherflow profile`) runs the backends:

| backend | build / location |
|---|---|
| OpenFHE kernels (`openfhe-*`) | OpenFHE 1.4: `cmake -S native/openfhe -B native/openfhe/build && cmake --build native/openfhe/build` |
| conversions for `nexus` | SEAL 4.1: `cmake -S native/conversion -B native/conversion/build && cmake --build native/conversion/build` |
| `nexus` reference kernels | NEXUS build: `CIPHERFLOW_NEXUS_BUILD` |
| `sci` | SCI (EzPC) bridges: `CIPHERFLOW_SCI_BIN` |
| `shaft` | SHAFT (CrypTen fork): `CIPHERFLOW_SHAFT_ROOT` |

### Checking a machine

Before trusting results computed on a machine, check that it computes correctly. The OpenFHE kernels are
deterministic, so the self-check runs them in fresh processes and requires that:
- every key works;
- every bootstrap is correct;
- the same input gives a bit-identical result every time;
- parallel threads agree bit for bit with a serial run.

```bash
cmake -S native/openfhe -B native/openfhe/build && cmake --build native/openfhe/build --target he_ops
python -m cipherflow selfcheck                    # about 15 min; needs about 23 GB of free memory
python -m cipherflow selfcheck --quick            # about 5 min
python -m cipherflow selfcheck --processes 12 --memtest 8G     # thorough, with memtester
python -m cipherflow selfcheck --processes 0 --race-rounds 0 --workload 2   # the BERT operators in sequence
python -m cipherflow selfcheck --mode precise --quick          # two-pass bootstrapping
python3 cipherflow/selfcheck.py --memtest-only --memtest 8G    # memory only; no OpenFHE needed
```

`--workload N` runs the ten BERT-base operators in sequence (as MicroBench does) in N processes, checks each
against plaintext, and bootstraps afterwards, so that any damage the kernels leave behind shows up.

The self-check prints `PASS` or the fault it found. It writes a JSON report with the machine, its ECC counters
when the kernel reports them, and every check.

Any of these means the machine corrupts data and its results cannot be trusted (timings are unaffected):
- a wrong or nondeterministic result from a deterministic computation;
- processes that disagree;
- a memtester failure.

A pass does not rule out rare faults.

<!-- ## Citation

```bibtex
@inproceedings{zhang2026cipherflow,
  title     = {CipherFlow: Hardware-Aware Compiler Framework for Low-Latency Hybrid Secure Inference},
  author    = {Zhang, Hedong and Zheng, Mengxin and Lou, Qian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
``` -->

## License

Apache License 2.0 — see [LICENSE](LICENSE).

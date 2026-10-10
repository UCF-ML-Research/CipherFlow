# CipherFlow Roadmap

CipherFlow plans private inference for a model written in PyTorch. For every operator it decides whether
the operator runs under homomorphic encryption (HE) or secure multi-party computation (MPC). It also decides
where to convert between the two and where to refresh a ciphertext, using costs measured for the deployment.

The [demo](demo) shows the path end to end. A PyTorch Transformer and a deployment go in, and the plan comes
out operator by operator. This roadmap follows that path:
- the model and the deployment that go in;
- the backends whose costs the planner weighs;
- the plan that comes out;
- running the plan.

Done items are checked, with the release that delivered them.
Planned items are unchecked.

The path has three stages:
1. **Discriminative Transformers (done):** BERT and ViT encoders, from PyTorch to measured plans.
2. **Generative models (next):** GPT-2, then LLaMA-style decoders ([section 7](#7-generative-models-next)).
3. **Private inference runtime (then):** plans executed under encryption by the two parties
   ([section 6](#6-running-the-plan)), on more backends ([section 3](#3-backends-and-costs)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/cipherflow-roadmap-dark.svg">
  <img alt="CipherFlow roadmap. Done: discriminative Transformers (BERT-base, ViT-B/16): PyTorch front end, measured HE and MPC backends, plans, calibration and emulation. Next: generative models (GPT-2, then LLaMA-style decoders): backend operators for decoders, a torch.export front end, prefill and decode plans. Then: a private inference runtime: a two-party runtime, encrypted execution from BERT to LLMs, GPU backends and SPU. Below, the details of the next stage: step 1 adds the backend operators, step 2 brings in the models; a LLaMA-style decoder block marks which operators exist today and which step 1 adds." src="docs/assets/cipherflow-roadmap-light.svg" width="100%">
</picture>

### Details

**② Generative models (next)** comes in two steps:
- **Step 1, backend operators.** Causal mask, RoPE, RMSNorm, SwiGLU, the KV cache, embedding lookup, LM head
  and sampling. Each gets HE kernels, MPC protocols and costs measured by MicroBench.
- **Step 2, the models.** A `torch.export` front end; prefill and decode planned separately, with the KV
  cache as state; per-token latency and communication. GPT-2 small first, then a LLaMA-style model.

**③ Private inference runtime (then):**
- A two-party runtime: the parties' keys, the transport, and the conversion protocol across processes.
- Encrypted execution of plans: one BERT-base layer first, then whole models, then LLMs.
- More backends: GPU kernels for OpenFHE, and SPU.

## 1. Model input: PyTorch front end

Today the input is an encoder (BERT, ViT) written in PyTorch. Anything outside the operator set fails
explicitly instead of being planned with parts missing. Next comes a front end for general models, and
generative models above all (section 7).

- [x] Built-in graphs: BERT-base, ViT-B/16, and the paper's graphs (`bert_paper`, `vit_paper`) (v0.1, v0.2.0)
- [x] `cf.models.from_torch`: encoder stacks traced with torch.fx (hand-written blocks, `nn.MultiheadAttention`, Hugging Face `BertEncoder` / `ViTEncoder`) (v0.1.1)
- [x] Explicit errors for operations outside the operator set (`UnsupportedModel` names every one) (v0.1.1)
- [x] Q/K/V projections as an operator (`Linear_QKV`), priced on every backend (v0.2.0)
- [x] Each operator's place in the PyTorch code (`node.source`, e.g. `blocks.0.attn: softmax`) (v0.4.0)
- [x] `cipherflow compile --torch FILE.py:NAME` (v0.4.0)
- [ ] `torch.export` (Core ATen) front end instead of torch.fx module matching
- [ ] `from_onnx`
- [ ] Embeddings and task heads (today the client computes them in plaintext)
- [ ] Attention masks (padding)
- [ ] HE kernels for sequence lengths other than 128-token blocks (other lengths are kept in MPC)

## 2. Deployment input: hardware and network

A plan is made for one deployment, given by the hardware's cost profile and the network between client and
server.

- [x] Built-in CPU and GPU profiles; the network given as `BANDWIDTH/RTT` (v0.1)
- [x] Security models: `2pc` (no third party) and `2pc-dealer` (trusted dealer); `cf.match` reports which backend pairs fit (v0.2.0)
- [x] Plan for this machine: MicroBench measures the costs a model is missing (`profile="auto"`, `--hardware local`, CPU)
- [x] Network measured between the two parties: `cipherflow netbench --serve`, then `--ip SERVER` (RTT, bandwidth each way) (v0.4.0)
- [ ] MicroBench on GPU (today: pass a measured GPU profile)
- [ ] Asymmetric links in the cost model: today one bandwidth is used for both directions (the slower one when measured)
- [ ] Links among more than two parties (for three-party backends)

## 3. Backends and costs

The planner weighs what each backend measures: operator costs, HE↔MPC conversions, bootstrapping, and the
CKKS levels each kernel consumes.

### MPC

- [x] SCI/BOLT (EzPC): two parties, no third party (v0.2.0)
- [x] SHAFT (CrypTen): two parties with a trusted dealer, CPU and GPU (v0.2.0)
- [ ] SPU: Cheetah (two parties) and SEMI2K (trusted dealer)
- [ ] The dealer's offline phase priced (SHAFT; today not costed)

### HE

- [x] NEXUS (SEAL / Phantom), reference-scaled costs (v0.1)
- [x] OpenFHE CKKS: the ten operators, bootstrapping and conversions on one 128-bit parameter set, fast and precise bootstrapping, measured (v0.2.0)
- [x] Wide-range OpenFHE GeLU, Softmax and LayerNorm, chosen per node from a calibration (v0.3.0)
- [x] Bootstrapping limits (levels left, amplitude, scaled bootstrapping) in the planner, baselines and emulator (v0.3.0)
- [x] `cipherflow selfcheck`: checks that a machine computes the OpenFHE kernels correctly (v0.2.0)
- [ ] Wide-range kernel costs re-measured on a dedicated machine (the current ones were measured under load and are too high)
- [ ] NEXUS on one executable parameter set
- [ ] GPU kernels for the OpenFHE backend

### Conversions

- [x] HE↔MPC conversions measured with real encryption (SEAL for NEXUS, OpenFHE) (v0.2.0)
- [x] Hardened conversion for OpenFHE: statistically hiding masks, exact decoding, wrap-free MPC → HE (v0.3.0)
- [ ] Hardened conversion for NEXUS (SEAL)

## 4. Planner

State-expanded planning over domains and CKKS levels, with residual regions folded (BlockFold).

- [x] State-expanded planning with BlockFold (SESE regions); the cost model; `plan.json` and CLI (v0.1)
- [x] Explicit limits: cost-source report and `strict`, method contracts, graph checks (v0.1.1)
- [x] One plan per backend pair, the fastest chosen, with the others and why any were excluded (v0.2.0)
- [x] Bootstrapping priced per ciphertext; kernels that bootstrap internally (v0.2.0)
- [x] Nested residual regions; V as its own edge; each action records its edge (v0.3.0)
- [x] Value contracts from a calibration keep operators out of HE where an approximation would not hold (v0.3.0)
- [x] Static placements as baselines (BumbleBee-, BOLT- and BLB-style rules, all-HE, all-MPC)
- [ ] Graphs beyond a chain of two-branch residual regions (cross-attention, encoder–decoder)
- [ ] Stateful models: separate prefill and decode plans with the KV cache as state

## 5. Output: the plan

- [x] `plan.json` (`cipherflow.plan/v1`): each operator's domain, levels and HE kernel; conversions and bootstraps on their edges; latency breakdown and online communication (v0.1; kernels and edges v0.3.0)
- [x] `plan.explain()` and `cipherflow explain` (v0.1)
- [x] The plan operator by operator, with PyTorch locations: `plan.steps()`, `explain(steps=True)`, `--steps` (v0.4.0)
- [x] Plan checks: every edge's conversions and bootstraps take the value to its consumer's domain and level (`check_plan`) (v0.3.0)

## 6. Running the plan

Plans are checked and emulated on real data today. Running them under encryption is the next milestone. All
of its cryptographic kernels exist as measured benchmarks; what is missing is the runtime that connects them.

### Emulation

- [x] Emulation on real data: each operator in its domain's numerical model (polynomial approximations, measured bootstrapping noise, fixed point, conversion rounding) (v0.3.0)
- [x] Calibration on real inputs (`runtime.calibrate`), with margins from measured calibration-to-test gaps (v0.3.0)
- [x] GLUE accuracy by emulation: SST-2, MRPC, RTE and STS-B plans match the plaintext model (v0.3.0)
- [ ] Value ranges enforced at run time (today they are calibrated, i.e. statistical)

### Encrypted execution

- [ ] HE kernels on real inputs and weights, with the client's and server's keys separated and ciphertexts serialized
- [ ] The conversion protocol split into its client and server halves, across processes
- [ ] MPC operators on real secret shares, driven by the plan
- [ ] Plan executor: operators, conversions and bootstraps in plan order, for both parties
- [ ] Two-party runtime: client and server processes, transport, measured time and traffic per action against the plan's estimate
- [ ] First milestone: one BERT-base layer under encryption, matching PyTorch to the error the emulation predicts
- [ ] A full BERT-base model on one input, end to end

## 7. Generative models (next)

Generative models come in two stages. First, the backends gain the operators that decoders add over
encoders: each one gets HE kernels, MPC protocols and measured costs. Then decoder models are traced from
PyTorch and planned.

### Step 1: backend operators

- [ ] Causal attention mask
- [ ] RoPE (rotary position embedding)
- [ ] RMSNorm
- [ ] SwiGLU: SiLU and the gated product of two projections
- [ ] KV cache: appending keys and values, attention over the cached ones
- [ ] Embedding lookup, LM head (vocabulary-size projection) and sampling
- [ ] For each operator: HE kernels with calibrated ranges, MPC protocols, and costs measured by MicroBench at prefill and decode shapes

### Step 2: the models

- [ ] `torch.export` front end for decoder models
- [ ] Prefill and decode planned separately, with the KV cache as state
- [ ] Per-token latency and communication for every deployment
- [ ] GPT-2 small, then a LLaMA-style model
- [ ] Plan checks and emulation on real prompts
- [ ] A general tensor IR: operand ownership (client, server, public), symbolic shapes, layouts

## 8. Demo and documentation

- [x] Interactive demo page (v0.1)
- [x] Demo input as PyTorch code and output as the plan operator by operator, with a JSON download (v0.4.0)
- [x] Demo folder (`demo/`): models, a script, the page builder, terminal recordings (v0.4.0)
- [x] Design notes: hardened conversion (`native/openfhe/CONVERSION.md`), wide-range kernels (`native/openfhe/NONLINEAR.md`), planner optimality (`compiler/state_expanded/OPTIMALITY.md`), cost provenance (`compiler/state_expanded/profiles/DATA_PROVENANCE.md`)
- [ ] Demo page with the measured OpenFHE + SCI profiles (today: the paper's profiles)
- [ ] The page for any plan (`cipherflow view plan.json`)

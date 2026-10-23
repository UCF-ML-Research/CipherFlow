# CipherFlow and other planners

Many tools compile models for HE or for MPC. Most work inside one primitive and could serve CipherFlow as
backends. A few, like CipherFlow, decide which operators run in HE and which in MPC; this page compares
CipherFlow with the two closest, CostSphere and EncFormer.

- [Where CipherFlow sits](#where-cipherflow-sits)
- [Comparison with CostSphere and EncFormer](#comparison-with-costsphere-and-encformer)
- [Reproducing the comparison](#reproducing-the-comparison)

## Where CipherFlow sits

| kind | systems | what they decide |
|---|---|---|
| HE compilers | Orion, HEIR, HECO, Coyote, nGraph-HE | packing, rotations, levels and bootstrapping, inside HE |
| MPC compilers | EzPC / CrypTFlow, SecretFlow-SPU, MP-SPDZ, CrypTen | lowering to one MPC protocol family |
| mixed protocols inside MPC | HyCC, Matchmaker | which MPC sharing or protocol suite runs each operation or batch |
| hybrid HE + MPC planners | MP2ML, BLB, EncFormer, CostSphere, CipherFlow | which operators run in HE and which in MPC, and where to convert |

The compilers in the first two rows are best compared as backends: one compiler on its own against CipherFlow
planning with it and another (see [Adding new backend primitives](../README.md#adding-new-backend-primitives)).
The comparison below is with the hybrid planners.

## Comparison with CostSphere and EncFormer

### Method

Neither paper released code, so both placement methods are re-implemented from the papers, in
[`demo/compare`](../demo/compare). Every method plans with CipherFlow's measured costs and at CipherFlow's
operator granularity, and one evaluator scores every plan. A difference in latency therefore comes from the
planning method, not from kernels or hardware.

- **CostSphere** (IACR ePrint 2025/1315) plans with an ILP over the operators in topological order:
  - a conversion is charged only between neighbours in that order;
  - a bootstrap is charged after a count of consecutive HE operators (its Eq. 4).

  The paper does not state the count. We use the level budget divided by the model's mean level use (9), and
  also the best count for each setting. Its Eq. 13, a sliding-window variant, runs as an ablation.
- **EncFormer** (IEEE TDSC 2026) uses a fixed split: linear layers and attention products in HE; Softmax,
  LayerNorm and GELU in MPC; no bootstrapping. The paper does not place the residual adds, so we take the
  cheaper choice. A second run gives every planner EncFormer's conversion, 3.3× cheaper (its Table VIII).

Not reproduced:
- CostSphere's mixing of HE and MPC inside Softmax, LayerNorm and GeLU, which needs costs per sub-operation;
- CostSphere's regression cost model: here it gets CipherFlow's exact costs;
- EncFormer's distilled Softmax and LayerNorm replacements, which change the model;
- EncFormer's GELU pre-evaluation and packing.

**Setup:**
- **Models:** BERT-base and ViT-B/16, the paper's graphs.
- **Costs:** the paper's profiles (`paper-cpu`, `paper-gpu`; see [Built-in profiles](backends.md#built-in-profiles)):
  - CPU: AMD Threadripper PRO 3955WX, NEXUS and SHAFT measured;
  - GPU: an RTX 3090 on the same host, SHAFT measured, NEXUS-CUDA partly measured and partly derived from
    CPU costs.
- **Networks:** 16 settings (10 Mbps to 3 Gbps × 1 to 80 ms), plus the papers' own LAN and WAN settings.
- **Latency:** estimated from the measured costs; the networks are applied by the cost model, not run.

### Results

Latency relative to CipherFlow, over the 16 network settings (above 1×: slower than CipherFlow):

| | BERT, CPU | BERT, GPU | ViT, CPU | ViT, GPU |
|---|---|---|---|---|
| CostSphere | 1.00–1.05× | 0.98–1.03× | 1.00–1.04× | 1.00–1.03× |
| CostSphere, best count per setting | 1.00× | 0.98–1.03× | 1.00× | 1.00–1.03× |
| CostSphere, Eq. 13 | 1.00–1.05× | 0.98–1.16× | 1.00–1.04× | 1.00–1.15× |
| EncFormer | 2.0–21.3× | 3.1–23.3× | 2.2–22.7× | 3.2–24.2× |

With EncFormer's conversions given to every planner, EncFormer stays 1.7–22.6× slower, and CostSphere
1.00–1.02× (1.00–1.03× with Eq. 13).

**What the results say:**
- **EncFormer.** Its fixed split loses where the best plan does not follow linear in HE, nonlinear in MPC. On
  CPU from 30 Mbps up the best plan is all MPC; on GPU at low bandwidth it is all HE. Cheaper conversions do
  not close the gap, because those plans hardly convert.
- **CostSphere.** With CipherFlow's exact costs it often finds a plan as fast as CipherFlow's: in 46 of the
  64 model, platform and network combinations, and 54 with its best count. Where the plan mixes HE and MPC,
  CipherFlow's modeling of residual joins and exact CKKS levels gains:
  - up to 4.8% over CostSphere's count from the level budget (BERT, CPU, 10 Mbps);
  - up to 4.4% over its best count for each setting (BERT, GPU, 400 Mbps / 4 ms).
- **The paper's speedups.** The 16.5× and 22× speedups are over fixed splits (BumbleBee, BOLT, BLB, and
  EncFormer's split), not over a cost-driven planner like CostSphere.

**The one setting where CostSphere is faster.** On BERT, GPU, 1 Gbps / 20 ms, CostSphere's plan takes 123.7 s
and CipherFlow's 125.9 s.
- **Cause:** CipherFlow's planner prices the conversion into QKᵀ at its packed Q/K/V input, three times the
  hidden state that the edge carries and the evaluator prices.
- **With edge-shape pricing:** planned with the edge's shape, CipherFlow takes 119.7 s there. CostSphere is
  then never faster, even with its best count, in any of the 176 runs: 2 models × 2 platforms × 22 networks
  × 2 conversion settings.

Changing the planner's pricing is on the [roadmap](../ROADMAP.md#4-planner).

The papers' own end-to-end numbers come from other kernels, hardware and security models, so they are not
compared here.

## Reproducing the comparison

```bash
python -m demo.compare.prior_planners --quick                    # BERT-base, four networks, about 2 minutes
python -m demo.compare.prior_planners                            # the comparison above, about 25 minutes
python -m demo.compare.prior_planners --encformer-conversions    # every planner with EncFormer's conversion
```

Each run prints the tables and writes every plan's latency, conversions and bootstraps to a CSV. The Eq. 13
ablation needs SciPy.

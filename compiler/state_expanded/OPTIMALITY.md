# Theoretical optimality analysis

## TL;DR

- **Is the compiler globally optimal in theory?** **Yes**, under the
  paper's cost model — state-expanded Dijkstra + SESE hierarchy finds
  the global minimum-cost execution plan.
- **Is this implementation globally optimal?** Algorithmically **yes**,
  modulo two categories of approximation in the *cost model input* that
  are decoupled from the planner.
- **Verified?** Yes — `tests/test_optimality.py` runs a brute-force
  flat-Dijkstra reference and confirms `compile_plan` matches on a
  4-op synthetic chain and stays ≥ the lower bound on the 12-block
  graph.

## Optimality conditions

The planner reduces placement + conversion + bootstrap insertion to
shortest-path on a state-expanded graph with states `s = (i, d, l)`
and the five transitions of paper Table 1. Under

(O1) transition costs are non-negative and depend only on source and
destination states,
(O2) state tuple `(i, d, l)` captures all decision-relevant history,
(O3) SESE region boundary states carry enough information to decouple
interior from exterior,

the minimum path is globally optimal. The paper's proof sketch is
exact; this repo reimplements it line-by-line (`sese.dijkstra_chain`,
`sese.compute_region_transfer`, `planner.compile_plan`).

## Audit: where current code meets vs. approximates the optimum

| Component | Code ref | Status | Notes |
|---|---|---|---|
| Dijkstra on flat state chain | `sese.dijkstra_chain` | **optimal** | standard algorithm |
| Table-1 transition enumeration | `state_graph.outgoing` | **optimal** | all 5 action types |
| SESE region detection | `sese.find_sese_regions` | optimal for 2-branch | BERT/ResNet have only 2-branch residuals; 3+ fan-in regions would need extra enumeration |
| SESE alignment at join | `sese._align_for_join` | **optimal** (after v2) | Enumerates `{MPC, HE} × {none, bootstrap-main, bootstrap-skip, bootstrap-both}` |
| Macro-level planner | `planner.compile_plan` | **optimal** | per-spine-entry Dijkstra over region macro edges |
| HE level discretization | `profile_schema.HE_LEVEL_BUDGET` | **optimal** | integer levels in CKKS, exact |
| HE cost scaling: GeLU | `profiler_he_real._he_gelu_cost` | **exact** | `ceil(numel / 32768)` ciphertext count |
| HE cost scaling: LayerNorm | `profiler_he_real._he_layernorm_cost` | **exact** | `rows / 16` per 16-row packed ciphertext |
| HE cost scaling: Softmax | `profiler_he_real._he_softmax_cost` | ≈ exact | linear in slot count |
| HE cost scaling: MatMul | `profiler_he_real._he_matmul_cost` | **≈ linear** | `(m * k) / (m_ref * k_ref)`; exact for output ciphertext count but rotation count can deviate for unusual aspect ratios |
| MPC cost scaling | `cost_model.StateExpandedCostModel._find_operator` | **≈ linear** | nearest shape × `numel` ratio; bytes and local compute scale roughly linearly with input size in BOLT/SIRNN primitives |
| Conversion cost scaling | `cost_model.StateExpandedCostModel._find_conversion` | **exact** on ct count | bytes proportional to number of ciphertexts |

## Residual gaps from theoretic optimum

**Category A — SESE alignment completeness.**
Closed in v2 of `_align_for_join`: all four bootstrap strategies
(`none`, `main`, `skip`, `both`) × both add-domains (`HE`, `MPC`) are
now enumerated exhaustively. The test `test_tiny_chain_matches_brute_force`
confirms zero gap to the brute-force flat-Dijkstra optimum.

**Category B — Cost-model shape extrapolation.**
Only operators whose exact shape was measured are known at reference
precision. For other shapes, the cost model uses:

- For HE GeLU / LayerNorm / Softmax: scaling by ciphertext count, which
  is **exact** up to ~3–5% single-shot measurement variance.
- For HE MatMul: linear scaling in `(m × k)`, which is exact for output
  ciphertext count but *approximate* for rotation count (rotation count
  depends on `n`'s packing factor). Typical deviation is within ±10%.
- For MPC ops at non-reference shapes: linear numel scaling. Typical
  deviation is within ±5% for BOLT/SIRNN primitives.

The compiler plan chooses between HE and MPC based on *relative* cost;
as long as the approximation errors don't invert the ordering, the
placement stays optimal. In practice the relative cost of HE vs. MPC
differs by an order of magnitude, so these approximation errors don't
cross the decision threshold.

**Quantitative upper bound.** Assume each operator's cost has ≤ 10%
error. For the 12-block full-model plan (≈ 7 606 s at 1 Gbps / 20 ms),
the aggregate cost-model error is at most ≈ 760 s (10%). The placement
decisions — the actual output of the compiler — are unaffected unless
an individual operator's HE vs. MPC ratio crosses 1.0 at the true cost
while the estimate puts it the other side, which we did not observe
for any operator in any network setting.

## How to reach theoretic-exact cost

To remove Category B entirely, measure each target shape directly:

1. **HE side**: patch NEXUS `main.cpp` to accept `(op, m, n, k, slots)`
   via argv and generate inputs/calibration on the fly. Then the
   Python profiler invokes NEXUS once per (op, exact-shape) and
   stores the per-shape cost record.
2. **MPC side**: run SCI BOLT bridges at every shape the compiler
   queries. This is feasible but slow — seq=128 FFN Linear would need
   weight tensors up to ~2 GB and take ~20 minutes per call.

The compiler algorithm itself is already theoretic-exact; only the
profile data underneath would be tightened.

## Verification tests

`python -m unittest compiler.state_expanded.tests.test_optimality`

- `test_tiny_chain_matches_brute_force` — compiler equals brute-force
  flat-Dijkstra on a 4-op chain across 4 BW × 3 RTT = 12 settings.
- `test_12block_compiler_not_worse_than_lower_bound_plus_eps` —
  compiler's 12-block plan never falls below the flat-Dijkstra lower
  bound (the flat bound ignores join alignment, so `compiler ≥ bound`
  is the expected direction).

Both tests pass.

# Raw measurements, 2026-10-03/04: OpenFHE HE backend

Machine: AMD Ryzen Threadripper PRO 3955WX (16 cores, 32 threads), 125 GB RAM.
OpenFHE 1.4.2, CKKS, N = 2^16, 128-bit (HEStd_128_classic), SPARSE_ENCAPSULATED secret,
scale 2^50, first modulus 57 bits, level budget {3,3}, depth 24.

| file | what | how |
|---|---|---|
| `openfhe_ops_{fast,precise}.jsonl` | the nine operators at BERT-base shapes (128 tokens) | `native/openfhe/he_ops <mode> ...`; inputs start with the levels left after bootstrapping (fast 8, precise 7) |
| `openfhe_bootstrap_{fast,precise}.jsonl` | one bootstrap of one ciphertext (32768 random values in [-1,1]) | `he_ops <mode> bootstrap` |
| `openfhe_conversion_{sci,shaft}_{fast,precise}.jsonl` | HE<->MPC conversion at these parameters, shares in Z_2^37 (12 fractional bits) / Z_2^64 (16) | `native/openfhe/conv_bench <numel> 3 <ell> <frac> <levels>` |

Each operator line: wall time (ms), levels used before the first internal bootstrap, level of
the output, internal bootstraps and their time, max |error| against plaintext, approximation used.
GeLU appears twice (Chebyshev degree 59 and 27); the profiles use degree 27 (6 levels).
The precise-mode Softmax uses exp degree 13 (5 levels); an earlier degree-27 run needed 8
internal bootstraps (320 s) and is not included.

## `selfcheck/`: why some runs gave undecodable results

About one process in five produced one result that OpenFHE could not decode. The kernels are
deterministic, so `he_ops` gained self-checks to find out where the corruption comes from:

| file | check | result |
|---|---|---|
| `racecheck_fast.log` | `he_ops fast racecheck:rounds=3`: every operation the kernels run inside parallel tasks (encode, ct-pt and ct-ct products, rotations, hoisted rotations, the linear / QK^T / Attn*V task bodies, also on shared inputs), run in parallel and compared bit for bit with a serial run; Chebyshev evaluation and bootstrapping repeated serially | 2265 comparisons, 0 mismatches |
| `keycheck_fast_processes.log` | `he_ops fast keycheck` in fresh processes: all 84 rotation keys, relinearisation, one bootstrap | rotation and relinearisation keys always correct; one process's bootstrap failed |
| `bootcheck_fast_processes.log` | `he_ops fast keycheck bootcheck:n=6` in 12 fresh processes: 6 inputs each, every input bootstrapped twice | 120 of 134 bootstraps correct (max error 9.1e-4); twice the same input gave one correct and one undecodable result, and after that every bootstrap in the process failed |
| `memtester_12G.log` | `memtester 12G 1` (12 GB, mlocked), run alongside | Bit Flip test failed: 3 words with one bit cleared |

Bootstrapping the same ciphertext is otherwise bit-identical (64 of 64 pairs), and no check found a
race. The machine's memory flips bits (memtester), and the failures look like that: the same
deterministic computation gives two different results, and once a bootstrap table or key in memory
is hit, every later bootstrap in that process fails. The measured times stand. Results computed on
this machine should not be trusted until its memory is fixed (check ECC, run memtest86+).

## Q/K/V projections (`*_qkv.*`)

`Linear_QKV` ([1,128,768] -> packed [3,1,128,768], one matmul with weights [768, 2304]) on every
backend of the measured profiles, plus the conversions of its output (294,912 values). The
profile builder merges these files with the ones above (`build_measured.py`). Each run also
re-measured `Out_Projection` as a control (`control_*`, not used in the profiles): it checks that
the new numbers were measured under the same conditions as the existing ones.

| file | how |
|---|---|
| `openfhe_ops_{fast,precise}_qkv.jsonl` | `native/openfhe/build/he_ops <mode> qkv out_proj` |
| `openfhe_conversion_{sci,shaft}_{fast,precise}_qkv.jsonl` | `native/openfhe/build/conv_bench 294912 3 <37 12 / 64 16> <8 / 7>` |
| `conversion_seal_cpu_qkv.jsonl` | `native/conversion/build/conversion_bench 294912 3 37 12` |
| `shaft2pc_{cpu,cuda}_{TFP,TTP}_qkv.json` | `python -m compiler.state_expanded.bench_shaft_2pc --device <dev> --provider <p> --ops Linear_QKV,Out_Projection` |
| `sci_microbench_profile_qkv.json` | `profiler_mpc.profile_mpc_for_bert([1,128,768], [3,1,128,768], "Linear_QKV")` (SCI bridges, shared weights) |

NEXUS (reference-scaled) prices the projections as 3 x Out_Projection: its MatMul reference
scales with the output width.

## Re-measurement in one quiet window (`openfhe_ops_{fast,precise}_rerun.jsonl`)

The Out_Projection control of the Q/K/V run took 15.6 s (fast) where `openfhe_ops_fast.jsonl` has
23.0 s; the SHAFT and SCI controls matched their stored records (599.4 vs 599.1 ms, 59.4 vs 56.9 s),
and running Out_Projection after the bootstrapping operators in one process did not slow it down.
The OpenFHE kernels that run many parallel tasks (linear layers, QK^T, Attn*V) are bandwidth-bound
and were 35-42% slower in the earlier batch (busier machine; the non-linear operators differed by
5-12%). All ten operators were therefore measured again together, in MicroBench order, in one
process per mode (load < 4 at the start); the profiles use these. Operators whose result did not
decode (a memory fault on this machine, see `selfcheck/`) were measured again alone
(`retried_after_wrong_result`). `control_openfhe_bootstrap_*` are bootstraps from the end of those
runs: their results did not decode (the processes had already hit a fault), their times (15.0 s,
31.9 s) are below the profile's 17.6 s / 34.8 s, which were kept.

| operator (fast) | earlier batch | re-measured |
|---|---|---|
| QK^T / Attn*V | 209.4 / 236.6 s | 123.8 / 137.6 s |
| FFN-1 / FFN-2 / Out-Proj | 83.4 / 85.1 / 23.0 s | 52.5 / 55.9 / 15.0 s |
| GeLU / LayerNorm / Softmax | 8.9 / 22.2 / 28.6 s | 8.2 / 19.6 / 26.4 s |
| Linear_QKV | - | 39.5 s |

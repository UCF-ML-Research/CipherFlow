# Raw measurements, 2026-10-03

Machine: AMD Ryzen Threadripper PRO 3955WX (16 cores), 125 GB RAM, 2x NVIDIA RTX 3090 (24 GB).
All two-party runs on localhost; network cost is composed later from bytes and rounds.

| file | what | how |
|---|---|---|
| `sci_microbench_profile.json` | SCI/BOLT MPC operators, BERT-base shapes at S=16 and S=128 | `cipherflow profile --backend sci` (two bridge processes per operator) |
| `shaft2pc_{cpu,cuda}_{TFP,TTP}.json` | SHAFT (CrypTen) operators at S=128 | `compiler/state_expanded/bench_shaft_2pc.py`: two party processes (one GPU each on cuda); TFP = online only, TTP = with a dealer process; `wire_bytes` from the loopback interface counter |
| `conversion_seal_cpu.jsonl` | HE<->MPC conversion, real CKKS (SEAL 4.1, N=2^16, NEXUS modulus chain) | `native/conversion/conversion_bench <numel> 3 37 12` |

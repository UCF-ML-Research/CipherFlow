# Raw measurements, 2026-10-06: wide-range LayerNorm, bootstrapping limits

Machine and OpenFHE parameters as in `../2026-10-04/`. Design: `native/openfhe/NONLINEAR.md`.

| file | how |
|---|---|
| `openfhe_ops_{fast,precise}_layernorm_wide.jsonl` | `he_ops <mode> layernorm_wide:vlo=0.016,vhi=3.15 layernorm_wide:vlo=0.47,vhi=123 layernorm_wide:vlo=0.47,vhi=2340 out_proj` |
| `openfhe_bootlevel_{fast,precise}.jsonl` | `he_ops <mode> bootlevel` (extends `../2026-10-05/`: 3-7 levels left, values repeated every 128 slots, a constant outside one block, products with a constant) |

`build_measured.py` prices `LayerNorm.wide` from the dearer of the two 2-step runs and
`LayerNorm.wider` from the 6-step run. fast: 81.5 / 81.9 / 156.1 s (4 / 4 / 8 bootstraps), max error
3.8e-4 / 5.9e-4 / 1.4e-3 in normalised-output units; precise: 159.1 / 158.6 / 307.2 s, 3.4e-8 /
1.2e-7 / 2.1e-7. The Out_Projection controls (23.5 s fast, 21.1 s precise, against 15.0 s) show a
busy machine: the costs are too high.

bootlevel: bootstrapping works at 1-7 levels left and after a product with a constant; values
repeated in every 128-slot block are bootstrapped worse from |x| ~ 8 (fast 1.8e-3 at 8, 1.3e-2 at 18;
random values: 7e-4 up to 32), and a vector with a constant outside one block much worse (constant
2.9: 1.2e-2, 10: 0.4, both modes): the mean sits in one coefficient. Hence the LayerNorm kernel's
handling of its unused blocks and the emulator's `BOOTSTRAP_MEAN_WEIGHT`.

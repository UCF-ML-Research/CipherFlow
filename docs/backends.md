# Backends and cost profiles

A plan uses one HE backend and one MPC backend, priced by a cost profile measured on a machine. This page lists
the backends and the built-in profiles, and explains how to measure costs on your own machine and check that
it computes correctly.

- [Choosing a backend pair](#choosing-a-backend-pair)
- [HE backends](#he-backends)
- [MPC backends](#mpc-backends)
- [Built-in profiles](#built-in-profiles)
- [Measuring costs on your machine](#measuring-costs-on-your-machine)
- [Checking a machine](#checking-a-machine)
- [Known limitations](#known-limitations)

## Choosing a backend pair

The backends are chosen with `he=` and `mpc=` in `cf.compile` (`--he`, `--mpc` on the command line):
- `"auto"`, the default, plans with every pair that fits and returns the fastest plan. Plans whose HE↔MPC
  conversions are priced with an unhardened protocol (today the NEXUS profiles) rank after all others.
- `plan.alternatives` lists the other pairs: their latency, `unhardened_conversions` where it applies, and
  why any were not used.
- `security=` sets which MPC security models are acceptable (see [MPC backends](#mpc-backends)).
- `min_precision=` sets the minimum HE bootstrapping precision in bits.

`cipherflow backends --model bert --hardware cpu --security 2pc-dealer` shows which pairs fit a deployment
without planning.

## HE backends

| HE backend | status | security | bootstrapping (one ciphertext, 32768 values, CPU) |
|---|---|---|---|
| `openfhe-fast` | measured: all ten operators run on one CKKS parameter set ([`native/openfhe`](../native/openfhe)) | 128-bit | 8 levels after, 10.5 bits, 17.6 s |
| `openfhe-precise` | measured, two-pass bootstrapping | 128-bit | 7 levels after, 23.5 bits, 34.8 s |
| `nexus` (SEAL / Phantom) | costs are NEXUS reference kernels scaled to each shape; its operators use three parameter sets, so they do not run as one pipeline | - | - |

Precision is -log2 of the largest bootstrapping error on random values in [-1, 1].

The OpenFHE kernels:
- pack 128-token blocks; other sequence lengths are kept out of HE;
- approximate GeLU, LayerNorm and Softmax with Chebyshev polynomials on fixed input ranges, checked against
  a calibration (see [Running a plan on real data](user-guide.md#running-a-plan-on-real-data));
- convert between HE and MPC with a hardened protocol for values below 2^12: statistically hiding masks on
  every slot, exact decoding, and a wrap-free MPC → HE direction (see
  [`native/openfhe/CONVERSION.md`](../native/openfhe/CONVERSION.md)).

An HE → MPC conversion needs 2 levels left on the ciphertext. An MPC → HE conversion costs about 2.8 s of
server time and 22 MB per ciphertext of 32768 values, for the wrap removal.

## MPC backends

| MPC backend | security model | devices |
|---|---|---|
| `sci` (SCI/BOLT, EzPC) | `2pc`: two parties, no third party | CPU |
| `shaft` (SHAFT, CrypTen) | `2pc-dealer`: Beaver triples from a trusted dealer; the offline phase is not costed | CPU, GPU |

`security="2pc"` (the default) allows only `sci`. `"2pc-dealer"` also allows `shaft`. `"any"` allows
everything, including profiles from unregistered backends. If no pair fits, `compile` raises
`UnsupportedBackend` with the reason for each one.

## Built-in profiles

Built-in profiles are named `<hardware>-<he>-<mpc>`:
- `cpu-openfhe-{fast,precise}-{sci,shaft}` are measured end to end on one machine: HE operators and
  bootstrapping with OpenFHE, MPC operators in two-party runs, and conversions with real encryption.
- `cpu-nexus-sci`, `cpu-nexus-shaft` and `gpu-nexus-shaft` have measured MPC and conversions, but NEXUS
  reference-scaled HE.
- `paper-cpu` / `paper-gpu` are the profiles used in the paper.

`cf.Deployment.of("cpu", ...)` and `"gpu"` select every built-in profile for that hardware, one per backend
pair. The CPU profiles were measured on an AMD Threadripper PRO 3955WX. The GPU profile is for two RTX 3090s:
SHAFT was measured there, and the NEXUS-CUDA costs are scaled from reference kernels. `cipherflow profiles`
lists them all.

The raw measurements and the script that builds the profiles are in `compiler/state_expanded/profiles/`
(`raw/`, `build_measured.py`).

## Measuring costs on your machine

Costs depend on the machine, and the best plan can change with them. On CPU, CipherFlow can measure the costs
a model needs on the machine that will run it:

```python
dep  = cf.Deployment.local("100Mbps/20ms", device="cpu")   # the target is the machine running this code
plan = cf.compile(graph, dep)                               # profile="auto": measures only what is missing
```

On the command line: `cipherflow compile ... --hardware local`. `cipherflow profile --model bert --device cpu
--dry-run` lists what this machine still needs to measure.

`profile="auto"` works out which cost entries the graph needs, loads this machine's cache, measures only the
missing entries with the installed backends, and saves them. The cache is `$CIPHERFLOW_CACHE`, or
`~/.cache/cipherflow/<machine>-<device>/`. MicroBench does not run on GPU yet; on GPU, pass a measured profile
file.

Measuring runs the backends, so they must be built first:

| backend | build / location |
|---|---|
| OpenFHE kernels (`openfhe-*`) | OpenFHE 1.4: `cmake -S native/openfhe -B native/openfhe/build && cmake --build native/openfhe/build` |
| conversions for `nexus` | SEAL 4.1: `cmake -S native/conversion -B native/conversion/build && cmake --build native/conversion/build` |
| `nexus` reference kernels | NEXUS build: `CIPHERFLOW_NEXUS_BUILD` |
| `sci` | SCI (EzPC) bridges: `CIPHERFLOW_SCI_BIN`; their sources are not in this repository yet |
| `shaft` | SHAFT (CrypTen fork): `CIPHERFLOW_SHAFT_ROOT` |

## Checking a machine

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

## Known limitations

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

[ROADMAP.md](../ROADMAP.md) lists what will lift these limits.

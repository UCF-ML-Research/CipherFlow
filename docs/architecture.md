# How CipherFlow works

CipherFlow plans against an abstraction of the backends. A backend is a set of *methods*. Each method is one
operator in one domain, with:
- the CKKS levels it consumes;
- the shapes and value ranges it accepts;
- its measured cost.

The backend also supplies the measured cost of its HE↔MPC conversions and bootstrapping. A library plugs in
by registering its methods and a cost profile, which MicroBench can measure on your machine.

<img alt="CipherFlow in the private-inference stack. Model layer: a PyTorch model, traced by cf.models.from_torch into an operator graph (operators, tensor shapes, residual edges). Deployment: hardware profile or this machine, network given or measured with --ip, security model. Planning layer (CipherFlow): MicroBench, state expansion over domains and CKKS levels, BlockFold; cf.compile returns plan.json. Backend layer: HE (OpenFHE, NEXUS) and MPC (SCI, SHAFT) report methods and measured costs and receive the plan; HE-MPC conversions and bootstrapping. Execution: cipherflow.runtime checks the plan and emulates it on real inputs; encrypted two-party execution is planned." src="assets/cipherflow-ecosystem.svg" width="100%">

- [Inputs and outputs](#inputs-and-outputs)
- [How the planner decides](#how-the-planner-decides)
- [Repository layout](#repository-layout)

## Inputs and outputs

| | interface | contents |
|---|---|---|
| ① input | operator graph `G = (V, E)` | operators, tensor shapes and residual edges, traced from a PyTorch `nn.Module` by `cf.models.from_torch` (or built in: `cf.models.bert()`, `vit()`) |
| ② input | deployment `D = (HW, BW, RTT)` | hardware profile (`cpu`, `gpu`, or measured by MicroBench); network (`--network 10Mbps/1ms`, or measured with `--ip SERVER`); security model (`2pc`, `2pc-dealer`) |
| ③ input | backend methods + measured costs | per operator and domain: time, bytes, rounds, CKKS levels, contracts; costs of HE↔MPC conversions and bootstrapping |
| optional input | calibration | value ranges on real inputs (`cf.runtime.calibrate`), so operators stay out of HE where an approximation would not hold |
| ④ output | `plan.json` | per operator: HE or MPC, CKKS levels, kernel; per edge: conversions and bootstraps; latency and communication |

The [plan format](../cipherflow/README.md#plan-format) describes `plan.json`.

## How the planner decides

The planner rests on three ideas:
- **MicroBench: costs measured where you deploy.** Every operator, conversion and bootstrap is timed on the
  target hardware, with its bytes and rounds, so the plan follows measured costs instead of a fixed rule.
- **State expansion: one search over domains and levels.** Each operator's state is its domain and, in HE,
  the CKKS levels left. Placement, conversions and bootstraps become one shortest-path search over these
  states.
- **BlockFold: residual blocks folded into steps.** Each residual block is a single-entry, single-exit region.
  Folding it into one step contains the growth from branching joins and keeps planning tractable on
  Transformer-scale graphs.

`cf.compile` plans with every pair of HE and MPC backends that fits the deployment (see
[Choosing a backend pair](backends.md#choosing-a-backend-pair)). By default (`safe=True`) it also scores five
static placements under the same cost model: BumbleBee-, BOLT- and BLB-style rules, all-HE and all-MPC. It
then returns the cheapest plan.

## Repository layout

| path | contents |
|---|---|
| `cipherflow/` | public API: `compile`, `Deployment`, `Plan`, PyTorch front end, runtime (checks, emulation, calibration), network measurement, CLI |
| `compiler/state_expanded/` | planner: state-expanded planning, BlockFold (SESE), cost model, MicroBench profilers, measured profiles |
| `ir/` | operator graph types |
| `backends/` | HE (NEXUS) adapters used by the MicroBench profilers |
| `native/` | measured kernels: OpenFHE HE operators, bootstrapping and conversions (`native/openfhe`), SEAL conversions (`native/conversion`) |
| `demo/` | the demo: PyTorch Transformers in, plans operator by operator out (script, web page builder, terminal recordings) |
| `docs/` | documentation (user guide, backends, this page, FAQ, tutorials); the [website](https://ucf-ml-research.github.io/CipherFlow/) and the plan visualization (`docs/planner/`), served by GitHub Pages and built by `demo/web`; figures and terminal recordings (`docs/assets/`) |

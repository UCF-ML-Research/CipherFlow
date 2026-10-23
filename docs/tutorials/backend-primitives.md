# Tutorial: adding new backend primitives

CipherFlow plans with the costs of the primitives it knows. You can bring your own primitives, and how you do it
depends on what you have:

| you have | use | CipherFlow then |
|---|---|---|
| a kernel for a whole operator, e.g. an HE Softmax | `cf.Extension(...).primitive(...)` | uses your kernel for that operator, at your measured cost |
| only a library's basic operations: HE add, multiply, rotate, bootstrap; MPC multiply, compare, truncate | `cf.PrimitiveBackend` | writes every operator as calls to them and prices it |
| a new operator, i.e. a PyTorch module CipherFlow does not know | `Extension.operator` + `primitive` | traces the module as one operator and plans it |

Run the code below from the repository root, so that `demo` imports. It plans the BERT-base encoder of the demo:

```python
import torch
import cipherflow as cf
from demo.transformer import bert

graph = cf.models.from_torch(bert(), torch.zeros(1, 128, 768))
deployment = cf.Deployment.of("cpu", "10Mbps/1ms")
builtin = cf.compile(graph, deployment, he="openfhe-fast", mpc="sci")       # to compare with
```

## Step 1: measure one call

The planner needs a cost per call, in milliseconds of local compute, bytes sent and rounds of communication.
What one call means depends on the primitive:

| primitive | one call | measure |
|---|---|---|
| HE kernel for an operator | the operator on its input shape, e.g. scores `[1, 12, 128, 128]` | time; CKKS levels consumed (input level − output level) |
| HE basic operation | one operation on one ciphertext, at the parameters you deploy | time per add, plaintext multiply, ciphertext multiply, rotation; time of one bootstrap; slots per ciphertext; levels after bootstrapping |
| MPC kernel or basic operation | one call over many elements, run by both parties | time and bytes per element (divide by the number of elements); rounds per call |

## Step 2a: a kernel for a whole operator

Declare the operator, the domain, the levels a call consumes, its cost, and optionally the inputs it accepts.
The primitive replaces the backend's own kernel for that operator in that domain:

```python
ext = cf.Extension("my-he-softmax")
ext.primitive(
    "Softmax", "HE",
    levels=4,                                    # CKKS levels one call consumes
    cost=cf.Cost(local_ms=31_000),               # one call on scores [1, 12, 128, 128], measured
    accepts=lambda node: None if node.input_shape[-1] == 128 else "rows of 128 scores only",
)
plan = cf.compile(graph, deployment, he="openfhe-fast", mpc="sci", extensions=[ext])
print(f"{builtin.cost['total_ms'] / 1e3:.0f} s -> {plan.cost['total_ms'] / 1e3:.0f} s")   # 25263 s -> 13676 s
```

The cost can also depend on the node's shape. Pass a function `(node, profile) -> cf.Cost`:

```python
def per_ciphertext(ms):                          # a kernel that takes `ms` per ciphertext of 32768 values
    return lambda node, profile: cf.Cost(local_ms=ms * -(-cf.extensions.numel(node.input_shape) // 32768))

ext = cf.Extension("my-he-softmax").primitive("Softmax", "HE", levels=4, cost=per_ciphertext(5_200))
```

`cf.extensions.like(op, domain, scale)` prices a node from the profile's measured record of another operator.
For example, `like("Softmax", "HE", lambda node, record: 0.4)` is 0.4 times the built-in HE Softmax.

## Step 2b: a backend from its basic operations

Declare the operations and their costs. CipherFlow lowers every operator to them:
- linear layers as BSGS matrix-vector products;
- Softmax, LayerNorm and GeLU as polynomials and Newton steps.

It counts the calls and CKKS levels, and prices each operator. The operations have fixed names:
- **HE:** `add`, `mul_plain`, `mul`, `rotate`, plus `bootstrap=`. Each is priced per call on one ciphertext of
  `slots` values; `levels` is the level budget after bootstrapping.
- **MPC:** `add`, `mul`, `matmul`, `trunc`, `cmp`. Each is priced per element (time and bytes), with rounds per call.
  `matmul` is optional: without it, linear layers use `mul`.


```python
my_ckks = cf.PrimitiveBackend.he(
    "my-ckks", slots=32768, levels=10,
    bootstrap=cf.Cost(local_ms=9_000),                        # one ciphertext
    primitives={"add": cf.Cost(0.3), "mul_plain": cf.Cost(3),
                "mul": cf.Cost(40), "rotate": cf.Cost(35)})  # one call on one ciphertext
my_2pc = cf.PrimitiveBackend.mpc(
    "my-2pc",
    primitives={"mul": cf.Cost(3e-4, 128, 1),                 # per element: ms, bytes; rounds per call
                "matmul": cf.Cost(2e-5, 8, 2),                # per multiply-accumulate with the server's weights
                "trunc": cf.Cost(2e-4, 128, 1),
                "cmp": cf.Cost(2e-3, 4000, 16)})

print(my_ckks.lower(graph))           # each operator as primitive calls, with its levels and cost
plan = cf.compile(graph, deployment, he="openfhe-fast", mpc="sci",
                  extensions=[my_ckks.extension(), my_2pc.extension()])
```

What the lowering does with the backend's parameters:
- **Level budget and bootstrapping.** An HE backend brings its own; a kernel that needs more levels than the budget
  bootstraps inside.
- **Missing operations.** An operator whose kernel needs an operation the backend lacks (e.g. `rotate`) stays out
  of that domain. `lower(graph)` says which operation is missing.
- **Kernel parameters.** Polynomial degrees and iteration counts can be changed with `params=`
  (see `cipherflow.lowering.DEFAULT_PARAMS`).

## Step 2c: a new operator

Register the PyTorch module class as an operator and give it at least one primitive. Pass the extension both to
`from_torch` and to `compile`:

```python
from torch import nn

class TopK(nn.Module):                    # scores -> indices of the k largest, along the last dimension
    def __init__(self, k):
        super().__init__()
        self.k = k
    def forward(self, scores):
        return torch.topk(scores, self.k, dim=-1).indices

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.head, self.topk = bert(2), nn.Linear(768, 768), TopK(4)
    def forward(self, x):
        return self.topk(self.head(self.encoder(x)))

topk = cf.Extension("my-topk")
topk.operator(TopK, "TopK")
topk.primitive("TopK", "MPC", cost=cf.Cost(local_ms=120, comm_bytes=8_000_000, comm_rounds=40))
ranker = cf.models.from_torch(Ranker(), torch.zeros(1, 128, 768), extensions=[topk])
plan = cf.compile(ranker, deployment, he="openfhe-fast", mpc="sci", extensions=[topk])
```

A node can run only in the domains its primitives cover. `TopK` has no HE primitive (CKKS has no comparison), so
it runs in MPC.

## Step 3: read the plan

There are three places to see how your primitives were used:
- **The step table.** `plan.explain(steps=True)` tags every operator that your primitives run, e.g.
  `Softmax [my-he-softmax]`.
- **The nodes.** `node.backend` is the extension's name, and `node.cost_source` is `"extension"`.
- **The warnings.** `plan.warnings` names the operators that an extension priced with its own costs.

Compare `plan.cost["total_ms"]` and `plan.counts()` with the built-in plan to see what the primitives changed.

[`demo/extensions`](../../demo/extensions) runs all three paths:
- `python -m demo.extensions.new_he_primitive` (2a);
- `python -m demo.extensions.primitive_backend` (2b);
- `python -m demo.extensions.private_rag` (2c).

## Step 4: check it on real data

`cipherflow.runtime` runs a plan on real inputs at plaintext speed, each operator in its planned domain (see
[Running a plan on real data](../user-guide.md#running-a-plan-on-real-data)). Pass your extension, so that the
model traces to the plan's graph:

```python
cal  = cf.runtime.calibrate(model, real_inputs, extensions=[ext])     # value ranges on real data
plan = cf.compile(graph, deployment, extensions=[ext], calibration=cal)
res  = cf.runtime.emulate(plan, model, x, extensions=[ext])
print(res.summary())
```

A new operator runs as its PyTorch module in the domain the plan gives it: in fixed point in MPC, exactly in HE,
with the plan's conversions and bootstraps on its edges. `python -m demo.extensions.private_rag_end_to_end` does
this for private RAG retrieval with a real model (e5-base-v2) and real documents (SciFact): the private top-10
equals the plaintext model's for 298 of the 300 test queries.

## Limits

CipherFlow plans with the costs and levels you declare, as given. The emulator runs a new operator as its
PyTorch code, not your kernels' own approximations; CipherFlow does not check your primitives' precision or
security, and does not execute extensions under encryption.
[ROADMAP.md](../../ROADMAP.md#extensions) lists these limits and the work that lifts them.

See the [API reference](../../cipherflow/README.md#extensions) for every argument.

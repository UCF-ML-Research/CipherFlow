# Extensions

An extension extends both the backend layer and the planner's operators. It brings two things:

- **Primitives.** A primitive is one implementation of an operator in one domain (HE or MPC). It declares
  what the planner needs: the CKKS levels it consumes, the inputs it accepts, and its cost.
- **Operators.** An operator is a PyTorch module class that `from_torch` traces as a single operator.

```python
import cipherflow as cf

ext = cf.Extension("my-he-softmax")
ext.primitive("Softmax", "HE", levels=4, cost=cf.Cost(local_ms=9_000),          # one call, measured
              accepts=lambda node: None if node.input_shape[-1] == 128 else "rows of 128 scores only")

rag = cf.Extension("private-rag")
rag.operator(TopK, "TopK")                                                        # nn.Module class -> operator
rag.primitive("TopK", "MPC", cost=cf.Cost(local_ms=800, comm_bytes=40_000_000, comm_rounds=200))

graph = cf.models.from_torch(model, x, extensions=[rag])
plan = cf.compile(graph, deployment, extensions=[ext, rag])
```

How the planner treats a primitive depends on the operator:
- **An operator the backend pair already implements in that domain:** the primitive replaces the built-in
  implementation, including its kernel variants.
- **A new operator:** the primitive is the operator's only implementation in that domain. A node can run only
  in the domains its primitives cover.

`accepts` returns `None` when the primitive can run a node, or the reason it cannot. A node it rejects stays
out of that domain.

A cost can be either of these:
- a fixed `cf.Cost(local_ms, comm_bytes, comm_rounds)` per call;
- a function `(node, profile) -> cf.Cost`. `cf.extensions.like(op, domain, scale)` prices a node from the
  profile's measured record of another operator.

Plans mark what an extension runs:
- `node.backend` is the extension's name;
- `explain(steps=True)` tags the operator, e.g. `Softmax [my-he-softmax]`;
- a warning says which operators the extension priced with its own costs.

## A new HE primitive

```bash
python -m demo.extensions.new_he_primitive                     # --speedup 2.5 --levels 4 by default
```

The demo plugs in a new HE Softmax kernel and plans BERT-base again. The kernel is hypothetical: 2.5× faster
than OpenFHE's built-in HE Softmax, using 4 CKKS levels. At 10 Mbps the built-in plan keeps Softmax in MPC. With
the new kernel, the whole model runs in HE:

```
BERT-base (12 blocks), CPU, 10Mbps/1ms, OpenFHE (fast) + SCI
  built-in HE Softmax:     25263 s   Softmax in MPC (sci); 132 HE / 12 MPC operators, 24 conversions, 59 bootstraps
  with my-he-softmax:      13431 s   Softmax in HE (my-he-softmax); 144 HE / 0 MPC operators, 0 conversions, 83 bootstraps
  speedup of the whole plan: 1.88x
```

For a real kernel, pass its measured cost and its level use; `--speedup` and `--levels` show how the plan
responds to both.

## A backend from its primitives

```bash
python -m demo.extensions.primitive_backend                     # --network 100Mbps/20ms --blocks 12
```

Here the user gives only a backend's primitives and the cost of one call of each. CipherFlow writes every
operator as a program over those primitives (a kernel template), counts the calls and CKKS levels, prices
the operator, and plans with the backend as an extension:

```python
my_ckks = cf.PrimitiveBackend.he("my-ckks", slots=32768, levels=10, bootstrap=cf.Cost(9_000),
                                 primitives={"add": cf.Cost(0.3), "mul_plain": cf.Cost(3),
                                             "mul": cf.Cost(40), "rotate": cf.Cost(35)})
my_2pc = cf.PrimitiveBackend.mpc("my-2pc", primitives={"mul": cf.Cost(3e-4, 128, 1), "matmul": cf.Cost(2e-5, 8, 2),
                                                       "trunc": cf.Cost(2e-4, 128, 1), "cmp": cf.Cost(2e-3, 4000, 16)})
print(my_ckks.lower(graph))                    # every operator as primitive calls, levels and cost
plan = cf.compile(graph, deployment, extensions=[my_ckks.extension(), my_2pc.extension()])
```

The two kinds of backend are priced differently:
- **HE primitives** cost one call on a ciphertext of `slots` values. The backend also brings its level budget
  and its bootstrapping, which replace the profile's.
- **MPC primitives** cost one element (local time and bytes) and a number of rounds per call; a call works on
  all its elements at once. `matmul` (shares × the server's weights) is optional; without it, linear layers
  use `mul`.

| operator | HE kernel | MPC kernel |
|---|---|---|
| linear layers | BSGS diagonal matrix-vector product (plaintext multiplications, rotations) | `matmul` (or `mul`) per multiply-accumulate, truncation |
| QKᵀ, Attn·V | SIMD ciphertext products, rotations to align and sum | `mul` per multiply-accumulate, truncation |
| Softmax | exp by squarings, row sums by rotation, 1/x by Newton steps | max by a comparison tree, exp by squarings, 1/x by Newton steps |
| LayerNorm | mean and variance by rotation, 1/√x start polynomial and Newton steps | variance, 1/√x by Newton steps |
| GeLU | degree-27 polynomial (Paterson–Stockmeyer) | two comparisons, degree-6 polynomial on the middle piece |
| residual add | addition | local |

The kernels' parameters (polynomial degrees, iteration counts) can be changed with `params=`. When a
kernel needs more levels than the budget, it bootstraps inside. When a backend lacks a primitive that a
kernel needs (e.g. rotation), the operator stays out of that domain.

With the illustrative costs of the demo, the two new backends share BERT-base at 100 Mbps:
- linear layers, attention products and GeLU in `my-ckks`;
- Softmax and LayerNorm in `my-2pc`.

At 1 Gbps everything moves to `my-2pc`.

Prototype limits:
- the kernels are cost models of standard constructions on one simple packing, not code that runs;
- HE↔MPC conversions keep the costs of the profile planned with;
- whether the primitives are precise enough and secure is the user's to ensure.

## A new workload: private RAG retrieval

```bash
python -m demo.extensions.private_rag                           # --docs 8192 --k 8 --network 100Mbps/20ms
```

A BERT encoder (PyTorch) encodes the client's question. Three new operators then do the retrieval:
- `MeanPool` pools the encoding into one query vector;
- `DocScores` scores the query against the server's private document embeddings;
- `TopK` keeps the best k.

The client learns only the indices of those k documents. The extension implements the operators as follows:

| operator | HE | MPC |
|---|---|---|
| `MeanPool` | rotations and additions, 1 level | local on secret shares |
| `DocScores` | plaintext document matrix × encrypted query, 1 level | matrix product on shares |
| `TopK` | — (CKKS has no comparisons) | comparisons |

CipherFlow places each operator. Here the document scores run in HE, and top-k runs in MPC:

```
   144  LayerNorm               MPC               encoder.blocks.11.ln2
   145  MeanPool [private-rag]  MPC               pool
          convert MPC -> HE  [1, 768]  (#145 -> #146)  0.13 s
   146  DocScores [private-rag] HE     8 -> 7     scores
          convert HE -> MPC  [1, 8192]  (#146 -> #147)  0.19 s
   147  TopK [private-rag]      MPC               topk
```

The costs are estimates scaled from the profile's measured operators (`rag_extension()` states each one).
Replace them with measurements of your kernels and protocols.

Other workloads follow the same pattern. A router (a classifier head and an argmax that picks a model) or a
database task (comparisons and aggregations over encrypted or shared records) needs:
1. its operators as PyTorch modules;
2. an `Extension` that registers them with `operator()`;
3. a `primitive()` per operator and domain.

## Limits

- Extensions are planned, not executed. `cipherflow.runtime` emulates the built-in operators only.
- `profile="auto"` (MicroBench) does not measure extension operators. Use a built-in or saved profile; the
  extension prices its own operators.
- Graphs must still be a chain of two-branch residual regions (`cf.check_graph`).

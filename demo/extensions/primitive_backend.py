"""Backends from their primitives: give the primitives and their costs, CipherFlow lowers every operator.

A new HE library is declared by its CKKS primitives (addition, plaintext and ciphertext
multiplication, rotation, bootstrapping) with the cost of one call, its slots and its level budget;
a new MPC protocol by its primitives (multiplication, products with the server's weights,
truncation, comparison) with the cost per element and the rounds per call. CipherFlow writes each
operator of the model as a program over those primitives, prices it, and plans with the backend.

    python -m demo.extensions.primitive_backend
    python -m demo.extensions.primitive_backend --network 1Gbps/1ms --blocks 2

The primitive costs below are illustrative orders of magnitude, not measurements of a library.
"""
from __future__ import annotations

import argparse
import warnings

import torch

import cipherflow as cf
from demo import transformer

# one call on a ciphertext of 32768 values (CKKS, N = 2^16, multi-threaded CPU)
MY_CKKS = cf.PrimitiveBackend.he(
    "my-ckks", slots=32768, levels=10, bootstrap=cf.Cost(9_000),
    primitives={"add": cf.Cost(0.3), "mul_plain": cf.Cost(3), "mul": cf.Cost(40), "rotate": cf.Cost(35)})

# per element: local time and bytes sent; rounds per call (a call works on all elements at once)
MY_2PC = cf.PrimitiveBackend.mpc(
    "my-2pc",
    primitives={"mul": cf.Cost(3e-4, 128, 1), "matmul": cf.Cost(2e-5, 8, 2),
                "trunc": cf.Cost(2e-4, 128, 1), "cmp": cf.Cost(2e-3, 4000, 16)})


def summary(plan) -> str:
    k = plan.counts()
    used = sorted({n.backend for n in plan.nodes if n.backend != "client"})
    return (f"{plan.cost['total_ms'] / 1e3:8.0f} s   {k['HE']:3d} HE / {k['MPC']:3d} MPC, {k['conversions']} conversions, "
            f"{k['bootstraps']} bootstraps   ({', '.join(used)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--network", default="100Mbps/20ms")
    a = ap.parse_args(argv)

    graph = cf.models.from_torch(transformer.bert(a.blocks), torch.zeros(1, 128, 768))
    block = cf.models.from_torch(transformer.bert(1), torch.zeros(1, 128, 768))
    print(MY_CKKS.lower(block), "\n")

    dep = cf.Deployment.of("cpu", a.network)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        plans = {
            "OpenFHE (fast) + SCI, measured": cf.compile(graph, dep, he="openfhe-fast", mpc="sci"),
            "my-ckks + SCI": cf.compile(graph, dep, he="openfhe-fast", mpc="sci", extensions=[MY_CKKS.extension()]),
            "my-ckks + my-2pc": cf.compile(graph, dep, he="openfhe-fast", mpc="sci",
                                           extensions=[MY_CKKS.extension(), MY_2PC.extension()]),
        }
    print(f"BERT-base ({a.blocks} blocks), CPU, {a.network}")
    for name, plan in plans.items():
        print(f"  {name:32s} {summary(plan)}")
    both = plans["my-ckks + my-2pc"]
    where = {}
    for node in both.nodes:
        if node.op_type != "Input":
            where.setdefault(node.op_type, set()).add(node.domain)
    print("\n  with my-ckks + my-2pc, each operator runs in: "
          + ", ".join(f"{op} {'/'.join(sorted(d))}" for op, d in where.items()))
    print("\nHE<->MPC conversions keep the costs of the profile planned with (OpenFHE + SCI). A conversion to MPC"
          "\nand back also refreshes a ciphertext's levels, so a plan can convert where it is cheaper than bootstrapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

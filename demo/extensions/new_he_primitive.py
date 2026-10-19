"""Bring your own HE primitive: plug a new HE kernel into the planner and see the plan change.

A primitive is one implementation of an operator in one domain. Here it is a new HE Softmax
kernel, declared with what the planner needs: the CKKS levels it consumes, the inputs it
accepts, and its cost. It replaces the backend's built-in HE Softmax; CipherFlow then decides
again, operator by operator, where Softmax and everything around it should run.

    python -m demo.extensions.new_he_primitive
    python -m demo.extensions.new_he_primitive --speedup 1.5 --levels 6 --network 1Gbps/20ms

The kernel here is hypothetical: its cost is the built-in OpenFHE Softmax's measured cost divided
by --speedup. For a real kernel, give its measured cost (cf.Cost per call, or a function of the
node's shapes) and its level use.
"""
from __future__ import annotations

import argparse
import warnings

import torch

import cipherflow as cf
from demo import transformer


def softmax_extension(speedup: float, levels: int) -> cf.Extension:
    ext = cf.Extension("my-he-softmax")
    ext.primitive(
        "Softmax", "HE",
        levels=levels,                                           # CKKS levels the kernel consumes
        accepts=lambda node: None if node.input_shape[-1] == 128 else "the kernel packs rows of 128 scores",
        # its cost per call; a measured kernel would give cf.Cost(local_ms=..., comm_bytes=0, comm_rounds=0)
        cost=cf.extensions.like("Softmax", "HE", lambda node, record: 1.0 / speedup),
    )
    return ext


def summary(plan) -> str:
    where = sorted({f"{n.domain} ({n.backend})" for n in plan.nodes if n.op_type == "Softmax"})
    k = plan.counts()
    return (f"{plan.cost['total_ms'] / 1e3:8.0f} s   Softmax in {', '.join(where)}; {k['HE']} HE / {k['MPC']} MPC "
            f"operators, {k['conversions']} conversions, {k['bootstraps']} bootstraps")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speedup", type=float, default=2.5, help="the new kernel vs. the built-in HE Softmax")
    ap.add_argument("--levels", type=int, default=4, help="CKKS levels the new kernel consumes")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--network", default="10Mbps/1ms")
    a = ap.parse_args(argv)

    graph = cf.models.from_torch(transformer.bert(a.blocks), torch.zeros(1, 128, 768))
    dep = cf.Deployment.of("cpu", a.network)
    ext = softmax_extension(a.speedup, a.levels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        before = cf.compile(graph, dep, he="openfhe-fast", mpc="sci")
        after = cf.compile(graph, dep, he="openfhe-fast", mpc="sci", extensions=[ext])

    print(f"BERT-base ({a.blocks} blocks), CPU, {a.network}, OpenFHE (fast) + SCI")
    print(f"  built-in HE Softmax:  {summary(before)}")
    print(f"  with my-he-softmax:   {summary(after)}")
    print(f"  speedup of the whole plan: {before.cost['total_ms'] / after.cost['total_ms']:.2f}x\n")
    print("  block 1 with the new primitive:")
    rows = after.explain_steps().splitlines()
    end = next(i for i, ln in enumerate(rows) if "blocks.1." in ln)
    print("\n".join(rows[:end]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

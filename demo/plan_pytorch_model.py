"""The demo in Python: a Transformer written in PyTorch in, CipherFlow's plan out.

    python -m demo.plan_pytorch_model                                   # BERT-base, CPU, 10 Mbps / 1 ms
    python -m demo.plan_pytorch_model --model vit --network 1Gbps/40ms
    python -m demo.plan_pytorch_model --blocks 2 -o plan.json
    # the network measured to the other party (there: cipherflow netbench --serve) instead of given
    python -m demo.plan_pytorch_model --ip SERVER_IP
    # the web demo's setting: the paper's GPU profile (NEXUS + SHAFT with a trusted dealer)
    python -m demo.plan_pytorch_model --hardware gpu --profile paper-gpu --security 2pc-dealer --no-qkv --network 1Gbps/40ms

Input: the PyTorch model (demo/transformer.py) and a deployment (hardware, bandwidth, round-trip
time). Output: the plan operator by operator -- HE or MPC, the CKKS levels, the conversions and
bootstraps before each operator, and where each operator is in the PyTorch code -- and plan.json.
"""
from __future__ import annotations

import argparse
import warnings

import torch

import cipherflow as cf
from demo import transformer

INPUT_SHAPE = {"bert": (1, 128, 768), "vit": (1, 197, 768)}     # [batch, tokens, hidden]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=sorted(INPUT_SHAPE), default="bert")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--hardware", default="cpu", help="cpu | gpu, or a built-in profile name")
    ap.add_argument("--profile", default=None, help="e.g. paper-gpu (default: the hardware's built-in profiles)")
    ap.add_argument("--security", default="2pc", choices=["2pc", "2pc-dealer", "any"])
    ap.add_argument("--network", default="10Mbps/1ms", help="BANDWIDTH/RTT")
    ap.add_argument("--peer", "--ip", dest="peer", default=None, metavar="HOST[:PORT]",
                    help="measure the network to `cipherflow netbench --serve` on HOST instead of --network")
    ap.add_argument("--no-qkv", action="store_true", help="leave out the Q/K/V projections (the paper's profiles)")
    ap.add_argument("-o", "--output", default=None, help="write plan.json")
    a = ap.parse_args(argv)

    # input 1: the model, written in PyTorch; CipherFlow traces it into its operators
    model = getattr(transformer, a.model)(a.blocks)
    graph = cf.models.from_torch(model, torch.zeros(INPUT_SHAPE[a.model]), qkv_projection=not a.no_qkv)

    # input 2: the deployment; the network given, or measured to the other party
    measured = None
    if a.peer:
        from cipherflow.netbench import measure
        measured = measure(a.peer)
        print(measured.summary())
    deployment = cf.Deployment.of(a.hardware, measured.network() if measured else a.network)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)       # listed in the plan's warnings
        plan = cf.compile(graph, deployment, profile=a.profile, security=a.security)
    if measured:
        plan.deployment["network_measured"] = measured.to_dict()

    # output: the plan, operator by operator (plan.steps() has the same as data)
    print(plan.explain(steps=True))
    if a.output:
        plan.to_json(a.output)
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

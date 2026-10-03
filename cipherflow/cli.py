"""Command-line interface: `cipherflow compile | explain | profiles`."""
from __future__ import annotations

import argparse
import sys
import warnings

from . import models, profiles
from .deploy import Deployment, Network
from .errors import CostExtrapolationWarning
from .plan import Plan
from .planning import compile as cf_compile


def _graph(a):
    return getattr(models, a.model)(num_blocks=a.blocks, **({"seq": a.seq} if a.model == "bert" else {"tokens": a.seq}))


def _cmd_compile(a) -> int:
    graph = _graph(a)
    if a.hardware == "local":
        dep = Deployment.local(Network.parse(a.network), device=a.device or "")
        prof = a.profile or "auto"
    else:
        dep = Deployment.of(a.hardware, Network.parse(a.network))
        prof = a.profile or a.hardware
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostExtrapolationWarning)   # explain() prints them
        plan = cf_compile(graph, dep, profile=prof, strict=a.strict)
    print(plan.explain())
    if a.output:
        plan.to_json(a.output)
        print(f"wrote {a.output}")
    return 0


def _cmd_explain(a) -> int:
    print(Plan.from_json(a.plan).explain())
    return 0


def _cmd_profile(a) -> int:
    from .measure import cache_dir, local_device, microbench
    dev = a.device or local_device()
    res = microbench(_graph(a), device=dev, dry_run=a.dry_run)
    if a.dry_run:
        print(f"cache: {cache_dir(dev)}")
        print(f"missing: {sum(d == 'HE' for _, d, _, _ in res.ops)} HE ops, "
              f"{sum(d == 'MPC' for _, d, _, _ in res.ops)} MPC ops, "
              f"{len(res.conversion_shapes)} conversion shapes, bootstrap={'yes' if res.bootstrap else 'no'}")
    return 0


def _cmd_profiles(a) -> int:
    for k, v in profiles.list_builtin().items():
        print(f"{k:6s} {v}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cipherflow", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="plan a model for a deployment")
    c.add_argument("--model", choices=["bert", "vit", "vit_paper"], default="bert",
                   help="vit: pre-LN ViT-B/16; vit_paper: the BERT-style ViT graph used in the paper")
    c.add_argument("--blocks", type=int, default=12)
    c.add_argument("--seq", type=int, default=None, help="sequence length / tokens (default 128 / 197)")
    c.add_argument("--hardware", default="cpu",
                   help="built-in profile (cpu | gpu), or 'local' to plan for this machine")
    c.add_argument("--device", default=None, help="with --hardware local: cpu | cuda (default: detected)")
    c.add_argument("--profile", default=None, help="profile JSON path or 'auto' (overrides --hardware)")
    c.add_argument("--network", default="100Mbps/20ms", help="BANDWIDTH/RTT, e.g. 10Mbps/1ms")
    c.add_argument("--strict", action="store_true",
                   help="refuse cost entries that are not in the profile instead of scaling them")
    c.add_argument("-o", "--output", default=None, help="write plan.json")
    c.set_defaults(func=_cmd_compile)
    e = sub.add_parser("explain", help="summarize a plan.json")
    e.add_argument("plan")
    e.set_defaults(func=_cmd_explain)
    m = sub.add_parser("profile", help="measure the cost entries a model needs on this machine (MicroBench)")
    m.add_argument("--model", choices=["bert", "vit", "vit_paper"], default="bert")
    m.add_argument("--blocks", type=int, default=12)
    m.add_argument("--seq", type=int, default=None)
    m.add_argument("--device", default=None, help="cpu | cuda (default: detected)")
    m.add_argument("--dry-run", action="store_true", help="only report what is missing from the cache")
    m.set_defaults(func=_cmd_profile)
    p = sub.add_parser("profiles", help="list built-in profiles")
    p.set_defaults(func=_cmd_profiles)
    a = ap.parse_args(argv)
    if getattr(a, "seq", 0) is None:
        a.seq = 128 if a.model == "bert" else 197
    try:
        return a.func(a)
    except ValueError as exc:            # UnsupportedGraph / UnsupportedOperator / CostExtrapolationError ...
        print(f"cipherflow: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

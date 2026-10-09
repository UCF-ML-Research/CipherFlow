"""Command-line interface: `cipherflow compile | explain | backends | profile | profiles | netbench | selfcheck`."""
from __future__ import annotations

import argparse
import sys
import warnings

from . import models, profiles
from .deploy import Deployment, Network
from .errors import CostExtrapolationWarning
from .plan import Plan
from .planning import compile as cf_compile


def _torch_module(spec: str):
    """FILE.py:NAME or package.module:NAME -> the nn.Module (NAME: a module, or a class or
    function that builds one without arguments)."""
    import importlib
    import importlib.util
    from pathlib import Path

    import torch
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise ValueError(f"--torch expects FILE.py:NAME or package.module:NAME, got {spec!r}")
    if path.endswith(".py"):
        if not Path(path).is_file():
            raise ValueError(f"--torch: no such file {path}")
        loader = importlib.util.spec_from_file_location(Path(path).stem, path)
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path)
    obj = getattr(mod, name)
    module = obj if isinstance(obj, torch.nn.Module) else obj()
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"--torch: {spec} is not an nn.Module and does not build one")
    return module.eval()


def _graph(a):
    if getattr(a, "torch", None):
        import torch
        shape = tuple(int(d) for d in a.input_shape.split(","))
        return models.from_torch(_torch_module(a.torch), torch.randn(*shape), qkv_projection=not a.no_qkv)
    return getattr(models, a.model)(num_blocks=a.blocks, **({"seq": a.seq} if a.model.startswith("bert") else {"tokens": a.seq}))


def _network(a):
    """--network as given, or measured to --peer (then printed, and kept in a.measured)."""
    a.measured = None
    if getattr(a, "peer", None):
        from .netbench import measure
        a.measured = measure(a.peer, seconds=a.net_seconds, streams=a.net_streams)
        print(a.measured.summary())
        return a.measured.network()
    return Network.parse(a.network)


def _deployment(a):
    net = _network(a)
    if a.hardware == "local":
        return Deployment.local(net, device=a.device or ""), a.profile or "auto"
    return Deployment.of(a.hardware, net), a.profile or a.hardware


def _cmd_compile(a) -> int:
    graph = _graph(a)
    dep, prof = _deployment(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostExtrapolationWarning)   # explain() prints them
        plan = cf_compile(graph, dep, profile=prof, he=a.he, mpc=a.mpc, security=a.security,
                          min_precision=a.min_precision, strict=a.strict)
    if a.measured is not None:
        plan.deployment["network_measured"] = a.measured.to_dict()
    print(plan.explain(steps=a.steps))
    if a.output:
        plan.to_json(a.output)
        print(f"wrote {a.output}")
    return 0


def _cmd_explain(a) -> int:
    print(Plan.from_json(a.plan).explain(steps=a.steps))
    return 0


def _cmd_backends(a) -> int:
    from .backends import HE_BACKENDS, MPC_BACKENDS, SECURITY_TEXT
    if a.model is None:
        hw = profiles.hardware_profiles()
        print("HE backends")
        for b in HE_BACKENDS.values():
            builtin = sorted(n for t in hw.values() for (h, _m), n in t.items() if h == b.name)
            prec = f"{b.precision_bits:g}-bit bootstrapping" if b.precision_bits else "precision not established"
            print(f"  {b.name:16s} {b.status:17s} {b.security:10s} {prec:27s} profiles: {', '.join(builtin) or '-'}")
            print(f"  {'':16s} {b.params}")
        print("MPC backends")
        for b in MPC_BACKENDS.values():
            builtin = sorted(n for t in hw.values() for (_h, m), n in t.items() if m == b.name)
            print(f"  {b.name:16s} {b.label:17s} {b.security:10s} devices={','.join(b.devices):17s} "
                  f"profiles: {', '.join(builtin) or '-'}")
            print(f"  {'':16s} {SECURITY_TEXT[b.security]}: {b.note}")
        return 0
    from .planning import match
    dep, prof = _deployment(a)
    print(match(_graph(a), dep, profile=prof, he=a.he, mpc=a.mpc, security=a.security,
                min_precision=a.min_precision))
    return 0


def _cmd_profile(a) -> int:
    from .measure import cache_dir, local_device, microbench
    dev = a.device or local_device()
    res = microbench(_graph(a), device=dev, backend=a.backend, he=a.he, dry_run=a.dry_run)
    if a.dry_run:
        print(f"cache: {cache_dir(dev, a.backend, a.he)}")
        print(f"missing: {sum(d == 'HE' for _, d, _, _ in res.ops)} HE ops, "
              f"{sum(d == 'MPC' for _, d, _, _ in res.ops)} MPC ops, "
              f"{len(res.conversion_shapes)} conversion shapes, bootstrap={'yes' if res.bootstrap else 'no'}")
    return 0


def _cmd_netbench(a) -> int:
    from . import netbench
    if a.serve:
        srv = netbench.Server(a.host, a.port)
        print(f"netbench server on {srv.host}:{srv.port}; on the client: cipherflow netbench --ip THIS_HOST:{srv.port} "
              f"(or cipherflow compile ... --ip THIS_HOST:{srv.port}). Ctrl-C stops.", flush=True)
        try:
            srv.serve()
        except KeyboardInterrupt:
            pass
        return 0
    if not a.peer:
        print("cipherflow netbench: give --serve (server side) or --peer HOST[:PORT] (client side)", file=sys.stderr)
        return 2
    print(netbench.measure(a.peer, port=a.port, rounds=a.rounds, seconds=a.seconds, streams=a.streams).summary())
    return 0


def _peer_args(p) -> None:
    p.add_argument("--peer", "--ip", dest="peer", default=None, metavar="HOST[:PORT]",
                   help="measure the network to a `cipherflow netbench --serve` at HOST (the other party) "
                        "and plan with it, instead of --network")
    p.add_argument("--net-seconds", type=float, default=2.0, help="with --peer: seconds of transfer each way")
    p.add_argument("--net-streams", type=int, default=1, help="with --peer: parallel TCP streams")


def _cmd_profiles(a) -> int:
    for k, v in profiles.list_builtin().items():
        print(f"{k:16s} {v}")
    for k, v in profiles.ALIASES.items():
        print(f"{k:16s} = {v}")
    return 0


def _backend_args(p) -> None:
    p.add_argument("--he", default="auto", help="HE backend: auto | nexus | openfhe-fast | openfhe-precise")
    p.add_argument("--min-precision", type=float, default=None,
                   help="minimum HE bootstrapping precision in bits")
    p.add_argument("--mpc", default="auto", help="MPC backend: auto | sci | shaft (default: auto)")
    p.add_argument("--security", default="2pc", choices=["2pc", "2pc-dealer", "any"],
                   help="2pc: two parties, no third party (default); 2pc-dealer: also allow a "
                        "trusted dealer (SHAFT); any")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cipherflow", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="plan a model for a deployment")
    c.add_argument("--model", choices=["bert", "vit", "bert_paper", "vit_paper"], default="bert",
                   help="vit: pre-LN ViT-B/16; bert_paper / vit_paper: the graphs used in the paper "
                        "(no Q/K/V projections; for the paper's profiles)")
    c.add_argument("--blocks", type=int, default=12)
    c.add_argument("--seq", type=int, default=None, help="sequence length / tokens (default 128 / 197)")
    c.add_argument("--torch", default=None, metavar="FILE.py:NAME",
                   help="plan a PyTorch encoder instead of --model: NAME is an nn.Module or builds one, e.g. "
                        "demo/transformer.py:bert (needs PyTorch)")
    c.add_argument("--input-shape", default="1,128,768", help="with --torch: the hidden states' shape B,S,H")
    c.add_argument("--no-qkv", action="store_true",
                   help="with --torch: leave the Q/K/V projections out, as the paper's graphs and profiles do")
    c.add_argument("--hardware", default="cpu",
                   help="cpu | gpu (built-in profiles, one per MPC backend), a built-in profile name "
                        "(e.g. cpu-nexus-sci), or 'local' to plan for this machine")
    c.add_argument("--device", default=None, help="with --hardware local: cpu | cuda (default: detected)")
    c.add_argument("--profile", default=None, help="profile JSON path or 'auto' (overrides --hardware)")
    c.add_argument("--network", default="100Mbps/20ms", help="BANDWIDTH/RTT, e.g. 10Mbps/1ms")
    _peer_args(c)
    _backend_args(c)
    c.add_argument("--strict", action="store_true",
                   help="refuse cost entries that are not in the profile instead of scaling them")
    c.add_argument("-o", "--output", default=None, help="write plan.json")
    c.add_argument("--steps", action="store_true",
                   help="also print the plan operator by operator, with the conversions and bootstraps")
    c.set_defaults(func=_cmd_compile)
    e = sub.add_parser("explain", help="summarize a plan.json")
    e.add_argument("plan")
    e.add_argument("--steps", action="store_true", help="also print the plan operator by operator")
    e.set_defaults(func=_cmd_explain)
    b = sub.add_parser("backends", help="list MPC backends; with --model, which ones fit a deployment")
    b.add_argument("--model", choices=["bert", "vit", "bert_paper", "vit_paper"], default=None)
    b.add_argument("--blocks", type=int, default=12)
    b.add_argument("--seq", type=int, default=None)
    b.add_argument("--hardware", default="cpu")
    b.add_argument("--device", default=None)
    b.add_argument("--profile", default=None)
    b.add_argument("--network", default="100Mbps/20ms")
    _backend_args(b)
    b.set_defaults(func=_cmd_backends)
    m = sub.add_parser("profile", help="measure the cost entries a model needs on this machine (MicroBench)")
    m.add_argument("--model", choices=["bert", "vit", "bert_paper", "vit_paper"], default="bert")
    m.add_argument("--blocks", type=int, default=12)
    m.add_argument("--seq", type=int, default=None)
    m.add_argument("--device", default=None, help="cpu | cuda (default: detected)")
    m.add_argument("--backend", default="sci", help="MPC backend to measure (default: sci)")
    m.add_argument("--he", default="nexus", help="HE backend to measure (default: nexus)")
    m.add_argument("--dry-run", action="store_true", help="only report what is missing from the cache")
    m.set_defaults(func=_cmd_profile)
    n = sub.add_parser("netbench", help="measure the network between the two parties (bandwidth, RTT)")
    n.add_argument("--serve", action="store_true", help="server side: wait for measurements")
    n.add_argument("--peer", "--ip", dest="peer", default=None, metavar="HOST[:PORT]", help="client side: measure to HOST")
    n.add_argument("--host", default="0.0.0.0", help="with --serve: address to listen on")
    n.add_argument("--port", type=int, default=7787)
    n.add_argument("--seconds", type=float, default=2.0, help="seconds of transfer each way")
    n.add_argument("--streams", type=int, default=1, help="parallel TCP streams")
    n.add_argument("--rounds", type=int, default=50, help="round trips for the RTT")
    n.set_defaults(func=_cmd_netbench)
    p = sub.add_parser("profiles", help="list built-in profiles")
    p.set_defaults(func=_cmd_profiles)
    sub.add_parser("selfcheck", add_help=False,
                   help="check that this machine computes the OpenFHE kernels correctly (selfcheck --help)")
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["selfcheck"]:
        from . import selfcheck
        return selfcheck.main(args[1:])
    a = ap.parse_args(argv)
    if getattr(a, "seq", 0) is None and getattr(a, "model", None):
        a.seq = 128 if a.model.startswith("bert") else 197
    try:
        return a.func(a)
    except (ValueError, ConnectionError) as exc:   # UnsupportedGraph / UnsupportedOperator / no netbench server ...
        print(f"cipherflow: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

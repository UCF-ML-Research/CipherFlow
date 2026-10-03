"""Automatic MicroBench: measure the cost entries a graph needs on this machine.

`microbench(graph)` works out which cost entries the planner needs for `graph`
(each operator shape in each available domain, both conversion directions for
every edge shape, and bootstrapping), loads this machine's cache, measures only
the missing entries with the installed backends, and saves them back. A second
call for the same or a structurally similar graph measures nothing.

Cache: $CIPHERFLOW_CACHE or ~/.cache/cipherflow/<machine-fingerprint>-<device>/
  profile.json      measured entries (LatencyProfile)
  references.json   NEXUS reference-kernel timings (HE costs for new shapes are
                    scaled from these, so new shapes need no new HE runs)

Backends wired for automatic measurement (CPU): NEXUS-CKKS (HE operators and
bootstrapping), SCI/BOLT bridges (MPC operators), and the HE<->MPC conversion
profiler. GPU measurement is not wired yet; use a measured profile file instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from compiler.state_expanded.profile_schema import (
    HE_LEVEL_BUDGET,
    BootstrapRecord,
    ConversionRecord,
    LatencyProfile,
    OperatorRecord,
)

from . import backends as _backends

Shape = Tuple[int, ...]
OpKey = Tuple[str, str, Shape, Shape]          # (op_type, domain, input_shape, output_shape)


class BackendUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------- machine identity

def local_hardware() -> Dict[str, str]:
    info = {"machine": platform.machine(), "cpu_model": "", "gpu_name": ""}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        info["gpu_name"] = out.splitlines()[0] if out else ""
    except Exception:
        pass
    return info


def local_device() -> str:
    return "cuda" if local_hardware()["gpu_name"] else "cpu"


def fingerprint(device: str) -> str:
    hw = local_hardware()
    raw = "|".join([hw["machine"], hw["cpu_model"], hw["gpu_name"] if device == "cuda" else "", device])
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def cache_dir(device: str) -> Path:
    root = Path(os.environ.get("CIPHERFLOW_CACHE", Path.home() / ".cache" / "cipherflow"))
    return root / f"{fingerprint(device)}-{device}"


# ----------------------------------------------------------------- requirements

@dataclass
class Requirements:
    ops: Set[OpKey] = field(default_factory=set)
    conversion_shapes: Set[Shape] = field(default_factory=set)
    bootstrap: bool = False


def required_entries(graph, methods: Iterable[_backends.Method]) -> Requirements:
    methods = list(methods)
    domains = _backends.supported_ops(methods)
    ruled_out = _backends.violations(graph, methods)   # contract violations need no measurement
    req = Requirements()
    for n in graph.nodes:
        for d in sorted(domains.get(n.op_type, ())):
            if (n.node_id, d) not in ruled_out:
                req.ops.add((n.op_type, d, tuple(n.input_shape), tuple(n.output_shape)))
    req.conversion_shapes = {tuple(e.tensor_shape) for e in graph.edges}
    req.bootstrap = any(m.domain == "HE" for m in methods)
    return req


def missing_entries(profile: Optional[LatencyProfile], req: Requirements) -> Requirements:
    have_ops: Set[OpKey] = set()
    have_conv: Set[Tuple[str, str, Shape]] = set()
    has_bs = False
    if profile is not None:
        have_ops = {(r.op_type, r.domain, tuple(r.input_shape), tuple(r.output_shape)) for r in profile.operators}
        have_conv = {(c.from_domain, c.to_domain, tuple(c.tensor_shape)) for c in profile.conversions}
        has_bs = profile.bootstrap is not None
    return Requirements(
        ops={k for k in req.ops if k not in have_ops},
        conversion_shapes={s for s in req.conversion_shapes
                           if ("HE", "MPC", s) not in have_conv or ("MPC", "HE", s) not in have_conv},
        bootstrap=req.bootstrap and not has_bs,
    )


# ----------------------------------------------------------------- measurers
# Each measurer takes the work list and the cache directory (for its own state,
# e.g. reference timings) and returns records. Tests replace MEASURERS.

def _with_source(rec, source: str):
    meta = dict(rec.metadata or {})
    meta.setdefault("source", source)
    meta["measured_on"] = local_hardware()
    return replace(rec, metadata=meta)


def _measure_he_cpu(ops: List[OpKey], cache: Path, verbose: bool) -> List[OperatorRecord]:
    from compiler.state_expanded import profiler_he_real as he
    if not he.NEXUS_MAIN.exists():
        raise BackendUnavailable(f"NEXUS binary not found at {he.NEXUS_MAIN}; set CIPHERFLOW_NEXUS_BUILD")
    ref_file = cache / "references.json"
    refs = json.loads(ref_file.read_text()) if ref_file.exists() else {}
    if not all(k in refs for k in ("GELU", "LayerNorm", "SoftMax", "MatMul")):
        if verbose:
            print("[microbench] running NEXUS reference kernels (one time per machine) ...", flush=True)
        refs.update(he.measure_nexus_references(verbose=verbose))
        ref_file.write_text(json.dumps(refs, indent=2))
    shapes = [(op, i, o) for (op, _d, i, o) in ops]
    return [_with_source(r, "NEXUS reference kernel, scaled to shape") for r in he.profile_he_real(shapes, refs)]


def _measure_mpc_cpu(ops: List[OpKey], cache: Path, verbose: bool) -> List[OperatorRecord]:
    from compiler.state_expanded import profiler_mpc as mpc
    if not mpc.BIN_GELU.exists():
        raise BackendUnavailable(f"SCI bridge binaries not found in {mpc.BIN_GELU.parent}; set CIPHERFLOW_SCI_BIN")
    out = []
    for k, (op, _d, i, o) in enumerate(ops, 1):
        if verbose:
            print(f"[microbench] MPC {op} {i} ({k}/{len(ops)}) ...", flush=True)
        t0 = time.time()
        rec = mpc.profile_mpc_for_bert(i, o, op)
        out.append(_with_source(rec, "SCI/BOLT bridge, localhost"))
        if verbose:
            print(f"             {rec.local_compute_ms:.0f} ms local, {rec.comm_bytes / 2**20:.1f} MiB, "
                  f"{rec.comm_rounds} rounds ({time.time() - t0:.0f} s)", flush=True)
    return out


def _measure_conversions_cpu(shapes: List[Shape], cache: Path, verbose: bool) -> List[ConversionRecord]:
    from compiler.state_expanded.profiler_conversion import profile_conversions
    return [_with_source(r, "mask-and-decrypt / re-encryption: emulated compute, analytic bytes and rounds")
            for r in profile_conversions(shapes, device="cpu")]


def _measure_bootstrap_cpu(cache: Path, verbose: bool) -> BootstrapRecord:
    from compiler.state_expanded import profiler_he_real as he
    if not he.NEXUS_BOOTSTRAP.exists():
        raise BackendUnavailable(f"NEXUS bootstrapping binary not found at {he.NEXUS_BOOTSTRAP}")
    if verbose:
        print("[microbench] running NEXUS bootstrapping ...", flush=True)
    return _with_source(he.profile_bootstrap_real(), "NEXUS bootstrapping binary")


MEASURERS: Dict[str, Dict[str, Callable]] = {
    "cpu": {"HE": _measure_he_cpu, "MPC": _measure_mpc_cpu,
            "conversion": _measure_conversions_cpu, "bootstrap": _measure_bootstrap_cpu},
}


# ----------------------------------------------------------------- public entry point

def load_cached(device: Optional[str] = None) -> Optional[LatencyProfile]:
    device = device or local_device()
    f = cache_dir(device) / "profile.json"
    return LatencyProfile.load(str(f)) if f.exists() else None


def microbench(graph, methods="auto", *, device: Optional[str] = None, dry_run: bool = False,
               verbose: bool = True) -> LatencyProfile | Requirements:
    """Return a profile covering `graph` on this machine, measuring only missing entries.

    dry_run=True returns the missing entries without measuring anything.
    """
    device = device or local_device()
    if device == "cuda" and not local_hardware()["gpu_name"]:
        raise BackendUnavailable("device='cuda' requested but no NVIDIA GPU is visible on this machine")
    meths = _backends.builtin_methods() if methods == "auto" else list(methods)
    cdir = cache_dir(device)
    cached = load_cached(device)
    todo = missing_entries(cached, required_entries(graph, meths))
    if dry_run:
        return todo

    n_he = sum(d == "HE" for _, d, _, _ in todo.ops)
    n_mpc = sum(d == "MPC" for _, d, _, _ in todo.ops)
    if verbose:
        print(f"[microbench] cache {cdir} | to measure: {n_he} HE ops, {n_mpc} MPC ops, "
              f"{len(todo.conversion_shapes)} conversion shapes, bootstrap={'yes' if todo.bootstrap else 'no'}",
              flush=True)
    if not (todo.ops or todo.conversion_shapes or todo.bootstrap):
        return cached

    if device not in MEASURERS:
        raise BackendUnavailable(
            f"automatic MicroBench is not wired for device={device!r} yet; pass a measured profile "
            "file (profile='path/to/profile.json'), or plan for CPU execution with "
            "Deployment.local(..., device='cpu')")
    m = MEASURERS[device]
    cdir.mkdir(parents=True, exist_ok=True)
    ops: List[OperatorRecord] = list(cached.operators) if cached else []
    convs: List[ConversionRecord] = list(cached.conversions) if cached else []
    boot = cached.bootstrap if cached else None
    for dom in ("HE", "MPC"):
        work = sorted(k for k in todo.ops if k[1] == dom)
        if work:
            ops += m[dom](work, cdir, verbose)
    if todo.conversion_shapes:
        have = {(c.from_domain, c.to_domain, tuple(c.tensor_shape)) for c in convs}
        new = m["conversion"](sorted(todo.conversion_shapes), cdir, verbose)
        convs += [c for c in new if (c.from_domain, c.to_domain, tuple(c.tensor_shape)) not in have]
    if todo.bootstrap:
        boot = m["bootstrap"](cdir, verbose)

    profile = LatencyProfile(platform=device, hardware=local_hardware(), he_level_budget=HE_LEVEL_BUDGET,
                             operators=ops, conversions=convs, bootstrap=boot,
                             metadata={"source": "cipherflow.microbench", "fingerprint": fingerprint(device)})
    profile.save(str(cdir / "profile.json"))
    if verbose:
        print(f"[microbench] saved {len(ops)} operator, {len(convs)} conversion entries to {cdir}", flush=True)
    return profile

"""Automatic MicroBench: measure the cost entries a graph needs on this machine.

`microbench(graph)` works out which cost entries the planner needs for `graph`
(each operator shape in each available domain, both conversion directions for
every edge shape, and bootstrapping), loads this machine's cache, measures only
the missing entries with the installed backends, and saves them back. A second
call for the same or a structurally similar graph measures nothing.

Cache: $CIPHERFLOW_CACHE or ~/.cache/cipherflow/<machine-fingerprint>-<device>/
  references.json              NEXUS reference-kernel timings (HE costs for new shapes are
                               scaled from these, so new shapes need no new HE runs)
  <he>-<mpc>/profile.json      measured entries (LatencyProfile) for that backend pair

Wired for automatic measurement, on CPU:
  nexus+sci    NEXUS reference kernels (HE) + SCI/BOLT bridges (MPC, two processes)
  nexus+shaft  NEXUS reference kernels (HE) + SHAFT/CrypTen (MPC, two party processes)
  both: HE<->MPC conversion with real SEAL encryption (native/conversion) and the NEXUS
  bootstrapping binary (priced per ciphertext of 8192 values).
GPU measurement is not wired yet; use a measured profile file.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
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


def machine_dir(device: str) -> Path:
    root = Path(os.environ.get("CIPHERFLOW_CACHE", Path.home() / ".cache" / "cipherflow"))
    return root / f"{fingerprint(device)}-{device}"


def cache_dir(device: str, backend: str = "sci", he: str = "nexus") -> Path:
    return machine_dir(device) / f"{he}-{backend}"


_REPO = Path(__file__).resolve().parents[1]


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
            print(f"[microbench] MPC {op} {i}" + (f" ({k}/{len(ops)})" if len(ops) > 1 else "") + " ...",
                  flush=True)
        t0 = time.time()
        rec = mpc.profile_mpc_for_bert(i, o, op)
        out.append(_with_source(rec, "SCI/BOLT bridge, localhost"))
        if verbose:
            print(f"             {rec.local_compute_ms:.0f} ms local, {rec.comm_bytes / 2**20:.1f} MiB, "
                  f"{rec.comm_rounds} rounds ({time.time() - t0:.0f} s)", flush=True)
    return out


def _conversion_bench() -> Path:
    exe = Path(os.environ.get("CIPHERFLOW_CONVERSION_BENCH",
                              _REPO / "native" / "conversion" / "build" / "conversion_bench"))
    if not exe.exists():
        raise BackendUnavailable(
            f"HE<->MPC conversion benchmark not found at {exe}; build it (needs SEAL 4.1): "
            "cmake -S native/conversion -B native/conversion/build && cmake --build native/conversion/build")
    return exe


def _conversions_seal(ell: int, frac: int, ring: str) -> Callable:
    """Real conversion protocol (SEAL CKKS at the NEXUS parameters, shares in Z_2^ell)."""
    def measure(shapes: List[Shape], cache: Path, verbose: bool) -> List[ConversionRecord]:
        exe = _conversion_bench()
        out: List[ConversionRecord] = []
        for shape in shapes:
            numel = 1
            for d in shape:
                numel *= int(d)
            if verbose:
                print(f"[microbench] conversion {tuple(shape)} ...", flush=True)
            res = subprocess.run([str(exe), str(numel), "3", str(ell), str(frac)],
                                 capture_output=True, text=True, timeout=3600)
            if res.returncode != 0:
                raise RuntimeError(f"conversion benchmark failed: {res.stderr.strip()[:300]}")
            for line in res.stdout.splitlines():
                c = json.loads(line)
                he2mpc = c["direction"] == "HE->MPC"
                out.append(_with_source(ConversionRecord(
                    from_domain="HE" if he2mpc else "MPC", to_domain="MPC" if he2mpc else "HE",
                    method="seal_mask_decrypt" if he2mpc else "seal_share_encrypt",
                    tensor_shape=tuple(shape), local_compute_ms=c["server_ms"] + c["client_ms"],
                    comm_bytes=int(c["bytes"]), comm_rounds=int(c["rounds"]),
                    metadata={"server_ms": c["server_ms"], "client_ms": c["client_ms"],
                              "ciphertexts": c["ciphertexts"], "max_abs_err": c["max_abs_err"],
                              "ring": ring}), "native/conversion (SEAL 4.1, N=2^16, NEXUS modulus chain)"))
        return out
    return measure


def _mpc_shaft(device: str) -> Callable:
    """SHAFT/CrypTen operators, two party processes (online phase, triples from a local seed)."""
    def measure(ops: List[OpKey], cache: Path, verbose: bool) -> List[OperatorRecord]:
        root = Path(os.environ.get("CIPHERFLOW_SHAFT_ROOT", _REPO / "baseline" / "SHAFT"))
        if not (root / "crypten").exists():
            raise BackendUnavailable(f"SHAFT not found at {root}; set CIPHERFLOW_SHAFT_ROOT")
        out: List[OperatorRecord] = []
        with tempfile.TemporaryDirectory() as td:
            jobs, res_file = Path(td) / "jobs.json", Path(td) / "out.json"
            jobs.write_text(json.dumps([[op, list(i), list(o)] for (op, _d, i, o) in ops]))
            if verbose:
                print(f"[microbench] SHAFT two-party: {', '.join(op for op, *_ in ops)} ...", flush=True)
            env = dict(os.environ, CIPHERFLOW_SHAFT_ROOT=str(root),
                       PYTHONPATH=os.pathsep.join([str(_REPO), os.environ.get("PYTHONPATH", "")]))
            res = subprocess.run([sys.executable, "-m", "compiler.state_expanded.bench_shaft_2pc",
                                  "--device", device, "--provider", "TFP", "--jobs", str(jobs), "-o", str(res_file)],
                                 cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=3600)
            if res.returncode != 0 or not res_file.exists():
                raise RuntimeError(f"SHAFT benchmark failed: {res.stderr.strip()[-400:]}")
            rows = json.loads(res_file.read_text())
        for r in rows:
            out.append(_with_source(OperatorRecord(
                op_type=r["op"], domain="MPC", method=f"shaft_crypten_2pc_{device}",
                input_shape=tuple(r["shape_in"]), output_shape=tuple(r["shape_out"]),
                local_compute_ms=r["ms"], comm_bytes=int(r["wire_bytes"]),
                comm_rounds=int(max(r["rounds_parties"])), he_level_delta=0,
                metadata={"provider": "TFP (online phase)", "ms_parties": r["ms_parties"],
                          "rounds_definition": "CrypTen communication calls per party (max over parties)"}),
                "SHAFT/CrypTen, two party processes on localhost"))
        return out
    return measure


def _measure_bootstrap_cpu(cache: Path, verbose: bool) -> BootstrapRecord:
    from compiler.state_expanded import profiler_he_real as he
    if not he.NEXUS_BOOTSTRAP.exists():
        raise BackendUnavailable(f"NEXUS bootstrapping binary not found at {he.NEXUS_BOOTSTRAP}")
    if verbose:
        print("[microbench] running NEXUS bootstrapping ...", flush=True)
    rec = _with_source(he.profile_bootstrap_real(), "NEXUS bootstrapping binary")
    # the NEXUS demo bootstraps one sparse ciphertext of 2^13 values: price per ciphertext
    return replace(rec, metadata=dict(rec.metadata or {}, slots=8192))


# device -> MPC backend -> measurers (HE and bootstrap use the machine directory for
# NEXUS reference timings, shared by all MPC backends)
# ---------------------------------------------------------------- OpenFHE backend (native/openfhe)

def _openfhe_bin(name: str, env: str) -> Path:
    exe = Path(os.environ.get(env, _REPO / "native" / "openfhe" / "build" / name))
    if not exe.exists():
        raise BackendUnavailable(
            f"OpenFHE kernels not found at {exe}; build them (needs OpenFHE 1.4): "
            "cmake -S native/openfhe -B native/openfhe/build && cmake --build native/openfhe/build")
    return exe


def _openfhe_spec(op: str, i: Shape, o: Shape) -> str:
    if op == "Residual_Add":
        return f"add:H={i[-1]}"
    if op == "GeLU":
        return f"gelu:F={i[-1]},degree=27"
    if op == "GeLU.wide":                     # cost and levels do not depend on the bound
        return f"gelu_wide:F={i[-1]},B=64"
    if op == "LayerNorm.wide":                # 2 Newton steps (variance ratio ~200)
        return f"layernorm_wide:H={i[-1]},vlo=0.47,vhi=123"
    if op == "LayerNorm.wider":               # 6 Newton steps (variance ratio ~5000)
        return f"layernorm_wide:H={i[-1]},vlo=0.47,vhi=2340"
    if op == "Softmax.wide":                  # priced at 4 rounds (scores as in BERT-base layer 2)
        return f"softmax_wide:heads={i[1]},lo=-28.2,hi=28.6,rmin=0"
    if op == "LayerNorm":
        return f"layernorm:H={i[-1]},degree=27"
    if op == "Softmax":
        return f"softmax:heads={i[1]}"
    if op == "Linear_QKV":
        return f"qkv:H={i[-1]}"
    if op in ("Out_Projection", "FFN_Linear_1"):
        return f"ffn1:H={i[-1]},F={o[-1]}"
    if op == "FFN_Linear_2":
        return f"ffn2:F={i[-1]},H={o[-1]}"
    if op == "Attention_QK_MatMul":
        return f"qk:heads={o[1]}"
    if op == "Attention_V_MatMul":
        return f"av:heads={i[1]}"
    raise BackendUnavailable(f"no OpenFHE kernel for {op}")


def openfhe_record(op: str, i: Shape, o: Shape, mode: str, r: dict) -> OperatorRecord:
    """An he_ops result as a profile record: an operator that bootstraps internally needs its
    levels plus one (the ciphertext it bootstraps keeps one) and returns at level_out."""
    inner = int(r["internal_bootstraps"]) > 0
    meta = {"source": f"native/openfhe/he_ops ({mode})", "internal_bootstraps": r["internal_bootstraps"],
            "bootstrap_ms": r["bootstrap_ms"], "max_abs_err": r["max_abs_err"], "approx": r.get("approx", ""),
            "ciphertexts": r["ciphertexts"]}
    if r.get("retried_after_wrong_result"):
        meta["retried_after_wrong_result"] = True
    if inner:
        meta["level_out"] = int(r["level_out"])
    return OperatorRecord(op_type=op, domain="HE", method=f"openfhe_ckks_{mode}", input_shape=tuple(i),
                          output_shape=tuple(o), local_compute_ms=float(r["ms"]), comm_bytes=0, comm_rounds=0,
                          he_level_delta=int(r["levels_used"]) + (1 if inner else 0), metadata=meta)


def _he_openfhe(mode: str) -> Callable:
    def measure(ops: List[OpKey], cache: Path, verbose: bool) -> List[OperatorRecord]:
        exe = _openfhe_bin("he_ops", "CIPHERFLOW_OPENFHE_OPS")
        specs = [_openfhe_spec(op, i, o) for (op, _d, i, o) in ops]
        if verbose:
            print(f"[microbench] OpenFHE ({mode}): {' '.join(specs)} ...", flush=True)
        def run(sp):
            res = subprocess.run([str(exe), mode] + sp, capture_output=True, text=True, timeout=6 * 3600)
            return [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")], res
        rows, res = run(specs)
        out = []
        for k, (op, _d, i, o) in enumerate(ops):
            r = rows[k] if k < len(rows) else None
            if r is None or r["max_abs_err"] < 0:          # crashed or not decodable: measure it again, alone
                if verbose:
                    print(f"[microbench] OpenFHE {op}: wrong result, measuring again", flush=True)
                again, res2 = run([specs[k]])
                if not again or again[0]["max_abs_err"] < 0:
                    raise RuntimeError(f"OpenFHE {op} {tuple(i)} gave a wrong result twice "
                                       f"({(res2.stderr or res.stderr).strip()[-200:]}); the kernels are "
                                       "deterministic, so check the machine: `he_ops fast keycheck bootcheck "
                                       "racecheck` and a memory test")
                r = dict(again[0], retried_after_wrong_result=True)
            out.append(openfhe_record(op, i, o, mode, r))
        return out
    return measure


def _conversions_openfhe(ell: int, frac: int, levels: int, ring: str) -> Callable:
    def measure(shapes: List[Shape], cache: Path, verbose: bool) -> List[ConversionRecord]:
        exe = _openfhe_bin("conv_bench", "CIPHERFLOW_OPENFHE_CONV")
        out: List[ConversionRecord] = []
        for shape in shapes:
            numel = 1
            for d in shape:
                numel *= int(d)
            if verbose:
                print(f"[microbench] conversion {tuple(shape)} (OpenFHE) ...", flush=True)
            res = subprocess.run([str(exe), str(numel), "3", str(ell), str(frac), str(levels)],
                                 capture_output=True, text=True, timeout=3600)
            if res.returncode != 0:
                raise RuntimeError(f"OpenFHE conversion benchmark failed: {res.stderr.strip()[:300]}")
            for line in res.stdout.splitlines():
                c = json.loads(line)
                he2mpc = c["direction"] == "HE->MPC"
                out.append(_with_source(ConversionRecord(
                    from_domain="HE" if he2mpc else "MPC", to_domain="MPC" if he2mpc else "HE",
                    method="openfhe_mask_decrypt" if he2mpc else "openfhe_share_encrypt",
                    tensor_shape=tuple(shape), local_compute_ms=c["server_ms"] + c["client_ms"],
                    comm_bytes=int(c["bytes"]), comm_rounds=int(c["rounds"]),
                    metadata={"server_ms": c["server_ms"], "client_ms": c["client_ms"],
                              "ciphertexts": c["ciphertexts"], "max_abs_err": c["max_abs_err"], "ring": ring,
                              "note": "mask in the slot domain: protocol costs; a hardened version masks "
                                      "coefficients with noise flooding"}),
                    "native/openfhe/conv_bench (OpenFHE CKKS, N=2^16)"))
        return out
    return measure


def _bootstrap_openfhe(mode: str) -> Callable:
    def measure(cache: Path, verbose: bool) -> BootstrapRecord:
        exe = _openfhe_bin("he_ops", "CIPHERFLOW_OPENFHE_OPS")
        if verbose:
            print(f"[microbench] OpenFHE bootstrapping ({mode}) ...", flush=True)
        res = subprocess.run([str(exe), mode, "bootstrap"], capture_output=True, text=True, timeout=3600)
        rows = [json.loads(l) for l in res.stdout.splitlines() if l.startswith("{")]
        if res.returncode != 0 or not rows:
            raise RuntimeError(f"OpenFHE bootstrapping failed: {res.stderr.strip()[-300:]}")
        r = rows[0]
        return _with_source(BootstrapRecord(method=f"openfhe_bootstrap_{mode}", local_compute_ms=float(r["ms"]),
                                            metadata={"slots": 32768, "levels_after": r["level_out"],
                                                      "max_abs_err": r["max_abs_err"], "approx": r.get("approx", "")}),
                            f"native/openfhe/he_ops ({mode})")
    return measure


MEASURERS: Dict[str, Dict[str, Dict[str, Callable]]] = {
    "cpu": {
        "nexus+sci": {"HE": _measure_he_cpu, "MPC": _measure_mpc_cpu,
                      "conversion": _conversions_seal(37, 12, "SCI: Z_2^37, 12 fractional bits"),
                      "bootstrap": _measure_bootstrap_cpu},
        "nexus+shaft": {"HE": _measure_he_cpu, "MPC": _mpc_shaft("cpu"),
                        "conversion": _conversions_seal(64, 16, "CrypTen: Z_2^64, 16 fractional bits"),
                        "bootstrap": _measure_bootstrap_cpu},
        **{f"openfhe-{mode}+{mpc}": {
            "HE": _he_openfhe(mode), "MPC": mpc_fn,
            "conversion": _conversions_openfhe(ell, frac, lv, ring), "bootstrap": _bootstrap_openfhe(mode)}
           for mode, lv in (("fast", 8), ("precise", 7))
           for mpc, mpc_fn, ell, frac, ring in (
               ("sci", _measure_mpc_cpu, 37, 12, "SCI: Z_2^37, 12 fractional bits"),
               ("shaft", _mpc_shaft("cpu"), 64, 16, "CrypTen: Z_2^64, 16 fractional bits"))},
    },
}


def measurable(device: str) -> List[str]:
    """Backend pairs ("<he>+<mpc>") whose costs can be measured automatically on `device`."""
    return sorted(MEASURERS.get(device, {}))


# ----------------------------------------------------------------- public entry point

def load_cached(device: Optional[str] = None, backend: str = "sci", he: str = "nexus") -> Optional[LatencyProfile]:
    device = device or local_device()
    candidates = [cache_dir(device, backend, he) / "profile.json"]
    if he == "nexus" and backend == "sci":              # v0.1 cache layout
        candidates.append(machine_dir(device) / "profile.json")
    f = next((c for c in candidates if c.exists()), None)
    if f is None:
        return None
    prof = LatencyProfile.load(str(f))
    meta = dict(prof.metadata or {})
    meta.setdefault("mpc_backend", backend)
    meta.setdefault("he_backend", he)
    return replace(prof, metadata=meta)


def microbench(graph, methods="auto", *, device: Optional[str] = None, backend: str = "sci",
               he: str = "nexus", dry_run: bool = False, verbose: bool = True) -> LatencyProfile | Requirements:
    """Return a profile covering `graph` on this machine for the HE backend `he` and the MPC
    backend `backend`, measuring only missing entries.

    dry_run=True returns the missing entries without measuring anything.
    """
    device = device or local_device()
    if device == "cuda" and not local_hardware()["gpu_name"]:
        raise BackendUnavailable("device='cuda' requested but no NVIDIA GPU is visible on this machine")
    if methods == "auto":
        hb = _backends.HE_BACKENDS.get(he)
        meths = _backends.builtin_methods(backend, he=he, level_deltas=None if hb and hb.level_deltas else {})
    else:
        meths = list(methods)
    key = f"{he}+{backend}"
    cdir = cache_dir(device, backend, he)
    cached = load_cached(device, backend, he)
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

    if key not in MEASURERS.get(device, {}):
        raise BackendUnavailable(
            f"automatic MicroBench is not wired for {key} on device={device!r} yet "
            f"(available: {measurable(device) or 'none'}); pass a measured profile file "
            "(profile='path/to/profile.json'), or plan for CPU execution with "
            "Deployment.local(..., device='cpu')")
    m = MEASURERS[device][key]
    mdir = machine_dir(device)
    cdir.mkdir(parents=True, exist_ok=True)
    ops: List[OperatorRecord] = list(cached.operators) if cached else []
    convs: List[ConversionRecord] = list(cached.conversions) if cached else []
    boot = cached.bootstrap if cached else None
    hb = _backends.HE_BACKENDS.get(he)
    budget = hb.levels if hb is not None and hb.levels else HE_LEVEL_BUDGET

    def save():
        LatencyProfile(platform=device, hardware=local_hardware(), he_level_budget=budget,
                       operators=ops, conversions=convs, bootstrap=boot,
                       metadata={"source": "cipherflow.microbench", "fingerprint": fingerprint(device),
                                 "mpc_backend": backend, "he_backend": he}).save(str(cdir / "profile.json"))

    for dom in ("HE", "MPC"):
        work = sorted(k for k in todo.ops if k[1] == dom)
        if dom == "HE" and work:
            ops += m[dom](work, mdir, verbose)
            save()
        for k, key in enumerate(work if dom == "MPC" else [], 1):   # slow: one at a time, saved as we go
            if verbose:
                print(f"[microbench] ({k}/{len(work)})", end=" ", flush=True)
            ops += m[dom]([key], mdir, verbose)
            save()
    if todo.conversion_shapes:
        have = {(c.from_domain, c.to_domain, tuple(c.tensor_shape)) for c in convs}
        new = m["conversion"](sorted(todo.conversion_shapes), mdir, verbose)
        convs += [c for c in new if (c.from_domain, c.to_domain, tuple(c.tensor_shape)) not in have]
    if todo.bootstrap:
        boot = m["bootstrap"](mdir, verbose)

    profile = LatencyProfile(platform=device, hardware=local_hardware(), he_level_budget=budget,
                             operators=ops, conversions=convs, bootstrap=boot,
                             metadata={"source": "cipherflow.microbench", "fingerprint": fingerprint(device),
                                       "mpc_backend": backend, "he_backend": he})
    profile.save(str(cdir / "profile.json"))
    if verbose:
        print(f"[microbench] saved {len(ops)} operator, {len(convs)} conversion entries to {cdir}", flush=True)
    return profile

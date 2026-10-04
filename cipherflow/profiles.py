"""Cost profiles (MicroBench output): built-ins, files, and the local cache.

A profile holds the costs of one HE backend and one MPC backend (and the conversions
between them) measured on one machine. Built-in profiles are named
<hardware>-<he backend>-<mpc backend>; the hardware names alone ("cpu", "gpu") select
among them in cipherflow.compile. The paper's profiles are kept as paper-cpu / paper-gpu.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

from compiler.state_expanded.profile_schema import LatencyProfile

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "compiler" / "state_expanded" / "profiles"

BUILTIN: Dict[str, Dict[str, str]] = {
    "cpu-nexus-sci": {
        "hardware": "cpu", "he": "nexus", "mpc": "sci", "role": "measured",
        "file": "profile_cpu_nexus_sci.json",
        "label": "CPU (Threadripper PRO 3955WX): NEXUS reference-scaled HE + SCI/BOLT (measured) "
                 "+ SEAL conversions (measured)"},
    "cpu-nexus-shaft": {
        "hardware": "cpu", "he": "nexus", "mpc": "shaft", "role": "measured",
        "file": "profile_cpu_nexus_shaft.json",
        "label": "CPU (Threadripper PRO 3955WX): NEXUS reference-scaled HE + SHAFT two-party (measured) "
                 "+ SEAL conversions (measured)"},
    "gpu-nexus-shaft": {
        "hardware": "gpu", "he": "nexus", "mpc": "shaft", "role": "measured",
        "file": "profile_gpu_nexus_shaft.json",
        "label": "GPU (2x RTX 3090): NEXUS-CUDA reference-scaled HE + SHAFT two-party (measured) "
                 "+ SEAL conversions (measured on CPU)"},
    "cpu-openfhe-fast-sci": {
        "hardware": "cpu", "he": "openfhe-fast", "mpc": "sci", "role": "measured",
        "file": "profile_cpu_openfhe_fast_sci.json",
        "label": "CPU (Threadripper PRO 3955WX): OpenFHE CKKS (one-pass bootstrapping, 13 bits) + SCI/BOLT "
                 "+ OpenFHE conversions, all measured"},
    "cpu-openfhe-fast-shaft": {
        "hardware": "cpu", "he": "openfhe-fast", "mpc": "shaft", "role": "measured",
        "file": "profile_cpu_openfhe_fast_shaft.json",
        "label": "CPU (Threadripper PRO 3955WX): OpenFHE CKKS (one-pass bootstrapping, 13 bits) + SHAFT two-party "
                 "+ OpenFHE conversions, all measured"},
    "cpu-openfhe-precise-sci": {
        "hardware": "cpu", "he": "openfhe-precise", "mpc": "sci", "role": "measured",
        "file": "profile_cpu_openfhe_precise_sci.json",
        "label": "CPU (Threadripper PRO 3955WX): OpenFHE CKKS (two-pass bootstrapping, 26 bits) + SCI/BOLT "
                 "+ OpenFHE conversions, all measured"},
    "cpu-openfhe-precise-shaft": {
        "hardware": "cpu", "he": "openfhe-precise", "mpc": "shaft", "role": "measured",
        "file": "profile_cpu_openfhe_precise_shaft.json",
        "label": "CPU (Threadripper PRO 3955WX): OpenFHE CKKS (two-pass bootstrapping, 26 bits) + SHAFT two-party "
                 "+ OpenFHE conversions, all measured"},
    "paper-cpu": {
        "hardware": "cpu", "he": "nexus", "mpc": "shaft", "role": "paper",
        "file": "profile_cpu_full_model_shaft.json",
        "label": "the paper's CPU profile (NEXUS + SHAFT single-process, allocated communication)"},
    "paper-gpu": {
        "hardware": "gpu", "he": "nexus", "mpc": "shaft", "role": "paper",
        "file": "profile_gpu_full_model.json",
        "label": "the paper's GPU profile (NEXUS-CUDA + SHAFT single-process, allocated communication)"},
}
# v0.1 names for load_profile (the paper's profiles) and 0.2.0.dev names
ALIASES: Dict[str, str] = {"cpu": "paper-cpu", "gpu": "paper-gpu",
                           "cpu-sci": "cpu-nexus-sci", "cpu-shaft": "cpu-nexus-shaft",
                           "gpu-shaft": "gpu-nexus-shaft"}


def hardware_profiles() -> Dict[str, Dict[Tuple[str, str], str]]:
    """hardware name -> {(he backend, mpc backend): built-in profile name}, measured profiles only."""
    out: Dict[str, Dict[Tuple[str, str], str]] = {}
    for name, v in BUILTIN.items():
        if v["role"] == "measured":
            out.setdefault(v["hardware"], {})[(v["he"], v["mpc"])] = name
    return out


class ProfileNotFound(LookupError):
    pass


def list_builtin() -> Dict[str, str]:
    return {k: v["label"] for k, v in BUILTIN.items()}


def load(spec: Union[str, Path, LatencyProfile]) -> LatencyProfile:
    """Built-in name (see BUILTIN; 'cpu' / 'gpu' are the paper's profiles), a JSON path,
    a LatencyProfile, or 'auto' (this machine's cached NEXUS + SCI profile)."""
    if isinstance(spec, LatencyProfile):
        return spec
    s = ALIASES.get(str(spec), str(spec))
    if s in BUILTIN:
        return LatencyProfile.load(str(_PROFILE_DIR / BUILTIN[s]["file"]))
    if s == "auto":
        from .measure import load_cached
        cached = load_cached()
        if cached is not None:
            return cached
        raise ProfileNotFound("no cached profile for this machine yet; run `cipherflow profile` or "
                              "cipherflow.compile(..., profile='auto') on the target machine")
    p = Path(s).expanduser()
    if p.exists():
        return LatencyProfile.load(str(p))
    raise ProfileNotFound(f"unknown profile {spec!r}; built-ins: {sorted(BUILTIN)}")


def scale_compute(profile: LatencyProfile, factor: float, label: str = "") -> LatencyProfile:
    """Project a profile onto hardware `factor`x faster in local compute.
    Communication bytes and rounds are protocol quantities and stay unchanged."""
    from dataclasses import replace
    return replace(
        profile,
        operators=[replace(r, local_compute_ms=r.local_compute_ms / factor) for r in profile.operators],
        conversions=[replace(c, local_compute_ms=c.local_compute_ms / factor) for c in profile.conversions],
        bootstrap=(None if profile.bootstrap is None else
                   replace(profile.bootstrap, local_compute_ms=profile.bootstrap.local_compute_ms / factor)),
        metadata=dict(profile.metadata or {}, projected_from=profile.platform,
                      compute_scale=factor, label=label),
    )

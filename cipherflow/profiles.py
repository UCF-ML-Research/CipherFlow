"""Cost profiles (MicroBench output): built-ins, files, and the local cache."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from compiler.state_expanded.profile_schema import LatencyProfile

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "compiler" / "state_expanded" / "profiles"

BUILTIN: Dict[str, Dict[str, str]] = {
    "cpu": {"file": "profile_cpu_full_model_shaft.json",
            "label": "CPU (AMD Threadripper PRO 3955WX), NEXUS CKKS + SHAFT"},
    "gpu": {"file": "profile_gpu_full_model.json",
            "label": "GPU (paper profile), NEXUS-CUDA + SHAFT"},
}

class ProfileNotFound(LookupError):
    pass


def list_builtin() -> Dict[str, str]:
    return {k: v["label"] for k, v in BUILTIN.items()}


def load(spec: Union[str, Path, LatencyProfile]) -> LatencyProfile:
    """Built-in name ('cpu', 'gpu'), a JSON path, a LatencyProfile, or 'auto'."""
    if isinstance(spec, LatencyProfile):
        return spec
    s = str(spec)
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

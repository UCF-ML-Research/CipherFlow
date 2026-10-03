"""Build a ``LatencyProfile`` JSON by running all the sub-profilers.

Usage::

    python -m compiler.state_expanded.build_profile --device cpu \
        --output compiler/state_expanded/profiles/profile_cpu.json

    python -m compiler.state_expanded.build_profile --device cuda \
        --output compiler/state_expanded/profiles/profile_gpu.json \
        [--skip-mpc]   # skip real SCI bridge runs (use when bridges are
                       #  unavailable or you only need the HE/GPU profile)

By default this will:
- profile every HE op in the BERT graph (fast, seconds)
- profile every MPC op via the real SCI bridges (minutes)
- profile each HE<->MPC conversion at every distinct edge shape
- record the bootstrap reference cost

With ``--skip-mpc`` we still emit the profile (useful for quick iteration
on HE or on GPU where MPC bridges are unavailable) but mark each MPC
record as ``feasible=False``; the compiler will then route everything
through HE and bootstrap / MPC->HE only as needed.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Iterable, List, Tuple

from .bert_graph import (
    default_manifest,
    enumerate_edge_shapes,
    enumerate_profile_shapes,
)
from .profile_schema import (
    HE_LEVEL_BUDGET,
    LatencyProfile,
    OperatorRecord,
)
from .profiler_conversion import profile_bootstrap, profile_conversions
from .profiler_he import profile_he_operators
from .profiler_mpc import profile_mpc_operators

Shape = Tuple[int, ...]


def _describe_hardware(device: str) -> dict:
    info = {
        "device": device,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if device == "cuda":
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_count"] = torch.cuda.device_count()
        except Exception:
            pass
    return info


def _mpc_infeasible_records(shapes: Iterable[Tuple[str, Shape, Shape]]) -> List[OperatorRecord]:
    return [
        OperatorRecord(
            op_type=op_type,
            domain="MPC",
            method="method_mpc_bolt",
            input_shape=ins,
            output_shape=outs,
            local_compute_ms=0.0,
            comm_bytes=0,
            comm_rounds=0,
            he_level_delta=0,
            feasible=False,
            metadata={"reason": "skipped by --skip-mpc"},
        )
        for op_type, ins, outs in shapes
    ]


def build_profile(device: str, skip_mpc: bool = False) -> LatencyProfile:
    manifest = default_manifest()
    op_shapes = enumerate_profile_shapes(manifest)
    edge_shapes = enumerate_edge_shapes(manifest)

    print(f"[profile] device={device}  shapes={len(op_shapes)}  edge_shapes={len(edge_shapes)}")

    # 1. HE operators.
    print("[profile] HE operators ...")
    he_records = profile_he_operators(op_shapes, device=device, warmups=2, repeats=5)
    for r in he_records:
        tag = "feas" if r.feasible else "SKIP"
        print(f"   HE  {r.op_type:22s} {tag} {r.local_compute_ms:.2f}ms  {r.input_shape} -> {r.output_shape}")

    # 2. MPC operators.
    if skip_mpc:
        print("[profile] MPC operators SKIPPED (--skip-mpc)")
        mpc_records = _mpc_infeasible_records(op_shapes)
    else:
        print("[profile] MPC operators (real SCI bridges) ...")
        mpc_records = profile_mpc_operators(op_shapes, verbose=True)

    # 3. Conversions.
    print("[profile] HE<->MPC conversions ...")
    conv_records = profile_conversions(edge_shapes, device=device)
    for r in conv_records:
        print(
            f"   CONV {r.from_domain}->{r.to_domain:3s} "
            f"local={r.local_compute_ms:.2f}ms bytes={r.comm_bytes:_} rounds={r.comm_rounds} "
            f"shape={r.tensor_shape}"
        )

    # 4. Bootstrapping.
    bs = profile_bootstrap(device=device)
    print(f"[profile] bootstrap {bs.local_compute_ms:.2f}ms (ref)")

    hw = _describe_hardware(device)

    return LatencyProfile(
        platform=device,
        hardware=hw,
        he_level_budget=HE_LEVEL_BUDGET,
        operators=list(he_records) + list(mpc_records),
        conversions=list(conv_records),
        bootstrap=bs,
        metadata={
            "bert_shape_manifest": manifest.as_dict(),
            "skip_mpc": skip_mpc,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--skip-mpc", action="store_true")
    args = ap.parse_args()

    profile = build_profile(args.device, skip_mpc=args.skip_mpc)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    profile.save(args.output)
    print(f"[profile] wrote {args.output}")


if __name__ == "__main__":
    main()

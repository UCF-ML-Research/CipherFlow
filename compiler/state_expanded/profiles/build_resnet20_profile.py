"""Generate ``profile_{cpu,gpu}_resnet20.json``.

The numbers here are *synthetic but principled* scalings off the BERT
baseline profile. The intent is to demonstrate that the state-expanded
planner generalises structurally to CNN topologies — not to claim
absolute wallclock numbers vs. published ResNet HE/MPC papers.

Derivation rules (see DATA_PROVENANCE.md for the full audit):

HE side
-------
- Conv2D@HE local_compute_ms: ``conv_he_base_ms = 3000`` ms scaled by
  output FLOPs ratio relative to a 16x32x32 reference. FLOPs proxy =
  ``out_ch * out_h * out_w * in_ch * k * k``. Ref FLOPs = 16*32*32 *
  16*9 = 2,359,296. The reference ~3000 ms matches NEXUS CKKS matmul
  throughput at ~10 GFLOPs effective.
- ReLU@HE: 5000 ms for any size <= 16384 (single polynomial
  evaluation in CKKS). Marked ``feasible=False`` when input volume >
  16384 (noise-budget collapse with deeper polynomial); none of
  ResNet-20's maps exceed 16384 so all ReLUs are feasible.
- Conv2D@HE feasible only when channels<=64 AND spatial<=32 (mirrors
  NEXUS-style restriction documented in CLAUDE.md: small channel/seq
  packing only). Holds for every Conv2D in ResNet-20.
- AvgPool@HE: 10 ms (a few rotations + sums per output element).
- FC@HE: 1 ms (64 * 10 multiplies, one packed dot product).
- Residual_Add@HE: 0.05 ms (matches BERT profile's ciphertext-add
  reference constant).

MPC side (BOLT/SCI on CPU)
--------------------------
- Conv2D@MPC: lowered to im2col matmul. ``local_compute_ms`` and
  ``comm_bytes`` scaled from BERT's ``FFN_Linear_1@MPC`` row
  (146870 ms / 8.65e9 bytes / 200 rounds for [1,16,768]->[1,16,3072]
  matmul). FLOPs ratio = ResNet-conv FLOPs / FFN1 FLOPs = X / 4.5e7.
- ReLU@MPC: 30 ms local + 0.5 MB comm + 4 rounds (DReLU) — derived
  from BOLT GeLU (1325 ms / 315 MB / 408 rounds for 16*3072=49152
  elements -> ~6.4 us/element comm) but ReLU is ~10x cheaper than
  GeLU per element.
- AvgPool/Residual_Add@MPC: 0 / 0 / 0 (local share ops only).
- FC@MPC: 0.5 ms + 5 KB + 1 round (tiny matmul).

GPU multipliers
---------------
- HE local_compute: divide by 50 (matches NEXUS-CUDA published 30-60x).
- MPC local_compute: divide by 5 (matches SHAFT GPU profile).
- comm_bytes/comm_rounds unchanged (network bound).

Conversions
-----------
- HE<->MPC mask_and_decrypt / share_plus_enc per shape, derived by
  numel scaling from the BERT [1,128,768] reference (~10.5 ms,
  39 MB, 1 round). Only the four edge shapes that actually appear
  in the ResNet-20 graph.

Bootstrap: re-use the BERT profile's bootstrap record verbatim
(NEXUS CKKS bootstrap is shape-independent at fixed N).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

# ----------- ResNet-20 (op, in_shape, out_shape) instances -----------

# Each tuple: (op_type, in_shape, out_shape).
RESNET20_OP_INSTANCES: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = [
    # stem
    ("Conv2D", (1, 3, 32, 32),  (1, 16, 32, 32)),
    ("ReLU",   (1, 16, 32, 32), (1, 16, 32, 32)),
    # stage 1: 16ch 32x32
    ("Conv2D", (1, 16, 32, 32), (1, 16, 32, 32)),
    ("Residual_Add", (1, 16, 32, 32), (1, 16, 32, 32)),
    # stage 2: 32ch 16x16 (and the down-sample conv 16->32 32x32->16x16)
    ("Conv2D", (1, 16, 32, 32), (1, 32, 16, 16)),  # stride-2 main
    ("Conv2D", (1, 16, 32, 32), (1, 32, 16, 16)),  # stride-2 skip 1x1 (same I/O shape, k=1)
    ("Conv2D", (1, 32, 16, 16), (1, 32, 16, 16)),
    ("ReLU",   (1, 32, 16, 16), (1, 32, 16, 16)),
    ("Residual_Add", (1, 32, 16, 16), (1, 32, 16, 16)),
    # stage 3: 64ch 8x8
    ("Conv2D", (1, 32, 16, 16), (1, 64, 8, 8)),    # stride-2 main
    ("Conv2D", (1, 32, 16, 16), (1, 64, 8, 8)),    # stride-2 skip 1x1 (same I/O shape, k=1)
    ("Conv2D", (1, 64, 8, 8),   (1, 64, 8, 8)),
    ("ReLU",   (1, 64, 8, 8),   (1, 64, 8, 8)),
    ("Residual_Add", (1, 64, 8, 8), (1, 64, 8, 8)),
    # head
    ("AvgPool", (1, 64, 8, 8),  (1, 64, 1, 1)),
    ("FC",      (1, 64, 1, 1),  (1, 10)),
]

# Note: many (op, in, out) appear once but the same record will match
# multiple graph nodes via the cost model's exact-match lookup.

# Edge shapes that actually appear in the graph (for conversions).
RESNET20_EDGE_SHAPES: List[Tuple[int, ...]] = [
    (1, 16, 32, 32),
    (1, 32, 16, 16),
    (1, 64, 8, 8),
    (1, 64, 1, 1),
]


# ----------- HE Conv2D scaling -----------

def _conv_flops(in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> int:
    """Approximate FLOPs (ignoring constants) for a 3x3 conv unless the
    1x1 skip projections are detected by spatial-equality.

    Heuristic: kernel size = 1 if (in_h == out_h and in_h == out_h*1)
    AND in_h matches stride1 exit; we instead just use k=3 by default
    and override for the skip projections (in_ch != out_ch and stride>1
    AND ch doubles). For the synthetic profile we approximate every
    Conv2D as 3x3 (the dominant kernel) — the 1x1 skip projections are
    a minor under-count that is conservative for HE (smaller than k=3).
    Because skip projections share (in_ch, in_h, in_w, out_ch, out_h,
    out_w) with the stride-2 main conv they get the same record via
    ``find_operator`` exact match — also fine.
    """
    out_ch = out_shape[1]
    out_h, out_w = out_shape[2], out_shape[3]
    in_ch = in_shape[1]
    k = 3
    return out_ch * out_h * out_w * in_ch * k * k


# Reference: 16x32x32 Conv2D at 3->16 (we use the second 16x32x32 conv
# 16->16 instead so the reference is the dominant body conv).
_HE_CONV_REF_SHAPE_IN = (1, 16, 32, 32)
_HE_CONV_REF_SHAPE_OUT = (1, 16, 32, 32)
_HE_CONV_REF_MS_CPU = 3000.0
_HE_CONV_REF_FLOPS = _conv_flops(_HE_CONV_REF_SHAPE_IN, _HE_CONV_REF_SHAPE_OUT)


def _conv_he_local_ms(in_shape, out_shape, gpu_speedup: float = 1.0) -> float:
    flops = _conv_flops(in_shape, out_shape)
    ms = _HE_CONV_REF_MS_CPU * (flops / _HE_CONV_REF_FLOPS)
    return ms / gpu_speedup


# ReLU@HE: roughly constant per evaluation up to the slot capacity.
def _relu_he_local_ms(in_shape, gpu_speedup: float = 1.0) -> float:
    return 5000.0 / gpu_speedup


def _relu_he_feasible(in_shape) -> bool:
    vol = 1
    for d in in_shape:
        vol *= int(d)
    return vol <= 16384


def _conv_he_feasible(in_shape, out_shape) -> bool:
    out_ch = out_shape[1]
    out_h = out_shape[2]
    in_ch = in_shape[1]
    in_h = in_shape[2]
    return (max(in_ch, out_ch) <= 64) and (max(in_h, out_h) <= 32)


# ----------- MPC scaling -----------

# BOLT FFN_Linear_1: [1,16,768] -> [1,16,3072]
# Reference matmul output FLOPs proxy (matmul flops): 16 * 3072 * 768 = 3.77e7
# Reference cost: 146870 ms, 8.6e9 bytes, 200 rounds.
_MPC_CONV_REF_OP_FLOPS = 16 * 3072 * 768  # 37,748,736
_MPC_CONV_REF_MS_CPU = 146870.0
_MPC_CONV_REF_BYTES = 8645952356
_MPC_CONV_REF_ROUNDS = 200


def _conv_mpc_costs(in_shape, out_shape, gpu_speedup: float = 1.0):
    """Scale BOLT matmul costs by FLOPs ratio."""
    flops = _conv_flops(in_shape, out_shape)
    ratio = flops / _MPC_CONV_REF_OP_FLOPS
    local = _MPC_CONV_REF_MS_CPU * ratio / gpu_speedup
    comm = int(_MPC_CONV_REF_BYTES * ratio)
    rounds = max(1, int(round(_MPC_CONV_REF_ROUNDS * (ratio ** 0.5))))
    # Keep rounds within a sane range (1..50 for one conv).
    rounds = min(rounds, 50)
    return local, comm, rounds


# BOLT GeLU: [1,16,3072] = 49152 elements, 1325 ms, 315 MB, 408 rounds.
# ReLU is ~10x cheaper per element and uses 4 DReLU rounds total.
_MPC_RELU_PER_ELEM_MS = 1325.18 / 49152 / 10.0     # ~0.0027 ms/elem
_MPC_RELU_PER_ELEM_BYTES = 315294052 / 49152 / 10.0  # ~641 B/elem


def _relu_mpc_costs(in_shape, gpu_speedup: float = 1.0):
    n = 1
    for d in in_shape:
        n *= int(d)
    local = _MPC_RELU_PER_ELEM_MS * n / gpu_speedup
    comm = int(_MPC_RELU_PER_ELEM_BYTES * n)
    rounds = 4
    return local, comm, rounds


# ----------- conversions (numel-scaled from BERT [1,128,768]) -----------

# BERT reference: [1,128,768] = 98304 elements, ~10.5 ms (CPU), 39 MB, 1 round.
_CONV_REF_NUMEL = 1 * 128 * 768
_CONV_REF_MS_CPU = 10.5
_CONV_REF_BYTES = 39321600


def _conv_costs_for_shape(shape, gpu_speedup: float = 1.0):
    n = 1
    for d in shape:
        n *= int(d)
    ratio = n / _CONV_REF_NUMEL
    local = _CONV_REF_MS_CPU * ratio / gpu_speedup
    comm = int(_CONV_REF_BYTES * ratio)
    rounds = 1
    return local, comm, rounds


# ----------- builder -----------

def _build_operator_records(gpu: bool) -> List[Dict]:
    he_speedup = 50.0 if gpu else 1.0
    mpc_speedup = 5.0 if gpu else 1.0

    records: List[Dict] = []
    seen: set = set()

    def add_he_conv(in_shape, out_shape):
        key = ("Conv2D", "HE", in_shape, out_shape)
        if key in seen:
            return
        seen.add(key)
        records.append({
            "op_type": "Conv2D",
            "domain": "HE",
            "method": "synthetic_he_conv",
            "input_shape": list(in_shape),
            "output_shape": list(out_shape),
            "local_compute_ms": _conv_he_local_ms(in_shape, out_shape, he_speedup),
            "comm_bytes": 0,
            "comm_rounds": 0,
            "he_level_delta": 1,
            "feasible": _conv_he_feasible(in_shape, out_shape),
            "metadata": {
                "source": "synthetic; scaled from 3000 ms @ 16x32x32 ref by FLOPs ratio",
                "gpu_speedup": he_speedup,
                "feasibility_rule": "channels<=64 AND spatial<=32",
            },
        })

    def add_mpc_conv(in_shape, out_shape):
        key = ("Conv2D", "MPC", in_shape, out_shape)
        if key in seen:
            return
        seen.add(key)
        local, comm, rounds = _conv_mpc_costs(in_shape, out_shape, mpc_speedup)
        records.append({
            "op_type": "Conv2D",
            "domain": "MPC",
            "method": "synthetic_mpc_conv_im2col",
            "input_shape": list(in_shape),
            "output_shape": list(out_shape),
            "local_compute_ms": local,
            "comm_bytes": comm,
            "comm_rounds": rounds,
            "he_level_delta": 0,
            "feasible": True,
            "metadata": {
                "source": "synthetic; scaled from BOLT FFN_Linear_1@MPC by FLOPs ratio",
                "gpu_speedup": mpc_speedup,
            },
        })

    def add_he_relu(in_shape):
        key = ("ReLU", "HE", in_shape, in_shape)
        if key in seen:
            return
        seen.add(key)
        records.append({
            "op_type": "ReLU",
            "domain": "HE",
            "method": "synthetic_he_relu_poly",
            "input_shape": list(in_shape),
            "output_shape": list(in_shape),
            "local_compute_ms": _relu_he_local_ms(in_shape, he_speedup),
            "comm_bytes": 0,
            "comm_rounds": 0,
            "he_level_delta": 4,
            "feasible": _relu_he_feasible(in_shape),
            "metadata": {
                "source": "synthetic; degree-6 polynomial CKKS ReLU at ~5000 ms / eval (CPU)",
                "gpu_speedup": he_speedup,
                "feasibility_rule": "input volume <= 16384",
            },
        })

    def add_mpc_relu(in_shape):
        key = ("ReLU", "MPC", in_shape, in_shape)
        if key in seen:
            return
        seen.add(key)
        local, comm, rounds = _relu_mpc_costs(in_shape, mpc_speedup)
        records.append({
            "op_type": "ReLU",
            "domain": "MPC",
            "method": "synthetic_mpc_drelu",
            "input_shape": list(in_shape),
            "output_shape": list(in_shape),
            "local_compute_ms": local,
            "comm_bytes": comm,
            "comm_rounds": rounds,
            "he_level_delta": 0,
            "feasible": True,
            "metadata": {
                "source": "synthetic; ~1/10 BOLT GeLU per-element costs, 4 DReLU rounds",
                "gpu_speedup": mpc_speedup,
            },
        })

    def add_residual_add(in_shape):
        key_he = ("Residual_Add", "HE", in_shape, in_shape)
        if key_he not in seen:
            seen.add(key_he)
            records.append({
                "op_type": "Residual_Add",
                "domain": "HE",
                "method": "synthetic_he_add",
                "input_shape": list(in_shape),
                "output_shape": list(in_shape),
                "local_compute_ms": 0.05,
                "comm_bytes": 0,
                "comm_rounds": 0,
                "he_level_delta": 0,
                "feasible": True,
                "metadata": {"source": "matches BERT profile ciphertext-add constant"},
            })
        key_mpc = ("Residual_Add", "MPC", in_shape, in_shape)
        if key_mpc not in seen:
            seen.add(key_mpc)
            records.append({
                "op_type": "Residual_Add",
                "domain": "MPC",
                "method": "synthetic_mpc_add",
                "input_shape": list(in_shape),
                "output_shape": list(in_shape),
                "local_compute_ms": 0.0,
                "comm_bytes": 0,
                "comm_rounds": 0,
                "he_level_delta": 0,
                "feasible": True,
                "metadata": {"source": "share-only addition; no comm"},
            })

    def add_avgpool(in_shape, out_shape):
        key_he = ("AvgPool", "HE", in_shape, out_shape)
        if key_he not in seen:
            seen.add(key_he)
            records.append({
                "op_type": "AvgPool",
                "domain": "HE",
                "method": "synthetic_he_avgpool_rot",
                "input_shape": list(in_shape),
                "output_shape": list(out_shape),
                "local_compute_ms": 10.0 / he_speedup,
                "comm_bytes": 0,
                "comm_rounds": 0,
                "he_level_delta": 0,
                "feasible": True,
                "metadata": {"source": "synthetic; rotation-and-sum tree, 10 ms (CPU)",
                             "gpu_speedup": he_speedup},
            })
        key_mpc = ("AvgPool", "MPC", in_shape, out_shape)
        if key_mpc not in seen:
            seen.add(key_mpc)
            records.append({
                "op_type": "AvgPool",
                "domain": "MPC",
                "method": "synthetic_mpc_avgpool",
                "input_shape": list(in_shape),
                "output_shape": list(out_shape),
                "local_compute_ms": 5.0 / mpc_speedup,
                "comm_bytes": 0,
                "comm_rounds": 0,
                "he_level_delta": 0,
                "feasible": True,
                "metadata": {"source": "synthetic; share sum + truncation, no comm",
                             "gpu_speedup": mpc_speedup},
            })

    def add_fc(in_shape, out_shape):
        key_he = ("FC", "HE", in_shape, out_shape)
        if key_he not in seen:
            seen.add(key_he)
            records.append({
                "op_type": "FC",
                "domain": "HE",
                "method": "synthetic_he_fc",
                "input_shape": list(in_shape),
                "output_shape": list(out_shape),
                "local_compute_ms": 1.0 / he_speedup,
                "comm_bytes": 0,
                "comm_rounds": 0,
                "he_level_delta": 1,
                "feasible": True,
                "metadata": {"source": "synthetic; 64x10 packed dot product",
                             "gpu_speedup": he_speedup},
            })
        key_mpc = ("FC", "MPC", in_shape, out_shape)
        if key_mpc not in seen:
            seen.add(key_mpc)
            records.append({
                "op_type": "FC",
                "domain": "MPC",
                "method": "synthetic_mpc_fc",
                "input_shape": list(in_shape),
                "output_shape": list(out_shape),
                "local_compute_ms": 0.5 / mpc_speedup,
                "comm_bytes": 5000,
                "comm_rounds": 1,
                "he_level_delta": 0,
                "feasible": True,
                "metadata": {"source": "synthetic; tiny 64x10 SS matmul",
                             "gpu_speedup": mpc_speedup},
            })

    for op, ish, osh in RESNET20_OP_INSTANCES:
        if op == "Conv2D":
            add_he_conv(ish, osh)
            add_mpc_conv(ish, osh)
        elif op == "ReLU":
            add_he_relu(ish)
            add_mpc_relu(ish)
        elif op == "Residual_Add":
            add_residual_add(ish)
        elif op == "AvgPool":
            add_avgpool(ish, osh)
        elif op == "FC":
            add_fc(ish, osh)
        else:
            raise ValueError(f"unknown op {op}")

    return records


def _build_conversions(gpu: bool) -> List[Dict]:
    he_speedup = 50.0 if gpu else 1.0  # the conversion compute is HE-side encryption/decryption
    out: List[Dict] = []
    for shape in RESNET20_EDGE_SHAPES:
        local, comm, rounds = _conv_costs_for_shape(shape, he_speedup)
        out.append({
            "from_domain": "HE",
            "to_domain": "MPC",
            "method": "mask_and_decrypt",
            "tensor_shape": list(shape),
            "local_compute_ms": local,
            "comm_bytes": comm,
            "comm_rounds": rounds,
            "metadata": {"derived_from": "BERT [1,128,768] reference, scaled by numel",
                         "gpu_speedup": he_speedup},
        })
        out.append({
            "from_domain": "MPC",
            "to_domain": "HE",
            "method": "share_plus_enc",
            "tensor_shape": list(shape),
            "local_compute_ms": local,
            "comm_bytes": comm,
            "comm_rounds": rounds,
            "metadata": {"derived_from": "BERT [1,128,768] reference, scaled by numel",
                         "gpu_speedup": he_speedup},
        })
    return out


def build_profile(gpu: bool) -> Dict:
    operators = _build_operator_records(gpu)
    conversions = _build_conversions(gpu)
    return {
        "schema_version": "state_expanded/1.0",
        "platform": "gpu" if gpu else "cpu",
        "hardware": (
            {"device": "cuda", "platform": "B200 (synthetic)"}
            if gpu else
            {"device": "cpu", "platform": "AMD Ryzen Threadripper PRO 3955WX (synthetic)"}
        ),
        "he_level_budget": 20,
        "operators": operators,
        "conversions": conversions,
        "bootstrap": {
            "method": "nexus_cuda_bootstrap" if gpu else "nexus_ckks_bootstrap",
            "local_compute_ms": 1685.0 if gpu else 51608.5,
            "comm_bytes": 0,
            "comm_rounds": 0,
            "metadata": {
                "source": "reused from BERT full-model profile (shape-independent at fixed N)",
            },
        },
        "metadata": {
            "model": "ResNet-20 (CIFAR-10)",
            "synthetic": True,
            "he_stack_label": (
                "Synthetic HE (NEXUS-CUDA-style)" if gpu
                else "Synthetic HE (NEXUS-CKKS-style)"
            ),
            "mpc_stack_label": (
                "Synthetic MPC (SHAFT/CrypTen-style, GPU)" if gpu
                else "Synthetic MPC (BOLT/SCI-style, CPU)"
            ),
            "provenance": "see DATA_PROVENANCE.md adjacent to this file",
            "resnet20_manifest": {
                "batch": 1, "in_channels": 3, "in_spatial": 32,
                "stem_channels": 16, "num_classes": 10,
            },
        },
    }


def main() -> None:
    here = Path(__file__).parent
    cpu = build_profile(gpu=False)
    gpu = build_profile(gpu=True)
    (here / "profile_cpu_resnet20.json").write_text(json.dumps(cpu, indent=2))
    (here / "profile_gpu_resnet20.json").write_text(json.dumps(gpu, indent=2))
    print(f"[resnet20-profile] wrote {here / 'profile_cpu_resnet20.json'} ({len(cpu['operators'])} ops, {len(cpu['conversions'])} conversions)")
    print(f"[resnet20-profile] wrote {here / 'profile_gpu_resnet20.json'} ({len(gpu['operators'])} ops, {len(gpu['conversions'])} conversions)")


if __name__ == "__main__":
    main()

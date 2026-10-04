"""Assemble the measured built-in profiles from the raw measurements.

    python -m compiler.state_expanded.profiles.build_measured [raw_dir ...]

Inputs are merged over all raw directories (default: all of raw/*) and over supplementary
files that extend a name (e.g. sci_microbench_profile_qkv.json next to sci_microbench_profile.json):
for the same operator and shape (or conversion direction and size), the newest one is used.

Inputs (raw/<date>/, see its README.md):
  sci_microbench_profile.json        SCI/BOLT MPC operators (cipherflow profile --backend sci)
  shaft2pc_{cpu,cuda}_{TFP,TTP}.json SHAFT two-party operators (bench_shaft_2pc.py)
  conversion_seal_cpu.jsonl          HE<->MPC conversion with SEAL (native/conversion)
HE operators and bootstrapping come from the paper profiles (NEXUS reference kernels scaled
to each shape), marked as such: NEXUS has no single executable parameter set.

Outputs: profile_{cpu_nexus_sci,cpu_nexus_shaft,gpu_nexus_shaft}.json next to this file, and,
when the OpenFHE measurements are present (openfhe_ops_<mode>.jsonl, openfhe_bootstrap_<mode>.jsonl,
openfhe_conversion_<mpc>_<mode>.jsonl from native/openfhe), profile_cpu_openfhe_<mode>_<mpc>.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIRS: list = []


def _find(name: str) -> Path:
    for d in RAW_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in RAW_DIRS]}")


def _has(name: str) -> bool:
    return any((d / name).exists() for d in RAW_DIRS)


def _files(stem: str, ext: str) -> list:
    """<stem>.<ext> and <stem>_*.<ext> in every raw directory, oldest first (later ones win)."""
    out = []
    for d in reversed(RAW_DIRS):                      # RAW_DIRS: newest first
        out += [f for f in sorted(d.glob(f"{stem}*.{ext}"))
                if f.name == f"{stem}.{ext}" or f.name.startswith(f"{stem}_")]
    if not out:
        raise FileNotFoundError(f"{stem}*.{ext} not found in {[str(d) for d in RAW_DIRS]}")
    return out


def _where(stem: str, ext: str) -> str:
    return ", ".join(sorted({f"raw/{f.parent.name}" for f in _files(stem, ext)}))


def _jsonl(stem: str, key) -> list:
    """Merged lines of <stem>*.jsonl, one per key (the newest)."""
    rows = {}
    for f in _files(stem, "jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                k = key(r)
                if k is not None:
                    rows[k] = r
    return list(rows.values())
NEXUS_BOOT_SLOTS = 8192        # NEXUS bootstrapping demo: sparse_slots = 2^13 (CPU and CUDA)
CONV_SHAPES = {98304: [1, 128, 768], 196608: [1, 12, 128, 128], 294912: [3, 1, 128, 768],
               393216: [1, 128, 3072]}


def _load(name):
    return json.loads((HERE / name).read_text())


def _he_part(paper):
    he = [r for r in paper["operators"] if r["domain"] == "HE"]
    for r in he:
        r["metadata"] = dict(r.get("metadata") or {}, status="reference-scaled")
    # the paper's graph has no Q/K/V projections; NEXUS's MatMul scaling is linear in the output
    # width, so the projections (weights [H, 3H]) cost three output projections
    for r in [r for r in he if r["op_type"] == "Out_Projection"]:
        q = json.loads(json.dumps(r))
        q.update(op_type="Linear_QKV", output_shape=[3] + list(r["input_shape"]),
                 local_compute_ms=3 * r["local_compute_ms"])
        q["metadata"] = dict(r["metadata"], note="Q/K/V projections: 3 x Out_Projection (NEXUS MatMul "
                                                 "reference scaled to weights [H, 3H])")
        he.append(q)
    boot = dict(paper["bootstrap"])
    boot["metadata"] = dict(boot.get("metadata") or {}, slots=NEXUS_BOOT_SLOTS,
                            note="one bootstrap refreshes 8192 values (NEXUS demo, N=2^15 sparse); "
                                 "priced per ciphertext. The demo's modulus exceeds the 128-bit bound for N=2^15")
    return he, boot


def _conversions(raw, mpc):
    out = []
    for c in sorted(_jsonl("conversion_seal_cpu", lambda c: (c["direction"], c["numel"])),
                    key=lambda c: (c["numel"], c["direction"])):
        shape = CONV_SHAPES.get(c["numel"])
        if shape is None:
            continue
        he2mpc = c["direction"] == "HE->MPC"
        out.append({
            "from_domain": "HE" if he2mpc else "MPC", "to_domain": "MPC" if he2mpc else "HE",
            "method": "seal_mask_decrypt" if he2mpc else "seal_share_encrypt",
            "tensor_shape": shape,
            "local_compute_ms": c["server_ms"] + c["client_ms"],
            "comm_bytes": int(c["bytes"]), "comm_rounds": int(c["rounds"]),
            "metadata": {"source": "native/conversion/conversion_bench (SEAL 4.1, N=2^16, NEXUS modulus chain)",
                         "server_ms": c["server_ms"], "client_ms": c["client_ms"],
                         "ciphertexts": c["ciphertexts"], "max_abs_err": c["max_abs_err"],
                         "measured_for_ring": "SCI 2^37, 12 fractional bits",
                         "mpc_backend": mpc},
        })
    return out


def _merged_json(stem: str, key) -> dict:
    rows = {}
    for f in _files(stem, "json"):
        data = json.loads(f.read_text())
        for r in (data["operators"] if isinstance(data, dict) else data):
            rows[key(r)] = r
    return rows


def _shaft(raw, device):
    key = lambda r: (r["op"], tuple(r["shape_in"]))
    tfp = _merged_json(f"shaft2pc_{device}_TFP", key)
    ttp = {k[0]: r for k, r in _merged_json(f"shaft2pc_{device}_TTP", key).items()}
    recs = []
    for r in tfp.values():
        recs.append({
            "op_type": r["op"], "domain": "MPC", "method": f"shaft_crypten_2pc_{device}",
            "input_shape": r["shape_in"], "output_shape": r["shape_out"],
            "local_compute_ms": r["ms"], "comm_bytes": int(r["wire_bytes"]),
            "comm_rounds": int(max(r["rounds_parties"])), "he_level_delta": 0, "feasible": True,
            "metadata": {"source": "bench_shaft_2pc.py: two party processes on localhost"
                                   + (", one RTX 3090 each" if device == "cuda" else ""),
                         "provider": "TFP (online phase)", "ms_parties": r["ms_parties"],
                         "rounds_definition": "CrypTen communication calls per party (max over parties)",
                         "logical_bytes_parties": r["logical_bytes_parties"],
                         "offline_dealer_wire_bytes": int(ttp[r["op"]]["wire_bytes"] - r["wire_bytes"])},
        })
    return recs


def _sci(raw):
    merged = _merged_json("sci_microbench_profile",
                          lambda r: (r["op_type"], r["domain"], tuple(r["input_shape"]), tuple(r["output_shape"])))
    recs = [r for r in merged.values() if r["domain"] == "MPC" and 128 in r["input_shape"][1:3]]
    for r in recs:
        r["metadata"] = dict(r.get("metadata") or {},
                             rounds_definition="SCI IOPack rounds, summed over 2 threads and both parties")
    return recs


BERT_SHAPES = {        # op -> (input shape, output shape) for BERT-base at 128 tokens
    "Residual_Add": ([1, 128, 768], [1, 128, 768]), "GeLU": ([1, 128, 3072], [1, 128, 3072]),
    "LayerNorm": ([1, 128, 768], [1, 128, 768]), "Softmax": ([1, 12, 128, 128], [1, 12, 128, 128]),
    "Linear_QKV": ([1, 128, 768], [3, 1, 128, 768]),
    "Out_Projection": ([1, 128, 768], [1, 128, 768]), "FFN_Linear_1": ([1, 128, 768], [1, 128, 3072]),
    "FFN_Linear_2": ([1, 128, 3072], [1, 128, 768]),
    "Attention_QK_MatMul": ([3, 1, 128, 768], [1, 12, 128, 128]),
    "Attention_V_MatMul": ([1, 12, 128, 128], [1, 128, 768]),
}
GELU_DEGREE = "Chebyshev degree 27"     # 6 levels: FFN_Linear_2 can follow without a bootstrap


def _openfhe(raw: Path, mode: str, mpc: str):
    from cipherflow.measure import openfhe_record
    rows = []
    for f in _files(f"openfhe_ops_{mode}", "jsonl"):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    he = {}
    for r in rows:
        if r["op"] == "GeLU" and GELU_DEGREE not in r.get("approx", ""):
            continue
        if r["op"] in BERT_SHAPES:
            i, o = BERT_SHAPES[r["op"]]
            he[r["op"]] = json.loads(json.dumps(openfhe_record(r["op"], i, o, mode, r).to_json()))
    boot = json.loads(_find(f"openfhe_bootstrap_{mode}.jsonl").read_text().splitlines()[0])
    convs = []
    for c in sorted(_jsonl(f"openfhe_conversion_{mpc}_{mode}", lambda c: (c["direction"], c["numel"])),
                    key=lambda c: (c["numel"], c["direction"])):
        shape = CONV_SHAPES.get(c["numel"])
        if shape is None:
            continue
        he2mpc = c["direction"] == "HE->MPC"
        convs.append({"from_domain": "HE" if he2mpc else "MPC", "to_domain": "MPC" if he2mpc else "HE",
                      "method": "openfhe_mask_decrypt" if he2mpc else "openfhe_share_encrypt",
                      "tensor_shape": shape, "local_compute_ms": c["server_ms"] + c["client_ms"],
                      "comm_bytes": int(c["bytes"]), "comm_rounds": int(c["rounds"]),
                      "metadata": {"source": "native/openfhe/conv_bench", "server_ms": c["server_ms"],
                                   "client_ms": c["client_ms"], "ciphertexts": c["ciphertexts"],
                                   "max_abs_err": c["max_abs_err"],
                                   "note": "mask in the slot domain: protocol costs; a hardened version masks "
                                           "coefficients with noise flooding"}})
    bootstrap = {"method": f"openfhe_bootstrap_{mode}", "local_compute_ms": float(boot["ms"]), "comm_bytes": 0,
                 "comm_rounds": 0, "metadata": {"source": f"native/openfhe/he_ops ({mode})", "slots": 32768,
                                                "levels_after": boot["level_out"], "max_abs_err": boot["max_abs_err"],
                                                "approx": boot.get("approx", "")}}
    return list(he.values()), bootstrap, convs, int(boot["level_out"])


def build(raw: Path):
    raw = Path(raw)
    cpu_paper, gpu_paper = _load("profile_cpu_full_model_shaft.json"), _load("profile_gpu_full_model.json")
    cases = {
        "profile_cpu_nexus_sci.json": (cpu_paper, "sci", _sci(raw), "SCI/BOLT (CPU, 2PC, no third party)"),
        "profile_cpu_nexus_shaft.json": (cpu_paper, "shaft", _shaft(raw, "cpu"), "SHAFT (CPU, CrypTen 2PC, trusted dealer)"),
        "profile_gpu_nexus_shaft.json": (gpu_paper, "shaft", _shaft(raw, "cuda"), "SHAFT (GPU, CrypTen 2PC, trusted dealer)"),
    }
    for fname, (paper, mpc, mpc_recs, label) in cases.items():
        he, boot = _he_part(paper)
        ops = {r["op_type"] for r in he}
        assert {r["op_type"] for r in mpc_recs} == ops, (fname, sorted(ops - {r["op_type"] for r in mpc_recs}))
        out = {
            "schema_version": paper.get("schema_version", "state_expanded/1.0"),
            "platform": paper["platform"], "hardware": paper["hardware"],
            "he_level_budget": paper["he_level_budget"],
            "operators": he + mpc_recs,
            "conversions": _conversions(raw, mpc),
            "bootstrap": boot,
            "metadata": {
                "he_backend": "nexus", "mpc_backend": mpc,
                "he_stack_label": paper["metadata"].get("he_stack_label", ""), "mpc_stack_label": label,
                "he_source": "NEXUS reference kernels scaled to each shape (reference-scaled, not an executable pipeline)",
                "mpc_source": "measured, " + (_where("sci_microbench_profile", "json") if mpc == "sci" else
                                              _where(f"shaft2pc_{'cuda' if paper['platform'] == 'cuda' else 'cpu'}_TFP", "json")),
                "conversion_source": "measured with SEAL, " + _where("conversion_seal_cpu", "jsonl")
                                     + (" (CPU; GPU server side not measured)" if paper["platform"] == "cuda" else ""),
                "bootstrap_source": "NEXUS bootstrapping demo, per ciphertext of 8192 values",
            },
        }
        (HERE / fname).write_text(json.dumps(out, indent=2))
        print("wrote", fname, len(out["operators"]), "operators,", len(out["conversions"]), "conversions")

    for mode in ("fast", "precise"):
        if not _has(f"openfhe_ops_{mode}.jsonl"):
            continue
        for mpc, mpc_recs, label in (("sci", _sci(raw), "SCI/BOLT (CPU, 2PC, no third party)"),
                                     ("shaft", _shaft(raw, "cpu"), "SHAFT (CPU, CrypTen 2PC, trusted dealer)")):
            he, boot, convs, levels = _openfhe(raw, mode, mpc)
            fname = f"profile_cpu_openfhe_{mode}_{mpc}.json"
            out = {
                "schema_version": cpu_paper.get("schema_version", "state_expanded/1.0"),
                "platform": "cpu", "hardware": cpu_paper["hardware"], "he_level_budget": levels,
                "operators": he + mpc_recs, "conversions": convs, "bootstrap": boot,
                "metadata": {
                    "he_backend": f"openfhe-{mode}", "mpc_backend": mpc,
                    "he_stack_label": f"OpenFHE CKKS (CPU, N=2^16, 128-bit, {mode} bootstrapping)",
                    "mpc_stack_label": label,
                    "he_source": "measured with native/openfhe/he_ops, " + _where(f"openfhe_ops_{mode}", "jsonl"),
                    "mpc_source": "measured, " + (_where("sci_microbench_profile", "json") if mpc == "sci"
                                                  else _where("shaft2pc_cpu_TFP", "json")),
                    "conversion_source": "measured with native/openfhe/conv_bench, "
                                         + _where(f"openfhe_conversion_{mpc}_{mode}", "jsonl"),
                },
            }
            (HERE / fname).write_text(json.dumps(out, indent=2))
            print("wrote", fname, len(out["operators"]), "operators,", len(convs), "conversions")


if __name__ == "__main__":
    dirs = [Path(a) for a in sys.argv[1:]] or sorted((HERE / "raw").iterdir(), reverse=True)
    RAW_DIRS[:] = [d for d in dirs if d.is_dir()]
    build(RAW_DIRS[-1])

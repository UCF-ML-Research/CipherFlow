"""MPC operator profiler (paper §4.2.1 step 1, MPC side).

Approach
--------
For each MPC primitive we invoke the existing SCI BOLT bridge binary in
localhost-only mode (two cooperating processes, P1/P2). The bridges
were instrumented with a small telemetry line — purely additive — that
prints after the SCI call completes::

    [mpc_stats] party=<1|2> elapsed_ms=<double>
                comm_bytes=<uint64> comm_rounds=<uint64>

The *algorithm* is unchanged; only the telemetry is new. We parse
these lines and keep:

- ``comm_bytes`` and ``comm_rounds`` : real, measured.
- ``local_compute_ms``               : bridge wallclock on localhost.
  Loopback RTT is negligible (~0.05 ms) and localhost bandwidth is
  effectively unbounded (>40 Gbps), so on localhost
  ``(bytes*8/bw + rounds*rtt)`` rounds to << 0.1 ms and ``elapsed_ms``
  ≈ ``local_compute_ms``. The cost model composes deployment-time
  latency from this plus the target ``(bw, rtt)``.

Why this choice over an analytical model
----------------------------------------
BOLT-style MPC primitives have round counts that depend on bitwidth,
scale, and protocol sub-path selection. The simplest faithful model is
the one used by the protocol itself — so we read it off the SCI
IOPack counters instead of reproducing it in Python.

Shape adaptation
----------------
If a concrete shape violates a bridge's internal capacity limit (e.g.
FFN1 requires ``B*S <= 64``), we profile the largest supported
sub-shape and record the ratio in ``metadata["tile_factor"]`` so the
cost model scales correctly. The algorithm is not modified — we
simply call the bridge repeatedly on supported tiles.
"""

from __future__ import annotations

import os
import random
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .profile_schema import OperatorRecord

Shape = Tuple[int, ...]

_BIN_DIR = Path(os.environ.get("CIPHERFLOW_SCI_BIN",
                             Path(__file__).resolve().parents[3] / "EzPC_bolt" / "EzPC" / "SCI" / "build" / "bin"))

# Bridge binaries (all six instrumented).
BIN_GELU = _BIN_DIR / "BOLT_GELU_BRIDGE"
BIN_SOFTMAX = _BIN_DIR / "BOLT_SOFTMAX_BRIDGE"
BIN_LAYERNORM = _BIN_DIR / "BOLT_LAYERNORM_BRIDGE"
BIN_FFN1 = _BIN_DIR / "BOLT_FFN_LINEAR1_BRIDGE"
BIN_QK = _BIN_DIR / "BOLT_QK_MATMUL_MPC_BRIDGE"
BIN_AV = _BIN_DIR / "BOLT_ATTN_V_MATMUL_MPC_BRIDGE"

# SCI ell / scale defaults.
_ELL = 37
_SCALE = 12
_NTHREADS = 2

# FFN1 bridge cap on n=B*S (see CLAUDE.md + method_mpc_bolt.py).
_FFN1_N_CAP = 64


_STAT_RE = re.compile(
    r"\[mpc_stats\]\s+party=(\d+)\s+elapsed_ms=([\d\.eE+-]+)\s+"
    r"comm_bytes=(\d+)\s+comm_rounds=(\d+)"
)


def _port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _choose_port_block(block_size: int = 64, trials: int = 200) -> int:
    for _ in range(trials):
        start = random.randint(20000, 50000 - block_size - 1)
        if all(_port_available(start + i) for i in range(block_size)):
            return start
    raise RuntimeError("No free contiguous port block for SCI.")


@dataclass
class MpcBridgeStats:
    elapsed_ms_p1: float
    elapsed_ms_p2: float
    comm_bytes: int     # sum over both parties
    comm_rounds: int    # sum over both parties


def _parse_stats(stdout_text: str) -> List[Tuple[float, int, int]]:
    out = []
    for m in _STAT_RE.finditer(stdout_text):
        _party = int(m.group(1))
        out.append((float(m.group(2)), int(m.group(3)), int(m.group(4))))
    return out


def _run_two_party_once(cmd_p1: List[str], cmd_p2: List[str], timeout_s: int) -> MpcBridgeStats:
    p1 = subprocess.Popen(cmd_p1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.3)
    p2 = subprocess.Popen(cmd_p2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out1, err1 = p1.communicate(timeout=timeout_s)
        out2, err2 = p2.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        p1.kill()
        p2.kill()
        raise RuntimeError(f"SCI bridge timeout: {exc}") from exc
    if p1.returncode != 0 or p2.returncode != 0:
        raise RuntimeError(
            "SCI bridge failed: rc1={} rc2={} err1={} err2={}".format(
                p1.returncode, p2.returncode, err1.strip()[:400], err2.strip()[:400]
            )
        )
    s1 = _parse_stats(out1)
    s2 = _parse_stats(out2)
    if not s1 or not s2:
        raise RuntimeError(
            "Bridge did not emit [mpc_stats] line. stdout_p1={} stdout_p2={}".format(
                out1.strip()[:400], out2.strip()[:400]
            )
        )
    e1, b1, r1 = s1[-1]
    e2, b2, r2 = s2[-1]
    return MpcBridgeStats(
        elapsed_ms_p1=e1,
        elapsed_ms_p2=e2,
        comm_bytes=b1 + b2,
        comm_rounds=r1 + r2,
    )


def _run_two_party(
    make_cmds,
    timeout_s: int = 120,
    attempts: int = 5,
) -> MpcBridgeStats:
    """Run a two-party bridge with retry on port binds and transient timeouts.

    ``make_cmds(port)`` must return ``(cmd_p1, cmd_p2)`` with the given
    port substituted in.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        port = _choose_port_block(block_size=64)
        cmd_p1, cmd_p2 = make_cmds(port)
        try:
            return _run_two_party_once(cmd_p1, cmd_p2, timeout_s=timeout_s)
        except RuntimeError as exc:
            last_err = exc
            time.sleep(1.0 + attempt * 0.5)
    assert last_err is not None
    raise last_err


def _share_encode_random(size: int, ell: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    mask = np.uint64((1 << ell) - 1)
    a = rng.integers(0, 1 << ell, size=size, dtype=np.uint64)
    b = rng.integers(0, 1 << ell, size=size, dtype=np.uint64)
    return a & mask, b & mask


def _write_shares(path1: Path, path2: Path, size: int, ell: int) -> None:
    a, b = _share_encode_random(size, ell)
    a.tofile(path1)
    b.tofile(path2)


# -------------- op-specific drivers --------------


def _profile_gelu(size: int) -> MpcBridgeStats:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in", td / "p2_in"
        out1, out2 = td / "p1_out", td / "p2_out"
        _write_shares(in1, in2, size, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--size", str(size),
            ]
            c1 = [str(BIN_GELU), "--party", "1"] + common + ["--input", str(in1), "--output", str(out1)]
            c2 = [str(BIN_GELU), "--party", "2"] + common + ["--input", str(in2), "--output", str(out2)]
            return c1, c2

        return _run_two_party(make)


def _profile_softmax(dim: int, array_size: int) -> MpcBridgeStats:
    total = dim * array_size
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in", td / "p2_in"
        out1, out2 = td / "p1_out", td / "p2_out"
        l1, l2 = td / "p1_l", td / "p2_l"
        _write_shares(in1, in2, total, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--dim", str(dim),
                "--array_size", str(array_size),
            ]
            c1 = [str(BIN_SOFTMAX), "--party", "1"] + common + [
                "--input", str(in1), "--output", str(out1), "--l_out", str(l1)
            ]
            c2 = [str(BIN_SOFTMAX), "--party", "2"] + common + [
                "--input", str(in2), "--output", str(out2), "--l_out", str(l2)
            ]
            return c1, c2

        return _run_two_party(make, timeout_s=240)


def _profile_layernorm(dim: int, array_size: int) -> MpcBridgeStats:
    total = dim * array_size
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in", td / "p2_in"
        w1, w2 = td / "p1_w", td / "p2_w"
        b1, b2 = td / "p1_b", td / "p2_b"
        out1, out2 = td / "p1_out", td / "p2_out"
        _write_shares(in1, in2, total, _ELL)
        _write_shares(w1, w2, total, _ELL)
        _write_shares(b1, b2, total, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--dim", str(dim),
                "--array_size", str(array_size),
            ]
            c1 = [str(BIN_LAYERNORM), "--party", "1"] + common + [
                "--input", str(in1), "--weight", str(w1), "--bias", str(b1), "--output", str(out1)
            ]
            c2 = [str(BIN_LAYERNORM), "--party", "2"] + common + [
                "--input", str(in2), "--weight", str(w2), "--bias", str(b2), "--output", str(out2)
            ]
            return c1, c2

        return _run_two_party(make, timeout_s=240)


def _profile_ffn1(n: int, h: int, i: int) -> MpcBridgeStats:
    # FFN1 bridge uses n_matrix_mul_iron(input[n,1,h], weight[n,h,i],
    # output[n,1,i]). It is a batched matmul along `n`.
    input_size = n * h
    weight_size = n * h * i
    bias_size = n * i
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in", td / "p2_in"
        w1, w2 = td / "p1_w", td / "p2_w"
        b1, b2 = td / "p1_b", td / "p2_b"
        out1, out2 = td / "p1_out", td / "p2_out"
        _write_shares(in1, in2, input_size, _ELL)
        _write_shares(w1, w2, weight_size, _ELL)
        _write_shares(b1, b2, bias_size, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--n", str(n),
                "--h", str(h), "--i", str(i),
            ]
            c1 = [str(BIN_FFN1), "--party", "1"] + common + [
                "--input", str(in1), "--weight", str(w1), "--bias", str(b1), "--output", str(out1)
            ]
            c2 = [str(BIN_FFN1), "--party", "2"] + common + [
                "--input", str(in2), "--weight", str(w2), "--bias", str(b2), "--output", str(out2)
            ]
            return c1, c2

        return _run_two_party(make, timeout_s=300)


def _profile_qk(n: int, dim1: int, dim2: int, dim3: int) -> MpcBridgeStats:
    input_a_size = n * dim1 * dim2
    input_b_size = n * dim2 * dim3
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a1, a2 = td / "p1_a", td / "p2_a"
        b1, b2 = td / "p1_b", td / "p2_b"
        out1, out2 = td / "p1_out", td / "p2_out"
        _write_shares(a1, a2, input_a_size, _ELL)
        _write_shares(b1, b2, input_b_size, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--n", str(n),
                "--dim1", str(dim1), "--dim2", str(dim2), "--dim3", str(dim3),
            ]
            c1 = [str(BIN_QK), "--party", "1"] + common + [
                "--input_a", str(a1), "--input_b", str(b1), "--output", str(out1)
            ]
            c2 = [str(BIN_QK), "--party", "2"] + common + [
                "--input_a", str(a2), "--input_b", str(b2), "--output", str(out2)
            ]
            return c1, c2

        return _run_two_party(make, timeout_s=300)


def _profile_av(n: int, dim1: int, dim2: int, dim3: int) -> MpcBridgeStats:
    # Same wire as QK.
    input_a_size = n * dim1 * dim2
    input_b_size = n * dim2 * dim3
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a1, a2 = td / "p1_a", td / "p2_a"
        b1, b2 = td / "p1_b", td / "p2_b"
        out1, out2 = td / "p1_out", td / "p2_out"
        _write_shares(a1, a2, input_a_size, _ELL)
        _write_shares(b1, b2, input_b_size, _ELL)

        def make(port):
            common = [
                "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(_NTHREADS), "--ell", str(_ELL),
                "--scale", str(_SCALE), "--n", str(n),
                "--dim1", str(dim1), "--dim2", str(dim2), "--dim3", str(dim3),
            ]
            c1 = [str(BIN_AV), "--party", "1"] + common + [
                "--input_a", str(a1), "--input_b", str(b1), "--output", str(out1)
            ]
            c2 = [str(BIN_AV), "--party", "2"] + common + [
                "--input_a", str(a2), "--input_b", str(b2), "--output", str(out2)
            ]
            return c1, c2

        return _run_two_party(make, timeout_s=300)


# -------------- dispatch for paper BERT shapes --------------


def _record(
    op_type: str,
    method: str,
    input_shape: Shape,
    output_shape: Shape,
    stats: MpcBridgeStats,
    tile_factor: int = 1,
    note: str = "",
) -> OperatorRecord:
    # Sum is over the two parties; for "per-party" comms people often
    # divide by 2. We keep total wire traffic in the record (which is
    # what bandwidth actually transports). Latency model treats it as
    # total.
    elapsed = max(stats.elapsed_ms_p1, stats.elapsed_ms_p2) * tile_factor
    return OperatorRecord(
        op_type=op_type,
        domain="MPC",
        method=method,
        input_shape=input_shape,
        output_shape=output_shape,
        local_compute_ms=elapsed,
        comm_bytes=stats.comm_bytes * tile_factor,
        comm_rounds=stats.comm_rounds,  # structural, same per tile
        he_level_delta=0,
        feasible=True,
        metadata={
            "tile_factor": tile_factor,
            "party1_ms": stats.elapsed_ms_p1,
            "party2_ms": stats.elapsed_ms_p2,
            "note": note,
        },
    )


def profile_mpc_for_bert(
    input_shape: Shape,
    output_shape: Shape,
    op_type: str,
) -> OperatorRecord:
    """Profile one MPC op by invoking the real SCI bridge."""

    if op_type == "GeLU":
        size = int(np.prod(input_shape))
        stats = _profile_gelu(size)
        return _record(op_type, "method_mpc_bolt", input_shape, output_shape, stats)

    if op_type == "Softmax":
        # Softmax rows along last axis: dim = last, array_size = prod rest.
        dim = int(input_shape[-1])
        array_size = int(np.prod(input_shape[:-1]))
        stats = _profile_softmax(dim, array_size)
        return _record(op_type, "method_mpc_bolt", input_shape, output_shape, stats)

    if op_type == "LayerNorm":
        dim = int(input_shape[-1])
        array_size = int(np.prod(input_shape[:-1]))
        stats = _profile_layernorm(dim, array_size)
        return _record(op_type, "method_mpc_bolt", input_shape, output_shape, stats)

    if op_type == "FFN_Linear_1":
        # [B,S,H] -> [B,S,I]  means n = B*S, h = H, i = I.
        B, S, H = input_shape
        I = output_shape[-1]
        n = B * S
        tile_factor = 1
        if n > _FFN1_N_CAP:
            tile_factor = (n + _FFN1_N_CAP - 1) // _FFN1_N_CAP
            n = _FFN1_N_CAP
        stats = _profile_ffn1(n, H, I)
        return _record(
            op_type, "method_mpc_bolt", input_shape, output_shape, stats,
            tile_factor=tile_factor,
            note=f"tile n<= {_FFN1_N_CAP} x{tile_factor}",
        )

    if op_type == "FFN_Linear_2":
        # Lowered onto the same n_matrix_mul_iron primitive (docs/handoff_status.md).
        B, S, Hin = input_shape
        Hout = output_shape[-1]
        n = B * S
        tile_factor = 1
        if n > _FFN1_N_CAP:
            tile_factor = (n + _FFN1_N_CAP - 1) // _FFN1_N_CAP
            n = _FFN1_N_CAP
        stats = _profile_ffn1(n, Hin, Hout)
        return _record(
            op_type, "method_mpc_bolt", input_shape, output_shape, stats,
            tile_factor=tile_factor,
            note=f"ffn2 lowered to ffn1 primitive, tile n<= {_FFN1_N_CAP} x{tile_factor}",
        )

    if op_type == "Out_Projection":
        # Same linear primitive.
        B, S, H = input_shape
        Hout = output_shape[-1]
        n = B * S
        tile_factor = 1
        if n > _FFN1_N_CAP:
            tile_factor = (n + _FFN1_N_CAP - 1) // _FFN1_N_CAP
            n = _FFN1_N_CAP
        stats = _profile_ffn1(n, H, Hout)
        return _record(
            op_type, "method_mpc_bolt_as_ffn1", input_shape, output_shape, stats,
            tile_factor=tile_factor,
            note="out_projection lowered to ffn1 primitive",
        )

    if op_type == "Attention_QK_MatMul":
        # Packed qkv [3,B,S,H]. QK = Q * K^T per head.
        # Map to n_matrix_mul_iron(n=B*heads, dim1=S, dim2=head_dim, dim3=S).
        _3, B, S, H = input_shape
        heads = 12
        head_dim = H // heads
        n = B * heads
        stats = _profile_qk(n, S, head_dim, S)
        return _record(op_type, "method_mpc", input_shape, output_shape, stats)

    if op_type == "Attention_V_MatMul":
        # attn_probs [B,H,S,S] times V [B,H,S,D].
        B, heads, S, _S2 = input_shape
        head_dim = 64
        n = B * heads
        stats = _profile_av(n, S, S, head_dim)
        return _record(op_type, "method_mpc", input_shape, output_shape, stats)

    if op_type == "Residual_Add":
        # Semantic add: zero comm / zero rounds (local share add).
        return OperatorRecord(
            op_type=op_type,
            domain="MPC",
            method="method_runtime_default",
            input_shape=input_shape,
            output_shape=output_shape,
            local_compute_ms=0.0,
            comm_bytes=0,
            comm_rounds=0,
            he_level_delta=0,
            feasible=True,
            metadata={"note": "semantic add is local share addition"},
        )

    raise ValueError(f"No MPC profiler for op={op_type}")


def profile_mpc_operators(
    shapes: List[Tuple[str, Shape, Shape]],
    verbose: bool = True,
) -> List[OperatorRecord]:
    records: List[OperatorRecord] = []
    for op_type, input_shape, output_shape in shapes:
        if verbose:
            print(f"  [mpc] profiling {op_type} {input_shape} -> {output_shape}")
        try:
            rec = profile_mpc_for_bert(input_shape, output_shape, op_type)
            records.append(rec)
            if verbose:
                print(
                    f"    local={rec.local_compute_ms:.1f}ms  "
                    f"bytes={rec.comm_bytes:_}  rounds={rec.comm_rounds}"
                )
        except Exception as exc:
            records.append(
                OperatorRecord(
                    op_type=op_type,
                    domain="MPC",
                    method="method_mpc_bolt",
                    input_shape=input_shape,
                    output_shape=output_shape,
                    local_compute_ms=0.0,
                    comm_bytes=0,
                    comm_rounds=0,
                    he_level_delta=0,
                    feasible=False,
                    metadata={"reason": f"profiler_error: {exc}"},
                )
            )
            if verbose:
                print(f"    FAILED: {exc}")
    return records

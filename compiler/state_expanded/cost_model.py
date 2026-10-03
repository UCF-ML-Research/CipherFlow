"""Bandwidth-aware cost composition.

Paper §4.2.1 decomposes latency into hardware-local compute plus
communication that depends on the deployment (bandwidth + RTT). We store
the hardware-dependent terms in a profile and compose the final latency
at planning time, so a single profile can be reused across every
``(bandwidth, RTT)`` setting.

Latency model for any measured record ``r`` under ``(bw_bps, rtt_ms)``:

    latency_ms(r) = r.local_compute_ms
                  + (r.comm_bytes * 8 / bw_bps) * 1000
                  + r.comm_rounds * rtt_ms

HE-only records have ``comm_bytes = 0`` and ``comm_rounds = 0`` so their
latency reduces to ``local_compute_ms``. Bootstrap records are a pure
compute term as well. Cross-domain conversions use the same formula:
the HE side contributes its compute, the MPC side contributes bytes and
rounds (mask-and-decrypt).

The cost model deliberately keeps no internal state: a single profile is
queried with (op_type, domain, shape) or (from_domain, to_domain, shape)
and the network setting is passed explicitly at each call. This matches
the paper's §4.2.1 "hardware-aware" claim — one profile, many
deployments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .profile_schema import (
    BootstrapRecord,
    ConversionRecord,
    LatencyProfile,
    OperatorRecord,
)

Shape = Tuple[int, ...]


@dataclass(frozen=True)
class NetworkSetting:
    """Deployment-time network parameters.

    ``bandwidth_bps``: link bandwidth in bits per second. For LAN use
    3e9 (3 Gbps); for WAN use 1e7 (10 Mbps).
    ``rtt_ms``: round-trip time in ms, added per communication round.
    """

    bandwidth_bps: float
    rtt_ms: float

    def label(self) -> str:
        if self.bandwidth_bps >= 1e9:
            bw = f"{self.bandwidth_bps / 1e9:g}Gbps"
        elif self.bandwidth_bps >= 1e6:
            bw = f"{self.bandwidth_bps / 1e6:g}Mbps"
        else:
            bw = f"{self.bandwidth_bps / 1e3:g}kbps"
        return f"{bw}/RTT{self.rtt_ms:g}ms"


def compose_latency(
    local_compute_ms: float, comm_bytes: int, comm_rounds: int, net: NetworkSetting
) -> float:
    """Return the deployment-time latency in ms for a single record."""
    if net.bandwidth_bps <= 0:
        raise ValueError("bandwidth_bps must be positive")
    comm_ms = (comm_bytes * 8.0 / net.bandwidth_bps) * 1000.0
    return float(local_compute_ms) + comm_ms + comm_rounds * float(net.rtt_ms)


@dataclass(frozen=True)
class CostEstimate:
    latency_ms: float
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    resolution: str  # "exact" | "nearest" | "infeasible"


def _shape_size(shape: Iterable[int]) -> int:
    n = 1
    for d in shape:
        n *= max(1, int(d))
    return n


class StateExpandedCostModel:
    """Look up record by (op, shape) and compose latency for a network.

    Resolution order for ``estimate_operator``:
    1. Exact match on (op_type, domain, input_shape, output_shape).
    2. Nearest match within the same (op_type, domain) by output numel
       distance, scaled linearly by numel ratio (protocol costs in both
       HE and MPC scale ~linearly with output element count, which is
       a well-established simplification used by BOLT/BumbleBee cost
       models).
    3. If the record is marked infeasible (``feasible=False``) the
       estimate is returned with ``resolution="infeasible"`` and the
       caller must treat the HE transition as disallowed.

    This is explicit rather than clever: the profile is the source of
    truth, extrapolation is logged, and infeasibility is never silently
    masked.
    """

    def __init__(self, profile: LatencyProfile) -> None:
        self.profile = profile

    # ---------- operator cost ----------

    def _find_operator(
        self, op_type: str, domain: str, input_shape: Shape, output_shape: Shape
    ) -> Tuple[OperatorRecord, str]:
        exact = self.profile.find_operator(op_type, domain, input_shape, output_shape)
        if exact is not None:
            return exact, "exact"
        candidates = self.profile.operators_for(op_type, domain)
        if not candidates:
            raise KeyError(
                f"No operator record for op={op_type}, domain={domain}"
            )
        target_numel = _shape_size(output_shape) or _shape_size(input_shape)
        nearest = min(
            candidates,
            key=lambda r: abs(_shape_size(r.output_shape) - target_numel),
        )
        return nearest, "nearest"

    def estimate_operator(
        self,
        op_type: str,
        domain: str,
        input_shape: Shape,
        output_shape: Shape,
        net: NetworkSetting,
    ) -> CostEstimate:
        rec, how = self._find_operator(op_type, domain, input_shape, output_shape)
        if not rec.feasible:
            return CostEstimate(
                latency_ms=float("inf"),
                local_compute_ms=rec.local_compute_ms,
                comm_bytes=rec.comm_bytes,
                comm_rounds=rec.comm_rounds,
                resolution="infeasible",
            )
        # Scale local compute + comm bytes by numel ratio when nearest-matching.
        local = rec.local_compute_ms
        comm_bytes = rec.comm_bytes
        comm_rounds = rec.comm_rounds
        if how == "nearest":
            ref = _shape_size(rec.output_shape) or 1
            tgt = _shape_size(output_shape) or ref
            ratio = float(tgt) / float(ref)
            local = local * ratio
            comm_bytes = int(round(comm_bytes * ratio))
            # Round count is structural in protocol, keep unchanged.
        lat = compose_latency(local, comm_bytes, comm_rounds, net)
        return CostEstimate(
            latency_ms=lat,
            local_compute_ms=local,
            comm_bytes=comm_bytes,
            comm_rounds=comm_rounds,
            resolution=how,
        )

    # ---------- conversion cost ----------

    def _find_conversion(
        self, from_domain: str, to_domain: str, tensor_shape: Shape
    ) -> Tuple[ConversionRecord, str]:
        exact = self.profile.find_conversion(from_domain, to_domain, tensor_shape)
        if exact is not None:
            return exact, "exact"
        candidates = self.profile.conversions_for(from_domain, to_domain)
        if not candidates:
            raise KeyError(
                f"No conversion record for {from_domain}->{to_domain}"
            )
        target_numel = _shape_size(tensor_shape)
        nearest = min(
            candidates,
            key=lambda r: abs(_shape_size(r.tensor_shape) - target_numel),
        )
        return nearest, "nearest"

    def estimate_conversion(
        self,
        from_domain: str,
        to_domain: str,
        tensor_shape: Shape,
        net: NetworkSetting,
    ) -> CostEstimate:
        if from_domain == to_domain:
            return CostEstimate(
                latency_ms=0.0,
                local_compute_ms=0.0,
                comm_bytes=0,
                comm_rounds=0,
                resolution="exact",
            )
        rec, how = self._find_conversion(from_domain, to_domain, tensor_shape)
        local = rec.local_compute_ms
        comm_bytes = rec.comm_bytes
        comm_rounds = rec.comm_rounds
        if how == "nearest":
            ref = _shape_size(rec.tensor_shape) or 1
            tgt = _shape_size(tensor_shape) or ref
            ratio = float(tgt) / float(ref)
            local = local * ratio
            comm_bytes = int(round(comm_bytes * ratio))
        lat = compose_latency(local, comm_bytes, comm_rounds, net)
        return CostEstimate(
            latency_ms=lat,
            local_compute_ms=local,
            comm_bytes=comm_bytes,
            comm_rounds=comm_rounds,
            resolution=how,
        )

    # ---------- bootstrap cost ----------

    def estimate_bootstrap(self, net: NetworkSetting) -> CostEstimate:
        rec = self.profile.bootstrap
        if rec is None:
            # Bootstrapping unavailable on this profile: caller should
            # fall back to HE->MPC->HE detour (documented in
            # `_cost_signature.py:BootstrapUnsupportedError`).
            return CostEstimate(
                latency_ms=float("inf"),
                local_compute_ms=0.0,
                comm_bytes=0,
                comm_rounds=0,
                resolution="infeasible",
            )
        lat = compose_latency(
            rec.local_compute_ms, rec.comm_bytes, rec.comm_rounds, net
        )
        return CostEstimate(
            latency_ms=lat,
            local_compute_ms=rec.local_compute_ms,
            comm_bytes=rec.comm_bytes,
            comm_rounds=rec.comm_rounds,
            resolution="exact",
        )

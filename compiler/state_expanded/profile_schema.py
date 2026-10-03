"""Profile schema — matches paper §4.2.1 latency-profile description.

Three record kinds:

- ``OperatorRecord``:   per (op_type, domain, input_shape, output_shape).
  ``local_compute_ms`` is wallclock of the compute path on the target
  platform; ``comm_bytes`` + ``comm_rounds`` are zero for HE and
  non-zero for MPC. ``he_level_delta`` is ``δ_i`` from paper Table 1.

- ``ConversionRecord``: per (from_domain, to_domain, tensor_shape).
  Both directions of HE<->MPC conversion are recorded separately.

- ``BootstrapRecord``:  a single entry per profile describing the
  bootstrapping cost under a fixed ciphertext modulus chain. Paper
  §4.2.1: "we also profile bootstrapping latency under a fixed
  ciphertext modulus q".

The profile is serialised as JSON so it can be regenerated on each
target platform (Threadripper CPU / B200 GPU) and reused across every
deployment (bandwidth, RTT) setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

Shape = Tuple[int, ...]

# HE level budget used throughout the compiler. Chosen to match the
# NEXUS COEFF_MODULI chain length in NEXUS/src/main.cpp (20 levels).
HE_LEVEL_BUDGET: int = 20


def _shape_tuple(shape: Iterable[int]) -> Shape:
    return tuple(int(d) for d in shape)


@dataclass(frozen=True)
class OperatorRecord:
    op_type: str
    domain: str  # "HE" | "MPC"
    method: str
    input_shape: Shape
    output_shape: Shape
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    he_level_delta: int = 0
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        d["output_shape"] = list(self.output_shape)
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "OperatorRecord":
        return cls(
            op_type=str(d["op_type"]),
            domain=str(d["domain"]),
            method=str(d.get("method", "")),
            input_shape=_shape_tuple(d["input_shape"]),
            output_shape=_shape_tuple(d["output_shape"]),
            local_compute_ms=float(d["local_compute_ms"]),
            comm_bytes=int(d.get("comm_bytes", 0)),
            comm_rounds=int(d.get("comm_rounds", 0)),
            he_level_delta=int(d.get("he_level_delta", 0)),
            feasible=bool(d.get("feasible", True)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(frozen=True)
class ConversionRecord:
    from_domain: str
    to_domain: str
    method: str
    tensor_shape: Shape
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tensor_shape"] = list(self.tensor_shape)
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ConversionRecord":
        return cls(
            from_domain=str(d["from_domain"]),
            to_domain=str(d["to_domain"]),
            method=str(d.get("method", "")),
            tensor_shape=_shape_tuple(d["tensor_shape"]),
            local_compute_ms=float(d["local_compute_ms"]),
            comm_bytes=int(d.get("comm_bytes", 0)),
            comm_rounds=int(d.get("comm_rounds", 0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass(frozen=True)
class BootstrapRecord:
    method: str
    local_compute_ms: float
    comm_bytes: int = 0
    comm_rounds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "BootstrapRecord":
        return cls(
            method=str(d.get("method", "")),
            local_compute_ms=float(d["local_compute_ms"]),
            comm_bytes=int(d.get("comm_bytes", 0)),
            comm_rounds=int(d.get("comm_rounds", 0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class LatencyProfile:
    platform: str  # "cpu" | "gpu"
    hardware: Dict[str, Any]
    he_level_budget: int
    operators: List[OperatorRecord] = field(default_factory=list)
    conversions: List[ConversionRecord] = field(default_factory=list)
    bootstrap: Optional[BootstrapRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------- lookup helpers ----------

    def operators_for(self, op_type: str, domain: str) -> List[OperatorRecord]:
        return [r for r in self.operators if r.op_type == op_type and r.domain == domain]

    def find_operator(
        self, op_type: str, domain: str, input_shape: Shape, output_shape: Shape
    ) -> Optional[OperatorRecord]:
        is_ = _shape_tuple(input_shape)
        os_ = _shape_tuple(output_shape)
        for r in self.operators_for(op_type, domain):
            if r.input_shape == is_ and r.output_shape == os_:
                return r
        return None

    def conversions_for(self, from_domain: str, to_domain: str) -> List[ConversionRecord]:
        return [
            r
            for r in self.conversions
            if r.from_domain == from_domain and r.to_domain == to_domain
        ]

    def find_conversion(
        self, from_domain: str, to_domain: str, tensor_shape: Shape
    ) -> Optional[ConversionRecord]:
        ts = _shape_tuple(tensor_shape)
        for r in self.conversions_for(from_domain, to_domain):
            if r.tensor_shape == ts:
                return r
        return None

    # ---------- JSON ----------

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": "state_expanded/1.0",
            "platform": self.platform,
            "hardware": self.hardware,
            "he_level_budget": self.he_level_budget,
            "operators": [r.to_json() for r in self.operators],
            "conversions": [r.to_json() for r in self.conversions],
            "bootstrap": self.bootstrap.to_json() if self.bootstrap else None,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LatencyProfile":
        payload = json.loads(Path(path).read_text())
        return cls(
            platform=str(payload["platform"]),
            hardware=dict(payload.get("hardware", {})),
            he_level_budget=int(payload.get("he_level_budget", HE_LEVEL_BUDGET)),
            operators=[OperatorRecord.from_json(r) for r in payload.get("operators", [])],
            conversions=[ConversionRecord.from_json(r) for r in payload.get("conversions", [])],
            bootstrap=(
                BootstrapRecord.from_json(payload["bootstrap"])
                if payload.get("bootstrap") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )

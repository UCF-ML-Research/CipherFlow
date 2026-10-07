"""Calibration: value statistics of a model's operators on real inputs (cipherflow.runtime.calibrate).

compile(..., calibration=cal) checks the HE methods' value contracts against them (see
cipherflow.kernel_params): an operator whose calibrated inputs fall outside an HE kernel's
approximation is kept out of HE, with the reason in plan.warnings.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Union


@dataclass
class Calibration:
    """node id -> statistics over the calibration inputs:
    in_min / in_max      operator inputs (Softmax: the scores)
    out_absmax           largest |output|
    out_absmean          largest |mean of the output| (bootstrapping is sensitive to it)
    var_min / var_max    LayerNorm: variance over the normalized axis, per token
    lme_min / lme_max    Softmax: log of the row mean of exp(scores), per row
    rowmax_min           Softmax: smallest row maximum"""
    model: str
    samples: int
    nodes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_json(self, path=None) -> str:
        text = json.dumps(asdict(self), indent=1)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def load(cls, src: Union[str, Path, "Calibration"]) -> "Calibration":
        if isinstance(src, Calibration):
            return src
        text = str(src)
        if not text.lstrip().startswith("{"):
            text = Path(src).read_text()
        d = json.loads(text)
        return cls(d["model"], int(d["samples"]), d.get("nodes", {}))

    def merge(self, other: "Calibration") -> "Calibration":
        nodes = {k: dict(v) for k, v in self.nodes.items()}
        for nid, s in other.nodes.items():
            mine = nodes.setdefault(nid, {})
            for k, v in s.items():
                if k not in mine:
                    mine[k] = v
                elif k.endswith("_min"):
                    mine[k] = min(mine[k], v)
                else:
                    mine[k] = max(mine[k], v)
        return Calibration(self.model, self.samples + other.samples, nodes)

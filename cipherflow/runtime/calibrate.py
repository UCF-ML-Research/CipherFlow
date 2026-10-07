"""calibrate(module, inputs): run a model in plaintext on real inputs and record, per CipherFlow
operator, the statistics the HE kernels' value contracts need (see cipherflow.calibration)."""
from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, Optional

import torch
import torch.fx as fx

from ..calibration import Calibration
from ..frontend import trace_torch


class _Observe(fx.Interpreter):
    def __init__(self, tr, stats: Dict[str, Dict[str, float]]):
        super().__init__(tr.gm)
        self.tr, self.stats = tr, stats

    def _put(self, nid, key, value, mode):
        s = self.stats.setdefault(nid, {})
        s[key] = value if key not in s else (min(s[key], value) if mode == "min" else max(s[key], value))

    def run_node(self, n):
        out = super().run_node(n)
        owner = self.tr.owner.get(n.name)
        if self.tr.kind.get(n.name) != "op" or owner is None or owner.startswith("MHA:"):
            return out
        op = self.tr.op[n.name]
        ins = [self.env[a] for a in n.all_input_nodes
               if isinstance(self.env.get(a), torch.Tensor) and self.env[a].is_floating_point()
               and self.tr.kind.get(a.name) != "const"]
        x = ins[0] if ins else None
        if x is not None:
            self._put(owner, "in_min", float(x.min()), "min")
            self._put(owner, "in_max", float(x.max()), "max")
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            self._put(owner, "out_absmax", float(out.abs().max()), "max")
            self._put(owner, "out_absmean", float(out.mean().abs()), "max")
        if op == "LayerNorm" and x is not None:
            var = x.var(-1, unbiased=False)
            self._put(owner, "var_min", float(var.min()), "min")
            self._put(owner, "var_max", float(var.max()), "max")
        if op == "Softmax" and x is not None:
            lme = torch.logsumexp(x, -1) - math.log(x.shape[-1])
            self._put(owner, "lme_min", float(lme.min()), "min")
            self._put(owner, "lme_max", float(lme.max()), "max")
            self._put(owner, "rowmax_min", float(x.max(-1).values.min()), "min")
        return out


def calibrate(module, inputs: Iterable, *, concrete_args=None, qkv_projection: bool = True,
              model: Optional[str] = None) -> Calibration:
    """`inputs`: an iterable of example inputs (tensors or tuples), e.g. the hidden states of real
    sentences. Node ids match cf.models.from_torch(module, ...)."""
    m = copy.deepcopy(module).double().eval()
    stats: Dict[str, Dict[str, float]] = {}
    traces = {}
    count = 0
    with torch.no_grad():
        for x in inputs:
            xs = x if isinstance(x, tuple) else (x,)
            xs = tuple(t.double() if isinstance(t, torch.Tensor) and t.is_floating_point() else t for t in xs)
            key = tuple(tuple(t.shape) if isinstance(t, torch.Tensor) else None for t in xs)
            if key not in traces:
                traces[key] = trace_torch(m, xs, concrete_args=concrete_args, qkv_projection=qkv_projection)
            tr = traces[key]
            run_args = list(xs) + list(tr.run_args[len(xs):])
            _Observe(tr, stats).run(*run_args)
            count += 1
    return Calibration(model or type(module).__name__, count, stats)

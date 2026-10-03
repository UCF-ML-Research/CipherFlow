
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FfnPackingContract:
    tensor_shape: tuple[int, ...]
    hidden_size: int
    tokens: int
    layout_family: str = "ffn"
    layout_name: str = "ffn_row_major"
    max_tokens: int = 4096
    allowed_hidden_sizes: tuple[int, ...] = (64, 768)

    def as_meta(self) -> dict[str, Any]:
        return {
            "layout_family": self.layout_family,
            "layout_name": self.layout_name,
            "tensor_shape": list(self.tensor_shape),
            "hidden_size": self.hidden_size,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "allowed_hidden_sizes": list(self.allowed_hidden_sizes),
        }


def supports_ffn_conversion_shape(shape: tuple[int, ...], *, max_tokens: int = 4096) -> bool:
    if len(shape) != 3:
        return False
    batch, seq_len, hidden = shape
    if hidden not in (64, 768):
        return False
    return batch > 0 and seq_len > 0 and batch * seq_len <= max_tokens


def build_ffn_packing_contract(
    shape: tuple[int, ...],
    *,
    max_tokens: int = 4096,
    allowed_hidden_sizes: tuple[int, ...] = (64, 768),
) -> FfnPackingContract:
    if len(shape) != 3:
        raise ValueError(f"FFN restricted packing expects [B, S, H], got shape={list(shape)}")
    batch, seq_len, hidden = shape
    tokens = batch * seq_len
    if hidden not in allowed_hidden_sizes:
        raise ValueError(
            "FFN restricted packing only supports hidden sizes "
            f"{list(allowed_hidden_sizes)}, got H={hidden}"
        )
    if tokens <= 0 or tokens > max_tokens:
        raise ValueError(
            f"FFN restricted packing requires 1 <= B*S <= {max_tokens}, got B*S={tokens}"
        )
    return FfnPackingContract(
        tensor_shape=shape,
        hidden_size=hidden,
        tokens=tokens,
        max_tokens=max_tokens,
        allowed_hidden_sizes=allowed_hidden_sizes,
    )


def prepare_he_tensor_for_mpc_ffn(x: np.ndarray, contract: FfnPackingContract) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if tuple(arr.shape) != contract.tensor_shape:
        raise ValueError(
            "FFN HE->MPC packing shape mismatch: "
            f"expected={list(contract.tensor_shape)} got={list(arr.shape)}"
        )
    return np.array(arr, dtype=np.float64, copy=True, order="C")


def prepare_mpc_tensor_for_he_ffn(x: np.ndarray, contract: FfnPackingContract) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if tuple(arr.shape) != contract.tensor_shape:
        raise ValueError(
            "FFN MPC->HE packing shape mismatch: "
            f"expected={list(contract.tensor_shape)} got={list(arr.shape)}"
        )
    return np.array(arr, dtype=np.float64, copy=True, order="C")

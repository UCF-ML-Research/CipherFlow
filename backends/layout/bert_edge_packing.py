from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BertEdgePackingContract:
    tensor_shape: tuple[int, ...]
    layout_family: str
    layout_name: str
    tokens: int
    hidden_size: int | None = None
    num_heads: int | None = None
    packed_dim: int | None = None

    def as_meta(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tensor_shape": list(self.tensor_shape),
            "layout_family": self.layout_family,
            "layout_name": self.layout_name,
            "tokens": self.tokens,
        }
        if self.hidden_size is not None:
            payload["hidden_size"] = self.hidden_size
        if self.num_heads is not None:
            payload["num_heads"] = self.num_heads
        if self.packed_dim is not None:
            payload["packed_dim"] = self.packed_dim
        return payload


def supports_bert_edge_conversion_shape(shape: tuple[int, ...], *, max_tokens: int = 4096) -> bool:
    try:
        build_bert_edge_packing_contract(shape, max_tokens=max_tokens)
        return True
    except ValueError:
        return False


def build_bert_edge_packing_contract(shape: tuple[int, ...], *, max_tokens: int = 4096) -> BertEdgePackingContract:
    if len(shape) == 3:
        batch, seq_len, hidden = shape
        tokens = batch * seq_len
        if batch < 1 or seq_len < 1 or tokens > max_tokens:
            raise ValueError(
                f"BERT edge packing requires 1 <= B*S <= {max_tokens}, got B*S={tokens} for shape={list(shape)}"
            )
        if hidden in {768, 3072, 64}:
            family = "bert_hidden_state" if hidden == 768 else "bert_ffn_intermediate"
            name = "bert_hidden_row_major" if hidden == 768 else "bert_ffn_row_major"
            return BertEdgePackingContract(
                tensor_shape=shape,
                layout_family=family,
                layout_name=name,
                tokens=tokens,
                hidden_size=hidden,
            )
        raise ValueError(
            "Unsupported rank-3 BERT edge hidden size; expected one of [64, 768, 3072], "
            f"got H={hidden} for shape={list(shape)}"
        )

    if len(shape) == 4:
        if shape[0] == 3 and shape[-1] == 768:
            packed, batch, seq_len, hidden = shape
            tokens = batch * seq_len
            if batch < 1 or seq_len < 1 or tokens > max_tokens:
                raise ValueError(
                    f"Packed QKV conversion requires 1 <= B*S <= {max_tokens}, got B*S={tokens}"
                )
            return BertEdgePackingContract(
                tensor_shape=shape,
                layout_family="bert_packed_qkv",
                layout_name="bert_qkv_packed_major",
                tokens=tokens,
                hidden_size=hidden,
                packed_dim=packed,
            )
        batch, heads, seq_q, seq_k = shape
        tokens = batch * seq_q
        if batch < 1 or heads < 1 or seq_q < 1 or seq_k < 1:
            raise ValueError(f"Attention-score conversion expects positive dims, got shape={list(shape)}")
        if seq_q != seq_k:
            raise ValueError(
                "Attention-score conversion expects square score matrices, "
                f"got shape={list(shape)}"
            )
        if tokens > max_tokens:
            raise ValueError(
                f"Attention-score conversion requires 1 <= B*S <= {max_tokens}, got B*S={tokens}"
            )
        return BertEdgePackingContract(
            tensor_shape=shape,
            layout_family="bert_attention_scores",
            layout_name="bert_attention_score_major",
            tokens=tokens,
            num_heads=heads,
        )

    raise ValueError(
        "BERT edge conversion only supports rank-3 hidden/intermediate tensors, rank-4 attention scores, "
        f"or packed-qkv rank-4 tensors; got shape={list(shape)}"
    )


def prepare_he_tensor_for_mpc_bert_edge(x: np.ndarray, contract: BertEdgePackingContract) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if tuple(arr.shape) != contract.tensor_shape:
        raise ValueError(
            "BERT HE->MPC packing shape mismatch: "
            f"expected={list(contract.tensor_shape)} got={list(arr.shape)}"
        )
    return np.array(arr, dtype=np.float64, copy=True, order="C")


def prepare_mpc_tensor_for_he_bert_edge(x: np.ndarray, contract: BertEdgePackingContract) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if tuple(arr.shape) != contract.tensor_shape:
        raise ValueError(
            "BERT MPC->HE packing shape mismatch: "
            f"expected={list(contract.tensor_shape)} got={list(arr.shape)}"
        )
    return np.array(arr, dtype=np.float64, copy=True, order="C")

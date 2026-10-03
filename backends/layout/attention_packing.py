
from __future__ import annotations

from typing import Any


ATTENTION_LAYOUT_REQUIREMENTS: dict[str, Any] = {
    "qk_cross_packing": {
        "source": "SCI/tests/bert_bolt/linear.cpp::plain_cross_packing_postprocess",
        "notes": "Reorders HE-converted QK shares into MPC row layout for softmax input.",
    },
    "softmax_preprocess": {
        "source": "SCI/tests/bert_bolt/linear.cpp::preprocess_softmax",
        "notes": "Packs MPC softmax shares for ss_to_he before HE-side softmax*V.",
    },
    "v_cross_packing": {
        "source": "SCI/tests/bert_bolt/linear.cpp::plain_cross_packing_postprocess_v",
        "notes": "Reorders converted HE softmax*V outputs back into MPC/column layout.",
    },
}


def describe_attention_layout_requirements() -> dict[str, Any]:
    return dict(ATTENTION_LAYOUT_REQUIREMENTS)


def require_attention_layout_support() -> None:
    raise NotImplementedError(
        "Attention HE<->MPC layout support is not implemented yet. "
        "Required steps include QK cross packing, softmax preprocessing, and V packing."
    )

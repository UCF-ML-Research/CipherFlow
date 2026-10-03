from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class OperatorNode:
    node_id: str
    op_type: str
    input_shape: Shape
    output_shape: Shape
    attributes: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DataEdge:
    src: str
    dst: str
    tensor_shape: Shape


@dataclass(frozen=True)
class OperatorGraph:
    graph_id: str
    nodes: List[OperatorNode]
    edges: List[DataEdge]

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

from .types import DataEdge, OperatorGraph, OperatorNode


@dataclass(frozen=True)
class BertBlockConfig:
    batch_size: int = 1
    seq_len: int = 128
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    graph_id: str = "bert_block"


def _shape_bsh(cfg: BertBlockConfig) -> tuple[int, int, int]:
    return (cfg.batch_size, cfg.seq_len, cfg.hidden_size)


def _shape_scores(cfg: BertBlockConfig) -> tuple[int, int, int, int]:
    return (cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.seq_len)


def _shape_intermediate(cfg: BertBlockConfig) -> tuple[int, int, int]:
    return (cfg.batch_size, cfg.seq_len, cfg.intermediate_size)


def _topological_order(graph: OperatorGraph) -> List[str]:
    indegree: Dict[str, int] = {node.node_id: 0 for node in graph.nodes}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges:
        indegree[edge.dst] += 1
        outgoing[edge.src].append(edge.dst)

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    order: List[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for dst in outgoing[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return order


def _validate_graph(graph: OperatorGraph) -> None:
    node_ids = {node.node_id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes):
        raise ValueError("OperatorGraph contains duplicate node_id values")

    for edge in graph.edges:
        if edge.src not in node_ids or edge.dst not in node_ids:
            raise ValueError(f"Edge references unknown nodes: {edge.src} -> {edge.dst}")

    order = _topological_order(graph)
    if len(order) != len(graph.nodes):
        raise ValueError("BERT block graph must be a DAG")


def build_bert_block_graph(config: BertBlockConfig) -> OperatorGraph:
    if config.hidden_size != 768:
        raise ValueError(f"This minimal builder currently expects hidden_size=768, got {config.hidden_size}")
    if config.hidden_size % config.num_heads != 0:
        raise ValueError(
            f"hidden_size must be divisible by num_heads; hidden_size={config.hidden_size}, num_heads={config.num_heads}"
        )

    bsh = _shape_bsh(config)
    scores = _shape_scores(config)
    intermediate = _shape_intermediate(config)

    nodes = [
        OperatorNode(
            node_id="n1",
            op_type="LayerNorm",
            input_shape=bsh,
            output_shape=bsh,
            attributes={"stage": "attention_pre", "position": 1},
        ),
        OperatorNode(
            node_id="n2",
            op_type="Attention_QK_MatMul",
            input_shape=bsh,
            output_shape=scores,
            attributes={"num_heads": config.num_heads},
        ),
        OperatorNode(
            node_id="n3",
            op_type="Softmax",
            input_shape=scores,
            output_shape=scores,
            attributes={"axis": -1},
        ),
        OperatorNode(
            node_id="n4",
            op_type="Attention_V_MatMul",
            input_shape=scores,
            output_shape=bsh,
            attributes={"num_heads": config.num_heads},
        ),
        OperatorNode(
            node_id="n5",
            op_type="Residual_Add",
            input_shape=bsh,
            output_shape=bsh,
            attributes={"inputs": 2, "stage": "attention_residual"},
        ),
        OperatorNode(
            node_id="n6",
            op_type="LayerNorm",
            input_shape=bsh,
            output_shape=bsh,
            attributes={"stage": "ffn_pre", "position": 2},
        ),
        OperatorNode(
            node_id="n7",
            op_type="FFN_Linear_1",
            input_shape=bsh,
            output_shape=intermediate,
            attributes={"out_features": config.intermediate_size},
        ),
        OperatorNode(
            node_id="n8",
            op_type="GeLU",
            input_shape=intermediate,
            output_shape=intermediate,
            attributes={"approximate": True},
        ),
        OperatorNode(
            node_id="n9",
            op_type="FFN_Linear_2",
            input_shape=intermediate,
            output_shape=bsh,
            attributes={"out_features": config.hidden_size},
        ),
        OperatorNode(
            node_id="n10",
            op_type="Residual_Add",
            input_shape=bsh,
            output_shape=bsh,
            attributes={"inputs": 2, "stage": "ffn_residual"},
        ),
    ]

    edges = [
        DataEdge(src="n1", dst="n2", tensor_shape=bsh),
        DataEdge(src="n2", dst="n3", tensor_shape=scores),
        DataEdge(src="n3", dst="n4", tensor_shape=scores),
        DataEdge(src="n1", dst="n4", tensor_shape=bsh),
        DataEdge(src="n4", dst="n5", tensor_shape=bsh),
        DataEdge(src="n1", dst="n5", tensor_shape=bsh),
        DataEdge(src="n5", dst="n6", tensor_shape=bsh),
        DataEdge(src="n6", dst="n7", tensor_shape=bsh),
        DataEdge(src="n7", dst="n8", tensor_shape=intermediate),
        DataEdge(src="n8", dst="n9", tensor_shape=intermediate),
        DataEdge(src="n9", dst="n10", tensor_shape=bsh),
        DataEdge(src="n6", dst="n10", tensor_shape=bsh),
    ]

    graph = OperatorGraph(graph_id=config.graph_id, nodes=nodes, edges=edges)
    _validate_graph(graph)
    return graph

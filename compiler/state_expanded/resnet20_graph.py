"""ResNet-20 (CIFAR-10) operator DAG used by the compiler.

Produces an :class:`OperatorGraph` whose topology matches the standard
He et al. 2016 ResNet-20 reference (3 stages of 3 BasicBlocks each, with
stride-2 down-sampling at the start of stages 2 and 3):

    Conv0 (3 -> 16, 3x3, 32x32) -> ReLU0
       | stage 1 (16 ch, 32x32)
       block_1_1, block_1_2, block_1_3
       | stage 2 (32 ch, 16x16, stride 2 in first block)
       block_2_1, block_2_2, block_2_3
       | stage 3 (64 ch, 8x8, stride 2 in first block)
       block_3_1, block_3_2, block_3_3
       | head
       AvgPool 8x8 -> FC 64 -> 10

Each BasicBlock is materialised as the sequence

    fork = previous_block_out
        Conv_a -> ReLU_mid -> Conv_b -> Residual_Add(skip = fork) -> ReLU_out

so the residual add is the SESE-region join, the inner conv chain is
the main path, and the fork->add edge is the skip path. This mirrors
``bert_graph.bert_multi_block_graph``'s residual wiring so the same
SESE detector works without modification.

BatchNorm is intentionally folded into the preceding Conv (standard
inference-time fusion) — at deployment time BN parameters become Conv
bias/scale terms and need no separate operator. We therefore only
materialise Conv, ReLU, Residual_Add, AvgPool, FC.

This graph is intended for the placement-fair comparison in the paper
revision; numerical weights are not part of the IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ir.types import DataEdge, OperatorGraph, OperatorNode

Shape = Tuple[int, ...]


@dataclass(frozen=True)
class ResNet20ShapeManifest:
    """Concrete shapes fed to the profiler for ResNet-20."""

    batch: int = 1
    in_channels: int = 3
    in_spatial: int = 32
    stem_channels: int = 16
    num_classes: int = 10

    def as_dict(self) -> dict:
        return {
            "batch": self.batch,
            "in_channels": self.in_channels,
            "in_spatial": self.in_spatial,
            "stem_channels": self.stem_channels,
            "num_classes": self.num_classes,
        }


def default_manifest() -> ResNet20ShapeManifest:
    return ResNet20ShapeManifest()


def _shape(*dims: int) -> Shape:
    return tuple(int(d) for d in dims)


def resnet20_graph(manifest: ResNet20ShapeManifest | None = None) -> OperatorGraph:
    """Return the operator DAG for ResNet-20 on CIFAR-10."""

    m = manifest or default_manifest()
    B = m.batch
    nodes: List[OperatorNode] = []
    edges: List[DataEdge] = []

    # ---- stem ----
    stem_in = _shape(B, m.in_channels, m.in_spatial, m.in_spatial)
    stem_out = _shape(B, m.stem_channels, m.in_spatial, m.in_spatial)

    nodes.append(
        OperatorNode(
            "stem_conv", "Conv2D", stem_in, stem_out,
            attributes={"kernel": 3, "stride": 1, "padding": 1,
                        "in_ch": m.in_channels, "out_ch": m.stem_channels},
        )
    )
    nodes.append(
        OperatorNode("stem_relu", "ReLU", stem_out, stem_out)
    )
    edges.append(DataEdge("stem_conv", "stem_relu", stem_out))

    # ---- stages ----
    # (channels, spatial). First block of each stage after the first
    # uses stride 2 -> spatial halves and channels double.
    stages: List[Tuple[int, int]] = [
        (16, 32),
        (32, 16),
        (64, 8),
    ]

    prev_out_id = "stem_relu"
    prev_out_shape = stem_out

    for s_idx, (ch, sp) in enumerate(stages):
        for b_idx in range(3):
            # Down-sample at the first block of stages 2 and 3.
            stride2 = (s_idx > 0 and b_idx == 0)

            block_id = f"s{s_idx + 1}_b{b_idx + 1}"
            in_shape = prev_out_shape
            mid_shape = _shape(B, ch, sp, sp)   # after Conv_a
            out_shape = _shape(B, ch, sp, sp)   # after Conv_b / Add / ReLU

            conv_a_id = f"{block_id}_conv_a"
            relu_mid_id = f"{block_id}_relu_mid"
            conv_b_id = f"{block_id}_conv_b"
            add_id = f"{block_id}_add"
            relu_out_id = f"{block_id}_relu_out"

            # Conv_a: optionally stride-2 (down-samples spatial; doubles ch)
            nodes.append(
                OperatorNode(
                    conv_a_id, "Conv2D", in_shape, mid_shape,
                    attributes={"kernel": 3, "stride": 2 if stride2 else 1,
                                "padding": 1, "in_ch": in_shape[1], "out_ch": ch},
                )
            )
            nodes.append(
                OperatorNode(relu_mid_id, "ReLU", mid_shape, mid_shape)
            )
            nodes.append(
                OperatorNode(
                    conv_b_id, "Conv2D", mid_shape, out_shape,
                    attributes={"kernel": 3, "stride": 1, "padding": 1,
                                "in_ch": ch, "out_ch": ch},
                )
            )
            nodes.append(
                OperatorNode(add_id, "Residual_Add", out_shape, out_shape)
            )
            nodes.append(
                OperatorNode(relu_out_id, "ReLU", out_shape, out_shape)
            )

            # main-path edges
            edges.append(DataEdge(prev_out_id, conv_a_id, in_shape))
            edges.append(DataEdge(conv_a_id, relu_mid_id, mid_shape))
            edges.append(DataEdge(relu_mid_id, conv_b_id, mid_shape))
            edges.append(DataEdge(conv_b_id, add_id, out_shape))
            # skip-path: fork = prev_out_id, join = add_id.
            # When stride2 (or channel-mismatch) the textbook ResNet
            # uses a 1x1 projection on the skip path. We model that as
            # a Conv2D node so the planner sees the extra cost.
            if stride2 or (in_shape[1] != ch) or (in_shape[2] != sp):
                proj_id = f"{block_id}_skip_proj"
                nodes.append(
                    OperatorNode(
                        proj_id, "Conv2D", in_shape, out_shape,
                        attributes={"kernel": 1, "stride": 2 if stride2 else 1,
                                    "padding": 0, "in_ch": in_shape[1],
                                    "out_ch": ch, "is_skip_proj": True},
                    )
                )
                edges.append(DataEdge(prev_out_id, proj_id, in_shape))
                edges.append(DataEdge(proj_id, add_id, out_shape))
            else:
                # identity skip: fork directly to join with same shape
                edges.append(DataEdge(prev_out_id, add_id, in_shape))

            edges.append(DataEdge(add_id, relu_out_id, out_shape))

            prev_out_id = relu_out_id
            prev_out_shape = out_shape

    # ---- head ----
    pool_in = prev_out_shape  # (B, 64, 8, 8)
    pool_out = _shape(B, 64, 1, 1)
    nodes.append(
        OperatorNode("avg_pool", "AvgPool", pool_in, pool_out,
                     attributes={"kernel": 8, "stride": 1})
    )
    fc_in = pool_out
    fc_out = _shape(B, m.num_classes)
    nodes.append(
        OperatorNode("fc", "FC", fc_in, fc_out,
                     attributes={"in": 64, "out": m.num_classes})
    )
    edges.append(DataEdge(prev_out_id, "avg_pool", pool_in))
    edges.append(DataEdge("avg_pool", "fc", pool_out))

    return OperatorGraph(
        graph_id=f"resnet20_B{B}",
        nodes=nodes,
        edges=edges,
    )


def enumerate_profile_shapes(
    manifest: ResNet20ShapeManifest | None = None,
) -> List[Tuple[str, Shape, Shape]]:
    """Unique (op_type, input_shape, output_shape) tuples."""

    g = resnet20_graph(manifest)
    seen: dict = {}
    for node in g.nodes:
        key = (node.op_type, node.input_shape, node.output_shape)
        seen.setdefault(key, key)
    return list(seen.values())


def enumerate_edge_shapes(
    manifest: ResNet20ShapeManifest | None = None,
) -> List[Shape]:
    """Unique tensor shapes that cross edges (used for conversion profiling)."""
    g = resnet20_graph(manifest)
    return sorted({edge.tensor_shape for edge in g.edges})

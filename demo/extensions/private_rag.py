"""A new workload through an extension: the retrieval step of private RAG.

The client's question is encoded by a BERT encoder, pooled into one query vector and scored
against the server's private document embeddings; the client learns the indices of the k best
documents and nothing else. Pooling, scoring and top-k are not Transformer operators, so an
extension brings them: the PyTorch modules below become its operators (MeanPool, DocScores,
TopK), with an implementation in HE and/or MPC each and a cost.

    python -m demo.extensions.private_rag
    python -m demo.extensions.private_rag --docs 100000 --k 5 --network 1Gbps/20ms

The costs are estimates scaled from the profile's measured operators (see `rag_extension`);
replace them with measurements of your own kernels and protocols.
"""
from __future__ import annotations

import argparse
import warnings

import torch
from torch import nn

import cipherflow as cf
from cipherflow.extensions import like, numel
from demo import transformer


class MeanPool(nn.Module):                     # [B, S, H] -> [B, H]
    def forward(self, x):
        return x.mean(dim=1)


class DocScores(nn.Module):                    # [B, H] -> [B, N]: the server's document embeddings x the query
    def __init__(self, docs):
        super().__init__()
        self.docs = nn.Parameter(docs, requires_grad=False)

    def forward(self, q):
        return q @ self.docs.T


class TopK(nn.Module):                         # [B, N] -> [B, k]: indices of the best documents
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, scores):
        return torch.topk(scores, self.k, dim=-1).indices


class Retriever(nn.Module):
    def __init__(self, n_docs: int, k: int, blocks: int = 12):
        super().__init__()
        self.encoder = transformer.BertEncoder(blocks)
        self.pool = MeanPool()
        self.scores = DocScores(torch.randn(n_docs, 768))
        self.topk = TopK(k)

    def forward(self, x):
        return self.topk(self.scores(self.pool(self.encoder(x))))


def _macs(node) -> int:
    return numel(node.input_shape) // node.input_shape[-1] * node.input_shape[-1] * node.output_shape[-1]


def rag_extension() -> cf.Extension:
    """The retrieval operators and their implementations. Costs are scaled from the profile's
    measured records (cf.extensions.like); each scaling is a stated estimate."""
    ext = cf.Extension("private-rag")
    ext.operator(MeanPool, "MeanPool").operator(DocScores, "DocScores").operator(TopK, "TopK")
    # mean over tokens: in HE log2(tokens) rotations and additions per ciphertext, priced as 8 additions;
    # on secret shares it is local
    ext.primitive("MeanPool", "HE", levels=1,
                  cost=like("Residual_Add", "HE", lambda n, r: 8 * numel(n.input_shape) / numel(r.input_shape)))
    ext.primitive("MeanPool", "MPC", cost=cf.Cost(local_ms=1.0))
    # plaintext document matrix x query: a linear layer, priced from the measured FFN_Linear_1 by multiply-accumulates
    by_macs = lambda n, r: _macs(n) / _macs(r)              # noqa: E731
    ext.primitive("DocScores", "HE", levels=1, cost=like("FFN_Linear_1", "HE", by_macs))
    ext.primitive("DocScores", "MPC", cost=like("FFN_Linear_1", "MPC", by_macs))
    # top-k needs comparisons, which CKKS does not have: MPC only, priced as k passes of the row
    # maximum inside the measured MPC Softmax (an upper bound: that Softmax also computes exp and 1/x)
    ext.primitive("TopK", "MPC",
                  cost=like("Softmax", "MPC", lambda n, r: n.output_shape[-1] * numel(n.input_shape) / numel(r.input_shape)))
    return ext


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", type=int, default=8192, help="documents in the server's index")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=12, help="encoder blocks")
    ap.add_argument("--network", default="100Mbps/20ms")
    ap.add_argument("--hardware", default="cpu")
    a = ap.parse_args(argv)

    rag = rag_extension()
    model = Retriever(a.docs, a.k, a.blocks).eval()
    graph = cf.models.from_torch(model, torch.zeros(1, 128, 768), extensions=[rag])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cf.CostExtrapolationWarning)
        plan = cf.compile(graph, cf.Deployment.of(a.hardware, a.network), extensions=[rag])

    print(plan.explain())
    print("  retrieval, operator by operator (the encoder before it is in plan.explain(steps=True)):")
    rows = plan.explain_steps().splitlines()
    tail = next(i for i, ln in enumerate(rows) if "pool" in ln)
    start = max(1, tail - 3)
    print("\n".join(rows[:1] + rows[start:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

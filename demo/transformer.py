"""The demo's input: Transformer encoders written in plain PyTorch.

CipherFlow traces them (`cf.models.from_torch`), plans every operator in HE or MPC for a
deployment, and reports the plan operator by operator, with where each operator is in this code:

    python -m demo.plan_pytorch_model --model bert --network 10Mbps/1ms
    cipherflow compile --torch demo/transformer.py:bert --network 10Mbps/1ms --steps
    cipherflow compile --torch demo/transformer.py:vit --input-shape 1,197,768 --steps

The forward of each encoder takes hidden states [batch, tokens, 768]: the embeddings (BERT) or the
flattened 16x16x3 image patches (ViT) are computed by the client before the encoder.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, hidden: int = 768, heads: int = 12):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)

    def forward(self, x):                                       # x: [B, S, H]
        B, S, H = x.shape
        d = H // self.heads

        def split(t):                                           # [B, S, H] -> [B, heads, S, d]
            return t.view(B, S, self.heads, d).transpose(1, 2)
        scores = split(self.q(x)) @ split(self.k(x)).transpose(-1, -2) / math.sqrt(d)
        probs = torch.softmax(scores, dim=-1)
        out = (probs @ split(self.v(x))).transpose(1, 2).reshape(B, S, H)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, hidden: int = 768, ffn: int = 3072):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PostLNBlock(nn.Module):                                   # BERT
    def __init__(self, hidden: int = 768, heads: int = 12, ffn: int = 3072):
        super().__init__()
        self.attn, self.ln1 = SelfAttention(hidden, heads), nn.LayerNorm(hidden)
        self.mlp, self.ln2 = MLP(hidden, ffn), nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.mlp(x))


class PreLNBlock(nn.Module):                                    # ViT
    def __init__(self, hidden: int = 768, heads: int = 12, ffn: int = 3072):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(hidden), SelfAttention(hidden, heads)
        self.ln2, self.mlp = nn.LayerNorm(hidden), MLP(hidden, ffn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class BertEncoder(nn.Module):
    """BERT-base encoder: post-LN blocks over hidden states [B, S, 768]."""

    def __init__(self, num_blocks: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList(PostLNBlock() for _ in range(num_blocks))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViTEncoder(nn.Module):
    """ViT-B/16: patch projection, pre-LN blocks, final LayerNorm over patches [B, 197, 768]."""

    def __init__(self, num_blocks: int = 12):
        super().__init__()
        self.patch = nn.Linear(768, 768)
        self.blocks = nn.ModuleList(PreLNBlock() for _ in range(num_blocks))
        self.norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.patch(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def bert(num_blocks: int = 12) -> BertEncoder:
    return BertEncoder(num_blocks).eval()


def vit(num_blocks: int = 12) -> ViTEncoder:
    return ViTEncoder(num_blocks).eval()

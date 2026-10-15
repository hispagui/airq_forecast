"""
Small Transformer forecaster.

A Transformer uses ATTENTION: every time step can look directly at every other time step
and decide how much each one matters ("the same hour yesterday is very relevant, 3 a.m.
two days ago much less"). An LSTM has to carry that information step by step through its
memory; attention reaches it in one jump.

Design (per station, weights shared by all stations, like the other models):
  1. Patching: the 72 hours are cut into chunks of `patch` hours (default 3 -> 24 tokens).
     Each chunk becomes one token. This is the idea of PatchTST (Nie et al., 2023): fewer,
     richer tokens make attention cheaper (attention cost grows with tokens^2) and less noisy.
  2. A learned position embedding tells the model WHERE each chunk is in the window
     (attention by itself ignores order).
  3. A learned [SUMMARY] token is added; after the Transformer layers it has gathered what
     matters from all chunks, and it is what the ForecastHead receives.

Extra settings (train.py --model-args): patch=3 heads=4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_model
from .common import ForecastHead


@register_model("transformer")
class TransformerForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1,
                 residual: bool = True, patch: int = 3, heads: int = 4, **_):
        super().__init__()
        patch, heads = int(patch), int(heads)
        if hidden % heads:
            raise ValueError(f"--hidden ({hidden}) must be divisible by heads ({heads}).")
        self.patch = patch
        self.pad = (-history) % patch                        # left-pad so chunks fit exactly
        n_tokens = (history + self.pad) // patch

        self.token_proj = nn.Linear(patch * n_features, hidden)
        self.summary = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, hidden))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.summary, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=2 * hidden, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)     # one sequence per station
        if self.pad:
            seq = torch.cat([seq.new_zeros(B * N, self.pad, F), seq], dim=1)
        tokens = self.token_proj(seq.reshape(B * N, -1, self.patch * F))   # (B*N, n_tok, D)
        tokens = torch.cat([self.summary.expand(B * N, -1, -1), tokens], dim=1) + self.pos
        h = self.encoder(tokens)
        enc = self.norm(h[:, 0]).view(B, N, -1)              # the [SUMMARY] token
        return self.head(enc, x, x_future)

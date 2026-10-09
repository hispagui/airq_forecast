"""
MLP baseline: each station's 72 h history is flattened into one vector and passed through
a small feed-forward network. Weights are shared by all stations (a "global" model), so it
learns from 56 stations at once. It has no notion of time order: a useful lower bound
for what the recurrent model adds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_model
from .common import ForecastHead


@register_model("mlp")
class MLPForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 128, dropout: float = 0.1, residual: bool = True, **_):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(history * n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=hidden, dropout=dropout, residual=residual)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(B, N, L * F)   # one vector per station
        return self.head(self.encoder(flat), x, x_future)

"""
GRU forecaster: the LSTM's lighter cousin.

A GRU (Gated Recurrent Unit) also reads each station's history hour by hour and keeps a
memory vector, but with a simpler update rule (2 gates instead of 3, no separate "cell").
It has ~25 % fewer parameters than an LSTM of the same size, trains faster, and is often
just as accurate: a standard comparison in forecasting papers.

Same structure as lstm.py: one shared GRU for all stations, then the shared ForecastHead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_model
from .common import ForecastHead


@register_model("gru")
class GRUForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1,
                 residual: bool = True, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)     # one sequence per station
        _, h = self.gru(self.input_proj(seq))                # h: (layers, B*N, hidden)
        enc = h[-1].view(B, N, -1)                           # last layer's final memory
        return self.head(enc, x, x_future)

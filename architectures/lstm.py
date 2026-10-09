"""
LSTM forecaster (the main Phase-1 model).

Each station's history is read hour by hour by an LSTM, a recurrent network that keeps a
memory vector updated at every step. Its final memory summarizes the last 72 h (level,
trend, daily pattern), and the shared ForecastHead turns it into 24 hourly predictions.

The same LSTM weights are used for all 56 stations: the batch of B samples x N stations is
reshaped into B*N independent sequences. Stations do NOT exchange information here; that
is exactly what the graph neural network of Phase 2 will add.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_model
from .common import ForecastHead


@register_model("lstm")
class LSTMForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1,
                 residual: bool = True, **_):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)     # one sequence per station
        _, (h, _) = self.lstm(self.input_proj(seq))
        enc = h[-1].view(B, N, -1)                           # last layer's final memory
        return self.head(enc, x, x_future)

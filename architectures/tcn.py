"""
Temporal Convolutional Network (TCN).

Instead of reading the hours one by one like an LSTM, a TCN slides small filters along
the time axis. Each filter looks at `kernel` hours at once (default 3), and stacking
layers with growing gaps between the hours they look at ("dilations" 1, 2, 4, 8, ...)
lets the last layer see the whole 72 h history:

    level 1: hours t, t-1, t-2          level 2: t, t-2, t-4         level 3: t, t-4, t-8 ...

The convolutions are CAUSAL: the output at hour t only uses hours <= t. All hours are
processed in parallel, so a TCN trains much faster than an LSTM, especially on a GPU / M1.

Each level is a residual block: two dilated convolutions, then "+ input", which keeps
training stable when stacking many levels (Bai, Kolter & Koltun, 2018).

The number of levels is chosen automatically so that the receptive field covers the whole
history (the --layers option of train.py is not used here).
Extra settings (train.py --model-args): kernel=3
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_

from . import register_model
from .common import ForecastHead


class CausalConv1d(nn.Conv1d):
    """Conv1d that pads on the left only, so the output at t never sees t+1, t+2, ..."""

    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__(channels, channels, kernel, dilation=dilation)
        self.left_pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, channels, time)
        return super().forward(F_.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
            CausalConv1d(channels, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
        )
        # LayerNorm over the channels of EACH hour separately (a norm that averages over
        # time, like BatchNorm/GroupNorm, would let later hours influence earlier outputs)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (batch, channels, time)
        return self.norm((x + self.net(x)).transpose(1, 2)).transpose(1, 2)


def levels_needed(history: int, kernel: int) -> int:
    """Smallest number of levels whose receptive field 1 + 2(k-1)(2^n - 1) covers history."""
    n = 1
    while 1 + 2 * (kernel - 1) * (2 ** n - 1) < history:
        n += 1
    return n


@register_model("tcn")
class TCNForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, dropout: float = 0.1, residual: bool = True,
                 kernel: int = 3, **_):
        super().__init__()
        kernel = int(kernel)
        self.n_levels = levels_needed(history, kernel)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** self.n_levels - 1)
        self.input_proj = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(*[TemporalBlock(hidden, kernel, 2 ** i, dropout)
                                      for i in range(self.n_levels)])
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * N, L, F)     # one sequence per station
        h = self.input_proj(seq).transpose(1, 2)             # (B*N, hidden, L)
        h = self.blocks(h)
        enc = h[:, :, -1].reshape(B, N, -1)                  # last hour sees the whole history
        return self.head(enc, x, x_future)

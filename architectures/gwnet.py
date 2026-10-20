"""
Graph WaveNet (Wu, Pan, Long, Jiang & Zhang, IJCAI 2019), adapted to this project.
Reference spatio-temporal graph model: arXiv:1906.00121.

Each layer does TIME then SPACE:

    1. Gated temporal convolution (the "WaveNet" part):
           h = tanh(conv_filter(x)) * sigmoid(conv_gate(x))
       Causal convolutions with kernel 2 and growing dilations. The dilations are chosen so
       that the receptive field covers exactly the 72 h of history (1, 2, 4, 8, 16, 32, 8):
       after the last layer only one time step is left, which has seen all 72 hours.
    2. Diffusion graph convolution (the "Graph" part), K = 2 steps:
           Z = sum over supports P and steps k = 0..K of  P^k h W_{P,k}
       with three supports:
           P_fwd = row-normalized A        information flows along the edges
           P_bwd = row-normalized A^T      ... and against them
           A_adp = softmax(ReLU(E1 E2^T))  a graph LEARNED from data: each station has two
                                           small learned vectors (source / target); stations
                                           whose vectors match become connected. It can find
                                           links that distance alone does not suggest.
    3. Residual connection + batch normalization, and a skip connection from every layer
       to the output.

The sum of the skip connections gives one vector per station, which goes to the same
ForecastHead as every other model in the project (residual on the last observed value,
future weather and calendar), so the comparison with the other models stays fair.
(In the paper the output layer predicts the horizon directly from the skip connections.)

Settings  (train.py --model gwnet --model-args KEY=VALUE ...)
    graph=distance | knn | none     fixed graph for P_fwd / P_bwd (the paper uses a
                                    thresholded Gaussian kernel on distances = "distance");
                                    none = adaptive graph only (paper, Eq. 7)
    adaptive=1 | 0                  use the learned graph
    channels=32  skip=64  emb_dim=10  order=2  k=8  sigma=<km>  threshold=0.1
    --hidden: size of the station vector given to the head (default 64); --dropout after
    each graph convolution (the paper uses 0.3). --layers is not used.

Note for the spatial holdout: the learned graph gives every station its own embedding, so
an unseen station's row of A_adp is never trained. Use adaptive=0 there, or report it as a
known limitation of Graph WaveNet (the paper studies a fixed set of sensors).
In eval mode the learned graph of the last batch is kept in `model.last_adaptive`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_

from . import register_model
from .common import ForecastHead
from .gnn import import_project


def dilation_schedule(history: int) -> list[int]:
    """Kernel-2 dilations doubling until the receptive field 1 + sum(d) reaches `history`."""
    ds, total, d = [], 0, 1
    while total + d <= history - 1:
        ds.append(d)
        total += d
        d *= 2
    if total < history - 1:
        ds.append(history - 1 - total)
    return ds


class DilatedConv2(nn.Module):
    """Causal kernel-2 dilated convolution over time, written as  W0 x(t-d) + W1 x(t).
    Mathematically identical to nn.Conv2d(c, c, (1, 2), dilation=(1, d)) without padding,
    but built from a plain 1x1 convolution, which works reliably on every device (dilated
    convolutions have produced NaNs on Apple's MPS backend) and is fast everywhere.
    Output length = input length - d."""

    def __init__(self, c_in: int, c_out: int, dilation: int):
        super().__init__()
        self.d = dilation
        self.conv = nn.Conv2d(2 * c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, C, N, T)
        return self.conv(torch.cat([x[..., :-self.d], x[..., self.d:]], dim=1))


class DiffusionConv(nn.Module):
    """Z = sum_{P, k} P^k X W, implemented by concatenating every P^k X then a 1x1 conv."""

    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int, dropout: float):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(c_in * (1 + n_supports * order), c_out, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        # x: (B, C, N, T); supports: (N, N) matrices, rows = receivers
        out = [x]
        for P in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("ij,bcjt->bcit", P, h)
                out.append(h)
        return self.drop(self.mlp(torch.cat(out, dim=1)))


@register_model("gwnet")
class GraphWaveNet(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, dropout: float = 0.1, residual: bool = True,
                 graph: str = "distance", adaptive: int = 1, channels: int = 32,
                 skip: int = 64, emb_dim: int = 10, order: int = 2, k: int = 8,
                 sigma: float | None = None, threshold: float = 0.1,
                 processed: str | None = None, **_):
        super().__init__()
        G = import_project("graph")
        channels, skip, emb_dim, order = int(channels), int(skip), int(emb_dim), int(order)
        self.use_adaptive = bool(int(adaptive))
        if graph not in ("distance", "knn", "none"):
            raise ValueError("gwnet: graph must be distance, knn or none")

        # ---- fixed graph: forward and backward transition matrices -------------------
        st = G.load_coordinates(processed or "data/processed")
        if len(st) != n_nodes:
            raise ValueError(f"{len(st)} station coordinates for {n_nodes} stations.")
        dist, _ = G.geometry(G.project_km(st["lat"].to_numpy(), st["lon"].to_numpy()))
        self.has_fixed = graph != "none"
        if self.has_fixed:
            A = (G.distance_graph(dist, sigma, float(threshold)) if graph == "distance"
                 else G.knn_graph(dist, int(k)))
            self.register_buffer("P_fwd", torch.tensor(G.row_normalize(A), dtype=torch.float32))
            self.register_buffer("P_bwd", torch.tensor(G.row_normalize(A.T), dtype=torch.float32))
        # ---- learned (adaptive) graph ------------------------------------------------
        if self.use_adaptive:
            self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        n_supports = 2 * self.has_fixed + self.use_adaptive

        # ---- temporal + spatial layers -------------------------------------------------
        self.dilations = dilation_schedule(history)
        self.receptive_field = 1 + sum(self.dilations)
        self.start = nn.Conv2d(n_features, channels, kernel_size=1)
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.skip_convs, self.gconvs, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for d in self.dilations:
            self.filter_convs.append(DilatedConv2(channels, channels, d))
            self.gate_convs.append(DilatedConv2(channels, channels, d))
            self.skip_convs.append(nn.Conv2d(channels, skip, kernel_size=1))
            self.gconvs.append(DiffusionConv(channels, channels, n_supports, order, dropout))
            self.norms.append(nn.BatchNorm2d(channels))
        self.end = nn.Conv2d(skip, hidden, kernel_size=1)
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)
        self.last_adaptive = None

    def supports(self) -> list[torch.Tensor]:
        s = [self.P_fwd, self.P_bwd] if self.has_fixed else []
        if self.use_adaptive:
            s.append(torch.softmax(F_.relu(self.E1 @ self.E2.T), dim=1))
        return s

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        h = x.permute(0, 3, 2, 1)                                  # (B, F, N, L)
        if L < self.receptive_field:
            h = F_.pad(h, (self.receptive_field - L, 0))
        h = self.start(h)                                          # (B, C, N, T)
        sup = self.supports()
        skip = 0
        for filt, gate, skip_conv, gconv, norm in zip(self.filter_convs, self.gate_convs,
                                                      self.skip_convs, self.gconvs, self.norms):
            res = h
            h = torch.tanh(filt(h)) * torch.sigmoid(gate(h))       # time shrinks by d
            skip = skip + skip_conv(h[..., -1:])                   # last step feeds the output
            h = gconv(h, sup) + res[..., -h.shape[-1]:]
            h = norm(h)
        enc = F_.relu(self.end(F_.relu(skip)))                     # (B, hidden, N, 1)
        enc = enc[..., 0].transpose(1, 2)                          # (B, N, hidden)
        if not self.training and self.use_adaptive:
            self.last_adaptive = sup[-1].detach()
        return self.head(enc, x, x_future)
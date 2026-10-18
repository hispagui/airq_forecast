"""
Graph neural network forecaster (Phase 2): "encode, then propagate".

    1. ENCODE   each station's 72 h history is read by a GRU (or a TCN), exactly like the
                Phase-1 models, giving one vector h_i per station.
    2. PROPAGATE  a few graph layers let every station mix in its neighbours' vectors:
                    h_i  <-  LayerNorm( h_i + GELU( W_self h_i + W_nb * sum_j A[i,j] h_j ) )
                After 2 layers a station has heard from its neighbours' neighbours.
    3. FORECAST the same ForecastHead as Phase 1 turns each h_i into 24 hourly values.

So the only difference with the Phase-1 GRU is step 2: information flows between stations
along the edges of a graph (see graph.py). Same inputs, same outputs, same training loop.

Graphs  (--model-args graph=...)
--------------------------------
    knn (default) | distance | random | none      fixed graphs built once from coordinates
    wind                                          a new directed graph for every forecast:
        for each candidate edge j -> i (the k nearest stations of i), a small network looks at
            * the distance between j and i
            * the wind at j projected on the direction j -> i ("is j upwind of i?")
            * the crosswind component and the wind speed
        for the past 6 h and for the 24 forecast hours, and outputs a weight. The weights of
        each station's neighbours are normalized to sum to 1 (softmax), like attention.
        These quantities do not change if the whole map and the wind are rotated together:
        the transport rule is rotation-invariant (see the discussion on symmetries).
        Needs weather in the inputs (wind_u, wind_v).

Settings  (train.py --model gnn --model-args KEY=VALUE ...)
    graph=knn  k=8  encoder=gru|tcn  gnn_layers=2  sigma=<km>  threshold=0.1
    --hidden, --layers (GRU layers), --dropout as for the other models.

For analysis, in eval mode the model keeps the adjacency of its last batch in
`model.last_adjacency`: (N, N) for fixed graphs, (B, N, N) for the wind graph.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F_

from . import register_model
from .common import ForecastHead
from .tcn import TemporalBlock, levels_needed

WIND_HISTORY_HOURS = 6     # wind of the last 6 h describes the current transport

def infer_feature_names(processed: str | None, n_features: int) -> list[str]:
    """Input feature names rebuilt from data/processed, when the caller did not pass them.
    x features are always [pm25, observed, *weather, *calendar] (see dataset.window_inputs)."""
    import json
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from features.dataset import find_processed_dir
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from features.dataset import find_processed_dir
    d = find_processed_dir(processed or "data/processed")
    wm = d / "weather_meta.json"
    weather = json.loads(wm.read_text())["features"] if wm.exists() else []
    with np.load(d / "dataset.npz") as z:
        n_cal = z["time_features"].shape[1]
    calendar = [f"calendar_{i}" for i in range(n_cal)]
    if n_features == 2 + len(weather) + n_cal:
        return ["pm25", "observed"] + weather + calendar
    if n_features == 2 + n_cal:                      # trained with --no-weather
        return ["pm25", "observed"] + calendar
    raise ValueError(f"Cannot match {n_features} input features to the data in {d}.")

class GraphLayer(nn.Module):
    """One round of message passing on a dense graph (55 stations: a 55x55 matrix product)."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.self_lin = nn.Linear(d, d)
        self.nb_lin = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: (B, N, D); A: (N, N) shared by the batch, or (B, N, N) one graph per sample
        msg = torch.einsum("ij,bjd->bid", A, h) if A.dim() == 2 else torch.einsum("bij,bjd->bid", A, h)
        return self.norm(h + self.drop(F_.gelu(self.self_lin(h) + self.nb_lin(msg))))


class WindAdjacency(nn.Module):
    """Directed, wind-dependent adjacency built from rotation-invariant edge features."""

    def __init__(self, unit: torch.Tensor, dist_scaled: torch.Tensor, mask: torch.Tensor,
                 hidden: int = 32):
        super().__init__()
        self.register_buffer("unit", unit)            # (N, N, 2) unit[i, j]: from j to i
        self.register_buffer("dist", dist_scaled)     # (N, N)
        self.register_buffer("mask", mask)            # (N, N) bool candidate edges
        self.mlp = nn.Sequential(nn.Linear(7, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def edge_features(self, wind: torch.Tensor) -> list[torch.Tensor]:
        """wind: (B, N, 2) at every station -> along, cross, speed for each edge j -> i."""
        N = wind.shape[1]
        w = wind[:, None, :, :]                                   # [b, i, j] = wind at j
        along = (w * self.unit).sum(-1)                           # > 0: wind blows j -> i
        cross = w[..., 0] * self.unit[..., 1] - w[..., 1] * self.unit[..., 0]
        speed = w.norm(dim=-1).expand(-1, N, -1)
        return [along, cross, speed]

    def forward(self, wind_hist: torch.Tensor, wind_fut: torch.Tensor) -> torch.Tensor:
        B = wind_hist.shape[0]
        feats = [self.dist.expand(B, -1, -1)] + self.edge_features(wind_hist) \
            + self.edge_features(wind_fut)
        logits = self.mlp(torch.stack(feats, dim=-1)).squeeze(-1)  # (B, N, N)
        logits = logits.masked_fill(~self.mask, float("-inf"))
        return torch.softmax(logits, dim=-1).nan_to_num(0.0)       # rows sum to 1


@register_model("gnn")
class GNNForecaster(nn.Module):
    def __init__(self, n_nodes, n_features, n_time_features, history, horizon, out_size=1,
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1, residual: bool = True,
                 graph: str = "knn", k: int = 8, encoder: str = "gru", gnn_layers: int = 2,
                 sigma: float | None = None, threshold: float = 0.1,
                 processed: str | None = None, feature_names: list[str] | None = None, **_):
        super().__init__()

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import features.graph as G # graph.py 
        
        k, gnn_layers = int(k), int(gnn_layers)
        self.graph_name, self.encoder_name = graph, encoder
        if graph not in G.ALL_GRAPHS:
            raise ValueError(f"graph={graph}: choose from {G.ALL_GRAPHS}")

        # ---- geometry from the station coordinates --------------------------------
        st = G.load_coordinates(processed or "data/processed")
        if len(st) != n_nodes:
            raise ValueError(f"{len(st)} station coordinates for {n_nodes} stations.")
        pos = G.project_km(st["lat"].to_numpy(), st["lon"].to_numpy())
        dist, unit = G.geometry(pos)
        self.register_buffer("pos_km", torch.tensor(pos, dtype=torch.float32))

        if graph == "wind":
            names = feature_names or infer_feature_names(processed, n_features)
            if "wind_u" not in names or "wind_v" not in names:
                raise ValueError("graph=wind needs wind_u and wind_v in the inputs "
                                 "(train with weather, without --no-weather).")
            self.iu, self.iv = names.index("wind_u"), names.index("wind_v")
            # x_future = [weather..., calendar...] = x features without [pm25, observed]
            self.fu, self.fv = self.iu - 2, self.iv - 2
            off = ~np.eye(n_nodes, dtype=bool)
            scale = float(np.median(dist[off]))
            self.wind_adj = WindAdjacency(
                torch.tensor(unit, dtype=torch.float32),
                torch.tensor(dist / scale, dtype=torch.float32),
                torch.tensor(G.knn_mask(dist, k)))
        else:
            A = G.build_graph(graph, dist, k=k, sigma=sigma, threshold=float(threshold))
            self.register_buffer("adj", torch.tensor(A, dtype=torch.float32))

        # ---- 1. encoder (one shared network for all stations) -----------------------
        self.input_proj = nn.Linear(n_features, hidden)
        if encoder == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
        elif encoder == "tcn":
            self.tcn = nn.Sequential(*[TemporalBlock(hidden, 3, 2 ** i, dropout)
                                       for i in range(levels_needed(history, 3))])
        else:
            raise ValueError("encoder must be gru or tcn")

        # ---- 2. graph layers, 3. forecast head --------------------------------------
        self.graph_layers = nn.ModuleList([GraphLayer(hidden, dropout) for _ in range(gnn_layers)])
        self.head = ForecastHead(hidden, n_nodes, n_time_features, horizon, out_size,
                                 hidden=2 * hidden, dropout=dropout, residual=residual)
        self.last_adjacency = None

    # ------------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, F = x.shape
        seq = self.input_proj(x.permute(0, 2, 1, 3).reshape(B * N, L, F))
        if self.encoder_name == "gru":
            _, h = self.rnn(seq)
            return h[-1].view(B, N, -1)
        return self.tcn(seq.transpose(1, 2))[:, :, -1].reshape(B, N, -1)

    def adjacency(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        if self.graph_name != "wind":
            return self.adj
        wind_hist = x[:, -WIND_HISTORY_HOURS:, :, [self.iu, self.iv]].mean(dim=1)   # (B, N, 2)
        wind_fut = x_future[:, :, :, [self.fu, self.fv]].mean(dim=1)                # (B, N, 2)
        return self.wind_adj(wind_hist, wind_fut)

    def forward(self, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)                                 # (B, N, hidden)
        A = self.adjacency(x, x_future)
        for layer in self.graph_layers:
            h = layer(h, A)
        if not self.training:
            self.last_adjacency = A.detach()
        return self.head(h, x, x_future)

"""Building blocks shared by the Phase-1 models."""

from __future__ import annotations

import torch
import torch.nn as nn


def last_observed(x: torch.Tensor) -> torch.Tensor:
    """Most recent observed normalized PM2.5 per (batch, station). x: (B, L, N, F) -> (B, N).
    Stations never observed in the window get 0 (= the training average)."""
    vals, ok = x[..., 0], x[..., 1] > 0.5
    L = vals.shape[1]
    pos = torch.arange(L, device=x.device).view(1, L, 1)
    idx = torch.where(ok, pos, torch.full_like(pos, -1)).amax(dim=1)          # (B, N)
    last = vals.gather(1, idx.clamp(min=0).unsqueeze(1)).squeeze(1)
    return torch.where(idx >= 0, last, torch.zeros_like(last))


class ForecastHead(nn.Module):
    """Turns one encoding per (batch, station) into a 24 h forecast.

    Inputs of the head, concatenated:
      * the encoder output for this station        (D)
      * a learned embedding of the station          (E)  "who am I": a traffic site in
                                                         Antwerp behaves differently from a
                                                         rural site in the Ardennes
      * what is known about the H future hours      (H * F_fut): calendar features, and
                                                         the weather at THIS station when
                                                         weather is used
    Output: (B, H, N, out_size).

    residual=True: the network predicts a CORRECTION to "last observed value" instead of
    the value itself. That is the best reference at short lead times, so the model starts
    from a sensible forecast and only has to learn how to depart from it.
    """

    def __init__(self, d_in: int, n_nodes: int, n_time_features: int, horizon: int,
                 out_size: int, hidden: int = 128, emb_dim: int = 8, dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.horizon, self.out_size, self.residual = horizon, out_size, residual
        self.station_emb = nn.Embedding(n_nodes, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_in + emb_dim + horizon * n_time_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * out_size),
        )

    def use_static_features(self, feats: torch.Tensor) -> None:
        """Describe each station by its static features (coordinates, altitude, type...)
        instead of a learned identity. feats: (N, S). Needed for the spatial holdout: a
        station never seen in training has no trained embedding, but it has features."""
        self.register_buffer("static_feats", feats.float())
        self.static_proj = nn.Linear(feats.shape[1], self.station_emb.embedding_dim)
        del self.station_emb

    def station_vectors(self) -> torch.Tensor:
        """(N, E): one "who am I" vector per station."""
        if hasattr(self, "static_proj"):
            return self.static_proj(self.static_feats)
        return self.station_emb.weight

    def forward(self, enc: torch.Tensor, x: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        B, N, _ = enc.shape
        emb = self.station_vectors().unsqueeze(0).expand(B, -1, -1)        # (B, N, E)
        if x_future.dim() == 4:      # (B, H, N, F_fut): per-station future features
            fut = x_future.permute(0, 2, 1, 3).reshape(B, N, -1)            # (B, N, H*F_fut)
        else:                        # (B, H, F_time): same for every station (older code)
            fut = x_future.reshape(B, 1, -1).expand(-1, N, -1)
        out = self.mlp(torch.cat([enc, emb, fut], dim=-1))                 # (B, N, H*K)
        out = out.view(B, N, self.horizon, self.out_size).permute(0, 2, 1, 3)  # (B, H, N, K)

        if self.residual:
            base = last_observed(x).unsqueeze(1).unsqueeze(-1)              # (B, 1, N, 1)
            if self.out_size == 2:   # gaussian_nll: [mean, log_std] -> shift the mean only
                out = torch.cat([out[..., :1] + base, out[..., 1:]], dim=-1)
            else:                    # point forecast or quantiles: shift every output
                out = out + base
        return out.squeeze(-1) if self.out_size == 1 else out

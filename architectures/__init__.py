"""
Model registry. Every model follows the same contract:

    model = build_model("lstm", info, out_size=1, **hyperparameters)
    pred  = model(x, x_future)
        x        : (B, L, N, F_in)   history
        x_future : (B, H, N, F_fut)  known-in-advance features of the hours to predict
                                     (weather at each station + calendar)
        pred     : (B, H, N) if out_size == 1, else (B, H, N, out_size)

To add a model (e.g. a GNN in phase 2): write it in a new file, decorate the class with
@register_model("gnn"), and import that file below. train.py needs no change.
"""

from __future__ import annotations

import torch.nn as nn

MODELS: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    def deco(cls):
        MODELS[name] = cls
        return cls
    return deco


def build_model(name: str, info: dict, out_size: int = 1, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](
        n_nodes=info["n_nodes"], n_features=info["n_features"],
        # n_time_features = number of known-future features per hour (calendar + weather);
        # the name is kept so that the Phase-1 model files need no change.
        n_time_features=info.get("n_future_features", info["n_time_features"]),
        history=info["history"],
        horizon=info["horizon"], out_size=out_size, **kwargs,
    )


# Import model files so their @register_model decorators run
from . import mlp, lstm  # noqa: E402,F401
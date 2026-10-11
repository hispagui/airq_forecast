"""
metrics.py
==========
Losses (what gradient descent minimizes) and metrics (what we report), kept separate.

LOSSES  -> computed in the NORMALIZED space, differentiable, masked.
           Pick one by name:  loss_fn = get_loss("mse")
           Signature: loss_fn(pred, y, mask) -> scalar tensor

METRICS -> computed in ug/m3 after de-normalization, on a whole split at once.
           Use the Evaluator: feed it batches, then call .compute()

Adding your own
---------------
    @register_loss("my_loss")
    def my_loss(pred, y, mask): ...           # return a scalar tensor

    @register_metric("my_metric")
    def my_metric(pred, true, ctx): ...       # pred/true: (S, H, N) numpy, NaN = missing
                                              # return a float, or an array of shape (H,)
That's all: it is then usable by name everywhere (train.py, configs).

Shapes used throughout
----------------------
    pred, y, mask : (B, H, N)  batch, horizon, stations
    quantile loss : pred is (B, H, N, Q), one output per quantile
    gaussian_nll  : pred is (B, H, N, 2), [mean, log_std]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# LOSSES
# --------------------------------------------------------------------------- #
LOSSES: dict[str, Callable] = {}
QUANTILES = (0.1, 0.5, 0.9)  # used by the quantile loss; the median is the point forecast


def register_loss(name: str):
    def deco(fn):
        LOSSES[name] = fn
        return fn
    return deco


def get_loss(name: str) -> Callable:
    if name not in LOSSES:
        raise KeyError(f"Unknown loss '{name}'. Available: {sorted(LOSSES)}")
    return LOSSES[name]


def _masked_mean(per_entry: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(M * l) / sum(M): mean over the entries that were actually observed."""
    mask = mask.to(per_entry.dtype)
    return (per_entry * mask).sum() / mask.sum().clamp(min=1.0)


@register_loss("mse")
def mse_loss(pred, y, mask):
    """Squared error -> estimates the conditional mean."""
    return _masked_mean((pred - y) ** 2, mask)


@register_loss("mae")
def mae_loss(pred, y, mask):
    """Absolute error -> estimates the conditional median, robust to outliers."""
    return _masked_mean((pred - y).abs(), mask)


@register_loss("huber")
def huber_loss(pred, y, mask, delta: float = 1.0):
    """Quadratic for |e| <= delta, linear beyond. delta is in normalized units (~1 std)."""
    per = torch.nn.functional.huber_loss(pred, y, reduction="none", delta=delta)
    return _masked_mean(per, mask)


@register_loss("quantile")
def quantile_loss(pred, y, mask, quantiles=QUANTILES):
    """Pinball loss, one output per quantile. pred: (B, H, N, Q)."""
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = y.unsqueeze(-1) - pred                              # (B, H, N, Q)
    per = torch.maximum(q * err, (q - 1) * err).mean(dim=-1)  # average over quantiles
    return _masked_mean(per, mask)


@register_loss("gaussian_nll")
def gaussian_nll_loss(pred, y, mask):
    """Model outputs a mean and a log standard deviation. pred: (B, H, N, 2)."""
    mean, log_std = pred[..., 0], pred[..., 1].clamp(-5, 5)
    per = 0.5 * ((y - mean) / log_std.exp()) ** 2 + log_std
    return _masked_mean(per, mask)


def point_forecast(pred: torch.Tensor, loss_name: str) -> torch.Tensor:
    """Turn the raw model output into one value per (B, H, N), whatever the loss."""
    if loss_name == "quantile":
        return pred[..., QUANTILES.index(0.5)]
    if loss_name == "gaussian_nll":
        return pred[..., 0]
    return pred


def output_size(loss_name: str) -> int:
    """How many numbers the model must output per (hour, station) for this loss."""
    return {"quantile": len(QUANTILES), "gaussian_nll": 2}.get(loss_name, 1)


# --------------------------------------------------------------------------- #
# METRICS (numpy, ug/m3, NaN = missing)
# --------------------------------------------------------------------------- #
METRICS: dict[str, Callable] = {}


def register_metric(name: str):
    def deco(fn):
        METRICS[name] = fn
        return fn
    return deco


def _err(pred, true):
    return pred - true  # NaN propagates where the target is missing


@register_metric("rmse")
def rmse(pred, true, ctx):
    return float(np.sqrt(np.nanmean(_err(pred, true) ** 2)))


@register_metric("mae")
def mae(pred, true, ctx):
    return float(np.nanmean(np.abs(_err(pred, true))))


@register_metric("bias")
def bias(pred, true, ctx):
    """Mean of (pred - true). Negative = the model under-predicts."""
    return float(np.nanmean(_err(pred, true)))


@register_metric("corr")
def corr(pred, true, ctx):
    """Pearson correlation between predicted and observed values."""
    ok = ~np.isnan(true) & ~np.isnan(pred)
    if pred[ok].std() == 0 or true[ok].std() == 0:  # constant forecast: undefined
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(pred[ok], true[ok])[0, 1])


@register_metric("rmse_per_horizon")
def rmse_per_horizon(pred, true, ctx):
    """RMSE for each lead time h = 1..H. Shows how fast the forecast degrades."""
    return np.sqrt(np.nanmean(_err(pred, true) ** 2, axis=(0, 2)))


@register_metric("mae_per_horizon")
def mae_per_horizon(pred, true, ctx):
    return np.nanmean(np.abs(_err(pred, true)), axis=(0, 2))


@register_metric("rmse_per_station")
def rmse_per_station(pred, true, ctx):
    """RMSE for each station. Useful later to see where the graph model helps."""
    return np.sqrt(np.nanmean(_err(pred, true) ** 2, axis=(0, 1)))


def _exceedance_counts(pred, true, threshold):
    ok = ~np.isnan(true)
    p, t = pred[ok] >= threshold, true[ok] >= threshold
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    return tp, fp, fn


@register_metric("exceed_precision")
def exceed_precision(pred, true, ctx):
    """Of the hours we flagged as polluted, how many really were."""
    tp, fp, _ = _exceedance_counts(pred, true, ctx["threshold"])
    return tp / (tp + fp) if tp + fp else float("nan")


@register_metric("exceed_recall")
def exceed_recall(pred, true, ctx):
    """Of the truly polluted hours, how many we caught."""
    tp, _, fn = _exceedance_counts(pred, true, ctx["threshold"])
    return tp / (tp + fn) if tp + fn else float("nan")


@register_metric("exceed_f1")
def exceed_f1(pred, true, ctx):
    tp, fp, fn = _exceedance_counts(pred, true, ctx["threshold"])
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


@register_metric("exceed_csi")
def exceed_csi(pred, true, ctx):
    """Critical Success Index tp/(tp+fp+fn), the standard score in weather forecasting."""
    tp, fp, fn = _exceedance_counts(pred, true, ctx["threshold"])
    return tp / (tp + fp + fn) if tp + fp + fn else float("nan")


@register_metric("skill_rmse")
def skill_rmse(pred, true, ctx):
    """1 - RMSE_model / RMSE_persistence.  > 0 means better than persistence, 1 = perfect."""
    base = ctx.get("baseline")
    if base is None:
        return float("nan")
    ok = ~np.isnan(true) & ~np.isnan(base)  # compare on exactly the same entries
    r_model = np.sqrt(np.nanmean((pred[ok] - true[ok]) ** 2))
    r_base = np.sqrt(np.mean((base[ok] - true[ok]) ** 2))
    return float(1 - r_model / r_base)


DEFAULT_METRICS = ("rmse", "mae", "bias", "corr", "skill_rmse",
                   "exceed_precision", "exceed_recall", "exceed_csi", "rmse_per_horizon")


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def persistence_forecast(x: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """'Same hour yesterday': forecast for t0+h is the value at t0+h-24.
    x: (B, L, N, F) with feature 0 = normalized PM2.5 and feature 1 = observed flag.
    Returns (forecast, valid) of shape (B, H, N), normalized. Needs history >= 24."""
    last_day = x[:, -24:, :, 0]
    last_ok = x[:, -24:, :, 1] > 0.5
    reps = -(-horizon // 24)  # ceil: for H > 24, repeat the daily cycle
    return last_day.repeat(1, reps, 1)[:, :horizon], last_ok.repeat(1, reps, 1)[:, :horizon]


def last_value_forecast(x: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """'Nothing changes': repeat the most recent OBSERVED value for all H future hours.
    Returns (forecast, valid) of shape (B, H, N), normalized."""
    vals, ok = x[..., 0], x[..., 1] > 0.5                       # (B, L, N)
    L = vals.shape[1]
    pos = torch.arange(L, device=x.device).view(1, L, 1)
    last_idx = torch.where(ok, pos, torch.full_like(pos, -1)).amax(dim=1)  # (B, N), -1 = never
    last = vals.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)    # (B, N)
    valid = (last_idx >= 0).unsqueeze(1).expand(-1, horizon, -1)
    return last.unsqueeze(1).expand(-1, horizon, -1), valid


BASELINES = {"last_value": last_value_forecast, "persistence": persistence_forecast}


# --------------------------------------------------------------------------- #
# Evaluator: accumulate a whole split, then compute every metric
# --------------------------------------------------------------------------- #
class Evaluator:
    """
    ev = Evaluator(normalizer, threshold=25.0, skill_baseline="last_value")
    for batch in loader:
        ev.update(point_pred, batch)            # point_pred normalized, (B, H, N)
    results = ev.compute()                      # dict name -> value
    print(format_results(results))

    The RMSE of every reference forecast in BASELINES is also reported ("last_value_rmse",
    "persistence_rmse"); `skill_rmse` is computed against `skill_baseline`.
    """

    def __init__(self, normalizer, metrics=DEFAULT_METRICS, threshold: float = 25.0,
                 skill_baseline: str | None = "last_value", with_persistence: bool = True):
        self.norm = normalizer
        self.metric_names = list(metrics)
        self.threshold = threshold
        self.skill_baseline = skill_baseline
        self.with_baselines = with_persistence and skill_baseline is not None
        self.reset()

    def reset(self):
        self._pred, self._true = [], []
        self._base = {k: [] for k in BASELINES}

    @torch.no_grad()
    def update(self, pred: torch.Tensor, batch: dict):
        norm = self.norm.to(pred.device)
        y, mask = batch["y"].to(pred.device), batch["y_mask"].to(pred.device)
        true = norm.denormalize(y).masked_fill(~mask, float("nan"))
        self._pred.append(norm.denormalize(pred).float().cpu().numpy())
        self._true.append(true.float().cpu().numpy())
        if self.with_baselines:
            x = batch["x"].to(pred.device)
            for name, fn in BASELINES.items():
                base, ok = fn(x, y.shape[1])
                base = norm.denormalize(base).masked_fill(~ok, float("nan"))
                self._base[name].append(base.float().cpu().numpy())

    def compute(self, metrics=None) -> dict:
        pred, true = np.concatenate(self._pred), np.concatenate(self._true)
        bases = {k: np.concatenate(v) for k, v in self._base.items() if v}
        ctx = {"threshold": self.threshold, "baseline": bases.get(self.skill_baseline)}
        out = {name: METRICS[name](pred, true, ctx) for name in (metrics or self.metric_names)}
        for name, base in bases.items():  # report the references themselves
            out[f"{name}_rmse"] = rmse(base, true, ctx)
            out[f"{name}_rmse_per_horizon"] = rmse_per_horizon(base, true, ctx)
        return out


def format_results(results: dict, prefix: str = "") -> str:
    """One readable line for scalars, plus compact per-horizon arrays."""
    scalars = [f"{k}={v:.3f}" for k, v in results.items() if np.isscalar(v)]
    lines = [prefix + "  ".join(scalars)]
    for k, v in results.items():
        if not np.isscalar(v):
            arr = np.asarray(v)
            pick = [i for i in (0, 5, 11, 23) if i < len(arr)]
            lines.append(f"{prefix}{k}: " + "  ".join(f"h+{i + 1}={arr[i]:.2f}" for i in pick))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
#   python metrics.py              -> on YOUR preprocessed data (data/processed)
#   python metrics.py --synthetic  -> on random numbers (no data needed)
# --------------------------------------------------------------------------- #
def _synthetic_test():
    torch.manual_seed(0)
    B, H, N = 8, 24, 5
    y = torch.randn(B, H, N)
    mask = torch.rand(B, H, N) > 0.2
    pred = y + 0.1 * torch.randn(B, H, N)
    y_garbage = y.masked_fill(~mask, 1e6)
    for name in ("mse", "mae", "huber"):
        a, b = get_loss(name)(pred, y, mask), get_loss(name)(pred, y_garbage, mask)
        assert torch.allclose(a, b), name
    print("OK: garbage in masked entries does not change any loss.")


def _real_data_test(processed: str, split: str, max_batches: int | None):
    """Score simple reference forecasts on real data, so you know what loss values mean."""
    from features.dataset import make_dataloaders  # local import: metrics.py works without dataset.py

    loaders, info = make_dataloaders(processed, history=72, horizon=24, batch_size=256)
    data, norm = info["data"], info["normalizer"]
    H = info["horizon"]

    # Climatology = average normalized PM2.5 per (station, hour of day UTC), learned on TRAIN only
    hours = ((data.timestamps // 3_600_000_000_000) % 24 if data.timestamps.max() > 1e17
             else (data.timestamps // 3_600_000_000) % 24).astype(np.int64)
    a, z = data.meta["splits"]["train"]
    v, m = data.values[a:z].numpy(), data.mask[a:z].numpy()
    clim = np.zeros((24, data.n_nodes), dtype=np.float32)
    for h in range(24):
        sel = hours[a:z] == h
        clim[h] = (v[sel] * m[sel]).sum(0) / np.maximum(m[sel].sum(0), 1)
    clim, hours_t = torch.from_numpy(clim), torch.from_numpy(hours)

    def zero(b):        # always predict the training average
        return torch.zeros_like(b["y"])

    def last_value(b):  # repeat the last observed hour for all 24 future hours
        return b["x"][:, -1:, :, 0].expand(-1, H, -1)

    def persistence(b):  # same hour yesterday
        return persistence_forecast(b["x"], H)[0]

    def climatology(b):  # typical value for this station at this hour of day
        idx = b["t0"][:, None] + torch.arange(H)[None, :]        # (B, H) time indices
        return clim[hours_t[idx]]                                # (B, H, N)

    references = {"zero (train average)": zero, "last observed value": last_value,
                  "persistence (yesterday)": persistence, "climatology (hour x station)": climatology}

    print(f"Reference forecasts on the {split} split ({info['sizes'][split]:,} samples)\n")
    print(f"{'forecast':<30}{'MSE loss':>10}{'MAE loss':>10}{'RMSE ug/m3':>12}"
          f"{'MAE ug/m3':>11}{'bias':>8}{'skill':>8}{'CSI':>7}")
    per_h = {}
    for name, fn in references.items():
        ev = Evaluator(norm)
        tot = {"mse": 0.0, "mae": 0.0}; nb = 0
        for i, b in enumerate(loaders[split]):
            if max_batches and i >= max_batches:
                break
            pred = fn(b)
            for k in tot:
                tot[k] += get_loss(k)(pred, b["y"], b["y_mask"]).item()
            nb += 1
            ev.update(pred, b)
        r = ev.compute()
        print(f"{name:<30}{tot['mse'] / nb:>10.3f}{tot['mae'] / nb:>10.3f}{r['rmse']:>12.2f}"
              f"{r['mae']:>11.2f}{r['bias']:>8.2f}{r['skill_rmse']:>8.3f}{r['exceed_csi']:>7.3f}")
        per_h[name] = r["rmse_per_horizon"]
    print("\nRMSE (ug/m3) per lead time:")
    for name, arr in per_h.items():
        print(f"  {name:<30}" + "  ".join(f"h+{i + 1}={arr[i]:.2f}" for i in (0, 5, 11, 23) if i < len(arr)))
    print("\nHow to read this: losses are in normalized units. Predicting the average gives MSE ~ 1.")
    print("A trained model must get below the best line above to be worth anything.")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max-batches", type=int, default=None, help="limit for a quick run")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    _synthetic_test()
    if not args.synthetic:
        try:
            from features.dataset import find_processed_dir
        except ImportError:
            raise SystemExit("\n! Cannot import dataset.py. Put metrics.py and dataset.py in the "
                             "same folder (or run from the folder that contains dataset.py).")
        try:
            processed = find_processed_dir(args.processed)
        except FileNotFoundError as e:
            raise SystemExit(f"\n! {e}")
        print()
        _real_data_test(str(processed), args.split, args.max_batches)
"""
evaluate.py
===========
Detailed evaluation of trained models: where, when and how well they forecast PM2.5.

    python evaluate.py                                   # latest run, test year (2025)
    python evaluate.py runs/lstm_mse_... runs/mlp_mse_...   # compare several models
    python evaluate.py --split val                       # validation year (2024)
    python evaluate.py --split recent                    # 2026 so far (needs preprocess --keep-recent)
    python evaluate.py --from 2025-11-01 --to 2026-03-01 # any period (Brussels dates)
    python evaluate.py --plots                           # + figures
    python evaluate.py --metrics rmse mae bias exceed_f1 # choose metrics (any name in metrics.py)
    python evaluate.py --breakdowns horizon month level  # choose breakdowns (see BREAKDOWNS)
    python evaluate.py --breakdowns wind_sector rain     # by weather (needs weather.npz)

It compares every model against reference forecasts (last value, persistence,
climatology) on EXACTLY the same hours and stations, then breaks the errors down by
lead time, station, month, season, hour of day, weekday and pollution level.

Outputs: <out>/   (default: runs/<run>/evaluation_<split>/, or runs/comparison_<date>/)
    summary.csv               one row per forecast (models + references), overall metrics
    alerts.csv                alert scores at several thresholds
    by_<breakdown>.csv        one table per breakdown (rows = groups, columns = forecast x metric)
    summary.json              everything above in one file
    *.png                     with --plots

EXTENDING (three registries, no other change needed)
----------------------------------------------------
1. New METRIC: register it in metrics.py (or in the "your additions" section below)
       @register_metric("my_metric")
       def my_metric(pred, true, ctx): ...      # (S, H, N) arrays in ug/m3, NaN = missing
   then:  python evaluate.py --metrics rmse my_metric
2. New BREAKDOWN (a way to group errors):
       @register_breakdown("my_grouping")
       def my_grouping(ctx): return labels, names   # labels broadcastable to (S, H, N)
3. New REFERENCE forecast to compare against:
       @register_reference("my_reference")
       def my_reference(ctx): return forecast        # (S, H, N) ug/m3, NaN where undefined
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import metrics as M  # the metric registry lives there
from dataset import window_inputs
from metrics import METRICS, register_metric  # noqa: F401  (register_metric used by additions)

LOCAL_TZ = "Europe/Brussels"
DEFAULT_METRICS = ["rmse", "mae", "bias", "corr", "skill_rmse"]
DEFAULT_BREAKDOWNS = ["horizon", "month", "season", "hour", "weekday", "level", "station",
                      "wind_sector", "rain"]
WEATHER_BREAKDOWNS = {"wind_sector", "rain"}   # skipped when no weather data is available
DEFAULT_THRESHOLDS = [15.0, 25.0, 50.0]
ALERT_METRICS = ["exceed_precision", "exceed_recall", "exceed_f1", "exceed_csi"]


# =========================================================================== #
# Your additions: new metrics, breakdowns and references go here
# =========================================================================== #
# Example metric (uncomment to use with --metrics max_error):
# @register_metric("max_error")
# def max_error(pred, true, ctx):
#     return float(np.nanmax(np.abs(pred - true)))
#
# Ideas for later:
#   * probabilistic scores (CRPS, interval coverage) for quantile models: they need the
#     full quantile output, so store it in train.py's test_predictions.npz first.
#   * skill against climatology instead of last value: set ctx["baseline"] accordingly.
#   * peak timing error: hours between predicted and observed daily maximum.


# =========================================================================== #
# Registries
# =========================================================================== #
BREAKDOWNS: dict[str, Callable] = {}
REFERENCES: dict[str, Callable] = {}


def register_breakdown(name: str):
    def deco(fn):
        BREAKDOWNS[name] = fn
        return fn
    return deco


def register_reference(name: str):
    def deco(fn):
        REFERENCES[name] = fn
        return fn
    return deco


# =========================================================================== #
# Loading data and predictions
# =========================================================================== #
def latest_run(runs_dir="runs") -> Path:
    runs = sorted(Path(runs_dir).glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"No trained model in {runs_dir}/. Run train.py first.")
    return runs[-1].parent


def timeline(data) -> pd.DatetimeIndex:
    ts = data.timestamps
    unit = "ns" if ts.max() > 1e17 else "us"
    return pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, utc=True))


def select_t0s(data, start: int, end: int, L: int, H: int,
               min_hist: float = 0.5, min_targ: float = 0.5) -> np.ndarray:
    """Forecast start indices whose 24 targets lie in [start, end), with enough real data.
    Same rule as dataset.AirQualityWindows, so the test numbers match train.py."""
    first, last = max(start, L), min(end, len(data.timestamps)) - H
    if last < first:
        raise SystemExit("The chosen period is too short (needs at least history + 24 h).")
    cand = np.arange(first, last + 1)
    m = data.mask.numpy()
    cum = np.concatenate([[0], np.cumsum(m.sum(axis=1), dtype=np.int64)])
    n = m.shape[1]
    hv = (cum[cand] - cum[cand - L]) / (L * n)
    tv = (cum[cand + H] - cum[cand]) / (H * n)
    return cand[(hv >= min_hist) & (tv >= min_targ)]


@torch.no_grad()
def run_model(fc, t0s: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Model forecasts (S, H, N) in ug/m3 for the given start indices."""
    d, L, H = fc.data, fc.L, fc.H
    out = []
    for i in range(0, len(t0s), batch_size):
        x, x_future = window_inputs(d, t0s[i:i + batch_size], L, H, fc.use_weather)
        pred = fc.model(x, x_future)
        out.append(fc.norm.denormalize(M.point_forecast(pred, fc.cfg["loss"])).numpy())
    return np.concatenate(out).astype(np.float32)


def resolve_period(args, data, times):
    splits = data.meta["splits"]
    if args.date_from or args.date_to:
        def pos(s, default):
            if s is None:
                return default
            ts = pd.Timestamp(s).tz_localize(LOCAL_TZ).tz_convert("UTC")
            return int(times.searchsorted(ts))
        a, z = pos(args.date_from, 0), pos(args.date_to, len(times))
        return f"{args.date_from or 'start'}_to_{args.date_to or 'end'}".replace(":", ""), a, z
    if args.split == "recent":
        a = splits["test"][1]
        if a >= len(times):
            raise SystemExit("No data after the test year. Run: preprocess.py --keep-recent")
        return "recent", a, len(times)
    a, z = splits[args.split]
    return args.split, a, z


# =========================================================================== #
# References (forecasts that need no learning)
# =========================================================================== #
@register_reference("last_value")
def ref_last_value(ctx):
    """Most recent observed value within the history window, repeated 24 times."""
    raw, mask, t0s, L, H = ctx["raw"], ctx["mask"], ctx["t0s"], ctx["L"], ctx["H"]
    T, N = raw.shape
    idx = np.where(mask, np.arange(T)[:, None], -1)
    last_idx = np.maximum.accumulate(idx, axis=0)                     # (T, N) last observed <= t
    li = last_idx[t0s - 1]                                            # (S, N)
    vals = raw[np.clip(li, 0, None), np.arange(N)[None, :]]
    vals = np.where(li >= t0s[:, None] - L, vals, np.nan)
    return np.repeat(vals[:, None, :], H, axis=1)


@register_reference("persistence")
def ref_persistence(ctx):
    """Same hour yesterday."""
    raw, mask, t0s, H = ctx["raw"], ctx["mask"], ctx["t0s"], ctx["H"]
    src = t0s[:, None] + np.arange(H)[None, :] - 24 * (1 + np.arange(H)[None, :] // 24)
    return np.where(mask[src], raw[src], np.nan)


@register_reference("climatology")
def ref_climatology(ctx):
    """Typical value for this station, month and local hour, learned on the TRAIN years."""
    raw, mask, times, (a, z) = ctx["raw"], ctx["mask"], ctx["times"], ctx["train_range"]
    local = times.tz_convert(LOCAL_TZ)
    key = (local.month.values - 1) * 24 + local.hour.values           # 288 (month, hour) cells
    lg = np.where(mask[a:z], np.log1p(raw[a:z]), 0.0)
    cnt = np.zeros((288, raw.shape[1])); tot = np.zeros((288, raw.shape[1]))
    np.add.at(tot, key[a:z], lg); np.add.at(cnt, key[a:z], mask[a:z])
    clim = np.expm1(tot / np.maximum(cnt, 1))                          # median-like, like the models
    clim[cnt == 0] = np.nan
    return clim[key[ctx["target_idx"]]]                                # (S, H, N)


# =========================================================================== #
# Breakdowns: how to group the (sample, lead time, station) entries
# =========================================================================== #
SEASONS = ["winter (DJF)", "spring (MAM)", "summer (JJA)", "autumn (SON)"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVELS = [0, 10, 25, 50, np.inf]


@register_breakdown("horizon")
def bd_horizon(ctx):
    H = ctx["H"]
    return np.arange(H)[None, :, None], [f"h+{h + 1}" for h in range(H)]


@register_breakdown("station")
def bd_station(ctx):
    N = len(ctx["stations"])
    return np.arange(N)[None, None, :], list(ctx["stations"])


@register_breakdown("month")
def bd_month(ctx):
    return (ctx["target_local"].month.values - 1).reshape(ctx["target_idx"].shape)[..., None], \
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@register_breakdown("season")
def bd_season(ctx):
    m = ctx["target_local"].month.values.reshape(ctx["target_idx"].shape)
    return ((m % 12) // 3)[..., None], SEASONS


@register_breakdown("hour")
def bd_hour(ctx):
    """Local hour of the TARGET (the hour being forecast)."""
    return ctx["target_local"].hour.values.reshape(ctx["target_idx"].shape)[..., None], \
        [f"{h:02d}h" for h in range(24)]


@register_breakdown("weekday")
def bd_weekday(ctx):
    return ctx["target_local"].dayofweek.values.reshape(ctx["target_idx"].shape)[..., None], WEEKDAYS


WIND_SECTORS = ["from N", "from NE", "from E", "from SE", "from S", "from SW", "from W",
                "from NW", "calm (<1 m/s)"]


def _weather_at_targets(ctx) -> np.ndarray:
    """(S, H, N, 3): wind_u, wind_v (m/s) and precipitation (mm) at each target hour."""
    if ctx.get("weather_raw") is None:
        raise SystemExit("This breakdown needs weather data: run prepare_weather.py.")
    return ctx["weather_raw"][ctx["target_idx"]]


@register_breakdown("wind_sector")
def bd_wind_sector(ctx):
    """By the direction the wind blows FROM at the station and target hour (8 sectors + calm).
    Shows whether errors depend on the air mass: e.g. easterly (continental) episodes."""
    w = _weather_at_targets(ctx)
    u, v = w[..., 0], w[..., 1]
    speed = np.hypot(u, v)
    dir_from = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    lab = (((dir_from + 22.5) // 45) % 8).astype(np.int64)
    lab = np.where(speed < 1.0, 8, lab)
    return np.where(np.isfinite(speed), lab, -1), WIND_SECTORS


@register_breakdown("rain")
def bd_rain(ctx):
    """By precipitation at the target hour: rain washes particles out of the air."""
    p = _weather_at_targets(ctx)[..., 2]
    lab = np.digitize(np.nan_to_num(p, nan=-1), [0.1, 1.0])
    return np.where(np.isfinite(p), lab, -1), ["dry (<0.1 mm/h)", "light (0.1-1 mm/h)",
                                               "rain (>=1 mm/h)"]


@register_breakdown("level")
def bd_level(ctx):
    """By OBSERVED concentration: how the model does on clean vs polluted hours."""
    lab = np.digitize(np.nan_to_num(ctx["true"], nan=-1), LEVELS[1:-1])
    names = [f"{LEVELS[i]:g}-{LEVELS[i + 1]:g} ug/m3" if np.isfinite(LEVELS[i + 1])
             else f">{LEVELS[i]:g} ug/m3" for i in range(len(LEVELS) - 1)]
    return lab, names


# =========================================================================== #
# Core computation
# =========================================================================== #
def scalar_metrics(pred, true, ctx, names):
    """Every requested metric that returns one number (array-valued ones are skipped here)."""
    out = {}
    for name in names:
        try:
            v = METRICS[name](pred, true, ctx)
        except Exception:  # e.g. a per-horizon metric on a flattened group
            continue
        if np.isscalar(v) or np.ndim(v) == 0:
            out[name] = float(v)
    out["n"] = int(np.isfinite(true).sum())
    return out


def evaluate_all(forecasts: dict, true: np.ndarray, ctx: dict, metric_names, breakdowns,
                 thresholds) -> dict:
    base_ctx = {**ctx, "baseline": forecasts.get("last_value"), "threshold": 25.0}
    res = {"summary": {}, "alerts": {}, "breakdowns": {}}

    for fname, pred in forecasts.items():
        res["summary"][fname] = scalar_metrics(pred, true, base_ctx, metric_names)
        for thr in thresholds:
            c = {**base_ctx, "threshold": thr}
            res["alerts"][(fname, thr)] = {m: float(METRICS[m](pred, true, c)) for m in ALERT_METRICS}
            res["alerts"][(fname, thr)]["observed_hours"] = int((true >= thr).sum())

    shape = true.shape
    for bname in breakdowns:
        labels, names = BREAKDOWNS[bname]({**ctx, "true": true})
        labels = np.broadcast_to(labels, shape)
        table = {}
        finite = np.isfinite(true)
        for g, gname in enumerate(names):
            sel = (labels == g) & finite              # entries of this group (flattened)
            if not sel.any():
                continue
            true_g = true[sel]
            base = base_ctx["baseline"]
            cg = {**base_ctx, "baseline": None if base is None else base[sel]}
            table[gname] = {(f, m): v for f, pred in forecasts.items()
                            for m, v in scalar_metrics(pred[sel], true_g, cg, metric_names).items()}
        res["breakdowns"][bname] = table
    return res


# =========================================================================== #
# Reporting
# =========================================================================== #
def print_report(res, title, model_names, out_dir):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    summ = pd.DataFrame(res["summary"]).T
    summ.index.name = "forecast"
    summ["n"] = summ["n"].astype(int)
    print("\nOverall (same hours and stations for every forecast; ug/m3):")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summ.to_string())
    summ.to_csv(out_dir / "summary.csv")

    alerts = pd.DataFrame(res["alerts"]).T
    alerts.index.names = ["forecast", "threshold"]
    alerts["observed_hours"] = alerts["observed_hours"].astype(int)
    alerts.index = alerts.index.set_levels(
        [f"{t:g} ug/m3" for t in alerts.index.levels[1]], level=1)
    print("\nAlerts (hour flagged when forecast >= threshold):")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(alerts.to_string())
    alerts.to_csv(out_dir / "alerts.csv")

    for bname, table in res["breakdowns"].items():
        df = pd.DataFrame(table).T
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["forecast", "metric"])
        df.index.name = bname
        df.to_csv(out_dir / f"by_{bname}.csv")
        # Compact console view: RMSE of each forecast per group
        if "rmse" in df.columns.get_level_values("metric"):
            view = df.xs("rmse", axis=1, level="metric")
            if bname == "horizon":
                view = view.iloc[[0, 2, 5, 11, 17, 23]] if len(view) >= 24 else view
            if bname == "station":
                view = view.sort_values(model_names[0], ascending=False)
                if len(view) > 10:
                    print(f"\nRMSE by station (5 worst / 5 best for {model_names[0]}; "
                          "full table in CSV):")
                    view = pd.concat([view.head(5), view.tail(5)])
                else:
                    print("\nRMSE by station:")
            else:
                print(f"\nRMSE by {bname}:")
            n = df.xs("n", axis=1, level="metric").iloc[:, 0].reindex(view.index)
            with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
                print(view.assign(**{"n entries": n.astype(int)}).to_string())


def save_json(res, path, meta):
    def key(k):
        return " @ ".join(map(str, k)) if isinstance(k, tuple) else str(k)

    def clean(o):
        if isinstance(o, dict):
            return {key(k): clean(v) for k, v in o.items()}
        if isinstance(o, float):
            return None if o != o else round(o, 5)
        return o
    path.write_text(json.dumps({"meta": meta, **clean(res)}, indent=2))


# =========================================================================== #
# Plots
# =========================================================================== #
# Series colors: models get the categorical slots in fixed order; references are neutral greys
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
REF_STYLE = {"last_value": ("#52514e", "--"), "persistence": ("#9a9893", ":"),
             "climatology": ("#9a9893", "-.")}


def make_plots(res, true, forecasts, model_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#9a9893", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9})

    def style(name):
        if name in model_names:
            return MODEL_COLORS[model_names.index(name) % 4], "-", 2.0
        c, ls = REF_STYLE.get(name, ("#9a9893", "-"))
        return c, ls, 1.5

    def line_plot(bname, fname, title, xlabel, rotate=False):
        if bname not in res["breakdowns"]:
            return
        df = pd.DataFrame(res["breakdowns"][bname]).T
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        rmse = df.xs("rmse", axis=1, level=1)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(rmse))
        for name in rmse.columns:
            c, ls, lw = style(name)
            ax.plot(x, rmse[name].values, color=c, ls=ls, lw=lw, label=name)
        step = max(1, len(x) // 12)
        ax.set_xticks(x[::step], rmse.index[::step], rotation=45 if rotate else 0)
        ax.set_xlabel(xlabel); ax.set_ylabel("RMSE (ug/m3)"); ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(out_dir / fname, dpi=140); plt.close(fig)

    line_plot("horizon", "rmse_by_horizon.png", "Error grows with lead time", "lead time")
    line_plot("month", "rmse_by_month.png", "Error by month of the target hour", "month")
    line_plot("hour", "rmse_by_hour.png", "Error by local hour of the target", "hour (Brussels)")

    # Observed vs forecast density for the first model, at h+1 and h+24
    pred = forecasts[model_names[0]]
    H = pred.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, h in zip(axes, (0, H - 1)):
        t, p = true[:, h].ravel(), pred[:, h].ravel()
        ok = np.isfinite(t) & np.isfinite(p)
        lim = np.nanpercentile(t[ok], 99.5)
        ax.hexbin(t[ok], p[ok], gridsize=60, extent=(0, lim, 0, lim), bins="log", cmap="Blues",
                  mincnt=1)
        ax.plot([0, lim], [0, lim], color="#52514e", lw=1, ls="--")
        ax.set_title(f"{model_names[0]}: h+{h + 1}", loc="left")
        ax.set_xlabel("observed (ug/m3)")
    axes[0].set_ylabel("forecast (ug/m3)")
    fig.suptitle("Points below the diagonal = under-prediction", x=0.01, ha="left", fontsize=9,
                 color="#52514e")
    fig.tight_layout(); fig.savefig(out_dir / "observed_vs_forecast.png", dpi=140); plt.close(fig)
    print(f"\nFigures saved in {out_dir}/")


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    p = argparse.ArgumentParser(description="Evaluate trained PM2.5 forecasters.")
    p.add_argument("runs", nargs="*", help="run folders (default: the most recent run)")
    p.add_argument("--split", default="test", choices=["val", "test", "recent"])
    p.add_argument("--from", dest="date_from", default=None, help="start date (Brussels), overrides --split")
    p.add_argument("--to", dest="date_to", default=None, help="end date (Brussels), exclusive")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help=f"scalar metrics from metrics.py. Available: {sorted(METRICS)}")
    p.add_argument("--breakdowns", nargs="+", default=None,
                   help=f"Available: {sorted(BREAKDOWNS)}")
    p.add_argument("--references", nargs="*", default=list(REFERENCES),
                   help=f"Available: {sorted(REFERENCES)}")
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--recompute", action="store_true",
                   help="re-run the model even if test_predictions.npz exists")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    explicit = args.breakdowns is not None
    args.breakdowns = args.breakdowns or DEFAULT_BREAKDOWNS
    unknown = [m for m in args.metrics if m not in METRICS] + \
              [b for b in args.breakdowns if b not in BREAKDOWNS]
    if unknown:
        raise SystemExit(f"Unknown metric/breakdown: {unknown}")

    from predict import Forecaster  # loads checkpoint + processed data + normalizer
    run_dirs = [Path(r) for r in args.runs] or [latest_run()]
    fcs = [Forecaster(r) for r in run_dirs]
    ref_fc = fcs[0]
    for fc in fcs[1:]:
        if fc.data.stations != ref_fc.data.stations or (fc.L, fc.H) != (ref_fc.L, ref_fc.H):
            raise SystemExit("Runs differ in stations, history or horizon; they can't be compared "
                             "on the same windows.")
    data, L, H = ref_fc.data, ref_fc.L, ref_fc.H
    if data.weather_raw is None:
        skipped = [b for b in args.breakdowns if b in WEATHER_BREAKDOWNS]
        if skipped and explicit:
            raise SystemExit(f"{skipped} need weather data: run download_weather.py and "
                             "prepare_weather.py first.")
        args.breakdowns = [b for b in args.breakdowns if b not in WEATHER_BREAKDOWNS]
    times = timeline(data)
    period, a, z = resolve_period(args, data, times)
    t0s = select_t0s(data, a, z, L, H)

    # ---- model forecasts ---------------------------------------------------
    # On the test year, reuse the predictions train.py saved (fast); otherwise run the model.
    saved = {}
    if period == "test" and not args.recompute:
        for rd in run_dirs:
            if (rd / "test_predictions.npz").exists():
                saved[rd.name] = np.load(rd / "test_predictions.npz")
        for s_ in saved.values():                       # keep windows present everywhere
            t0s = t0s[np.isin(t0s, s_["t0"])]
    forecasts, model_names = {}, []
    for fc, rd in zip(fcs, run_dirs):
        if rd.name in saved:
            pos = {int(t): i for i, t in enumerate(saved[rd.name]["t0"])}
            forecasts[rd.name] = saved[rd.name]["pred"][[pos[int(t)] for t in t0s]]
        else:
            print(f"Running {rd.name} on {len(t0s):,} forecasts...")
            forecasts[rd.name] = run_model(fc, t0s)
        model_names.append(rd.name)

    # ---- truth and context -------------------------------------------------
    raw, mask = data.raw, data.mask.numpy()
    target_idx = t0s[:, None] + np.arange(H)[None, :]                    # (S, H)
    true = np.where(mask[target_idx], raw[target_idx], np.nan).astype(np.float32)
    ctx = {"raw": raw, "mask": mask, "t0s": t0s, "L": L, "H": H, "times": times,
           "stations": data.stations, "target_idx": target_idx,
           "target_local": times[target_idx.ravel()].tz_convert(LOCAL_TZ),
           "train_range": tuple(data.meta["splits"]["train"]),
           "weather_raw": data.weather_raw}

    for rname in args.references:
        forecasts[rname] = REFERENCES[rname](ctx).astype(np.float32)

    # Common mask: every forecast is scored on exactly the same entries
    common = np.isfinite(true)
    for f in forecasts.values():
        common &= np.isfinite(f)
    dropped = 1 - common.sum() / max(np.isfinite(true).sum(), 1)
    true = np.where(common, true, np.nan)

    # ---- evaluate ----------------------------------------------------------
    res = evaluate_all(forecasts, true, ctx, args.metrics, args.breakdowns, args.thresholds)

    out_dir = args.out or (run_dirs[0] / f"evaluation_{period}" if len(run_dirs) == 1
                           else Path("runs") / f"comparison_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    first = times[t0s[0]].tz_convert(LOCAL_TZ); last = times[t0s[-1] + H - 1].tz_convert(LOCAL_TZ)
    title = (f"Period: {period} | targets {first:%Y-%m-%d} -> {last:%Y-%m-%d} | "
             f"{len(t0s):,} forecasts x {H} h x {len(data.stations)} stations\n"
             f"Skill = 1 - RMSE / RMSE(last_value). {dropped:.1%} of observed entries left out "
             f"because a reference was undefined there.")
    print_report(res, title, model_names, out_dir)
    save_json(res, out_dir / "summary.json",
              {"runs": [str(r) for r in run_dirs], "period": period, "n_forecasts": len(t0s),
               "metrics": args.metrics, "thresholds": args.thresholds})
    if args.plots:
        make_plots(res, true, forecasts, model_names, out_dir)
    print(f"\nTables saved in {out_dir}/")


if __name__ == "__main__":
    main()
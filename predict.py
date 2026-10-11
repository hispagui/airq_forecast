"""
predict.py
==========
Ask a trained model for a 24 h PM2.5 forecast.

    python predict.py                                  # latest run, forecast right after the
                                                       # last hour in the data (a true forecast)
    python predict.py --date "2025-01-15 08:00"        # forecast issued at that hour
                                                       # (Brussels time); compared with what
                                                       # was really observed
    python predict.py --date "2025-01-15 08:00" --plot BETR801 BELAL01   # + a figure
    python predict.py --run runs/lstm_mse_20261008_231542                 # a specific run

Output
------
    printed summary + runs/<run>/forecasts/forecast_<date>.csv
        one row per forecast hour (Brussels time), one column per station, values in ug/m3
    with --plot: forecast_<date>.png (history, forecast and observations)

Weather
-------
Models trained with weather need the weather of the 24 forecast hours:
  * forecast inside the data (a past date): the stored reanalysis weather is used;
  * forecast beyond the end of the data: the weather FORECAST is fetched from the
    Open-Meteo Forecast API (internet needed), engineered and normalized exactly like the
    training features.

From Python
-----------
    from predict import Forecaster
    fc = Forecaster("runs/lstm_mse_20261008_231542")
    table = fc.forecast("2025-01-15 08:00")      # pandas DataFrame, ug/m3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from architectures import build_model
from dataset import Normalizer, load_processed, window_inputs
from metrics import output_size, point_forecast

LOCAL_TZ = "Europe/Brussels"


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Same calendar features as preprocess.py (needed for hours beyond the data)."""
    local = index.tz_convert(LOCAL_TZ)
    hour = local.hour + local.minute / 60
    doy, dow = local.dayofyear - 1, local.dayofweek
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float),
    ], axis=1)
    return feats.astype(np.float32)


def latest_run(runs_dir: str = "runs") -> Path:
    runs = sorted(Path(runs_dir).glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No trained model found in {runs_dir}/. Run train.py first.")
    return runs[-1].parent


class Forecaster:
    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        self.run_dir = Path(run_dir)
        ckpt = torch.load(self.run_dir / "best.pt", map_location=device, weights_only=False)
        cfg = self.cfg = ckpt["config"]
        self.data = load_processed(cfg["processed"])
        if self.data.stations != cfg["stations"]:
            raise RuntimeError("The processed data has different stations than at training time. "
                               "Re-run preprocess.py with the same settings, or retrain.")
        self.norm = Normalizer(self.data.norm_mean, self.data.norm_std)
        self.L, self.H = cfg["history"], cfg["horizon"]
        # Older (Phase-1) runs have no weather keys: they used PM2.5 + calendar only
        self.use_weather = cfg.get("use_weather", False)
        if self.use_weather:
            if self.data.weather is None:
                raise RuntimeError("This model was trained with weather, but weather.npz is "
                                   "missing or outdated. Run prepare_weather.py.")
            if self.data.weather_features != cfg["weather_features"]:
                raise RuntimeError(f"Weather features differ from training: "
                                   f"{self.data.weather_features} vs {cfg['weather_features']}. "
                                   "Re-run prepare_weather.py with the same variables, or retrain.")
        info = {"n_nodes": self.data.n_nodes, "n_features": cfg["n_features"],
                "n_time_features": cfg["n_time_features"],
                "n_future_features": cfg.get("n_future_features", cfg["n_time_features"]),
                "history": self.L, "horizon": self.H}
        self.model = build_model(cfg["model"], info, out_size=output_size(cfg["loss"]),
                                 hidden=cfg["hidden"], layers=cfg["layers"],
                                 dropout=cfg["dropout"], residual=not cfg["no_residual"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        unit = "ns" if self.data.timestamps.max() > 1e17 else "us"
        self.times = pd.DatetimeIndex(pd.to_datetime(self.data.timestamps, unit=unit, utc=True))

    # ---------------------------------------------------------------- #
    def _t0_index(self, date: str | None) -> int:
        """Index of the first forecast hour. Past the end of the data = a genuine forecast."""
        T = len(self.times)
        if date is None:
            return T  # right after the last observed hour
        ts = pd.Timestamp(date)
        ts = ts.tz_localize(LOCAL_TZ) if ts.tzinfo is None else ts
        ts = ts.tz_convert("UTC").floor("h")
        t0 = int((ts - self.times[0]) / pd.Timedelta(hours=1))
        if t0 < self.L:
            raise ValueError(f"Need {self.L} h of history before {date}; data starts "
                             f"{self.times[0].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}.")
        if t0 > T:
            raise ValueError(f"The data ends {self.times[-1].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}; "
                             "the model needs the 72 h before the forecast. Update the data first.")
        return t0

    def forecast_weather(self, future_times: pd.DatetimeIndex) -> torch.Tensor:
        """Weather forecast for the given hours at every station, normalized: (H, N, F_w)."""
        from download_weather import fetch_forecast
        from prepare_weather import RAW_FOR_FEATURE, engineer_features, normalize
        wm = self.data.weather_meta
        coords = pd.DataFrame(wm["station_coords"])
        needed = sorted({r for f in wm["features"] for r in RAW_FOR_FEATURE[f]})
        print(f"Fetching the weather forecast for {len(coords)} stations (Open-Meteo)...")
        df = fetch_forecast(coords["lat"], coords["lon"], future_times, needed)
        raw = {v: df.pivot(index="time", columns="point", values=v)
                    .reindex(index=future_times, columns=range(len(coords))).to_numpy()
               for v in needed}
        names, feats, _ = engineer_features(raw, only=wm["features"])
        miss = np.isnan(feats).mean(axis=(0, 1))
        for n, m in zip(names, miss):
            if m > 0:
                print(f"  ! {n}: {m:.0%} missing in the forecast, replaced by the training average")
        return torch.from_numpy(normalize(feats, names, wm["normalization"]))

    @torch.no_grad()
    def forecast(self, date: str | None = None) -> pd.DataFrame:
        t0 = self._t0_index(date)
        d = self.data
        start = self.times[0] + pd.Timedelta(hours=t0)
        future_times = pd.date_range(start, periods=self.H, freq="h", tz="UTC")

        if t0 + self.H <= len(self.times):      # every forecast hour is in the data
            x, x_future = window_inputs(d, [t0], self.L, self.H, self.use_weather)
        else:                                   # beyond the data: calendar + weather forecast
            ft = torch.from_numpy(time_features(future_times)).unsqueeze(0)
            fw = (self.forecast_weather(future_times).unsqueeze(0)
                  if self.use_weather else None)
            x, x_future = window_inputs(d, [t0], self.L, self.H, self.use_weather,
                                        future_time=ft, future_weather=fw)

        pred = self.model(x, x_future)
        ugm3 = self.norm.denormalize(point_forecast(pred, self.cfg["loss"]))[0].numpy()  # (H, N)

        table = pd.DataFrame(ugm3, columns=d.stations,
                             index=future_times.tz_convert(LOCAL_TZ).rename("time (Brussels)"))
        self.last = {"t0": t0, "table": table}
        return table

    def observed(self, table: pd.DataFrame) -> pd.DataFrame | None:
        """What was really measured over the forecast hours (None if in the future)."""
        t0 = self.last["t0"]
        if t0 + self.H > len(self.times):
            return None
        obs = self.data.raw[t0:t0 + self.H]
        return pd.DataFrame(obs, index=table.index, columns=table.columns)

    def history(self, hours: int = 72) -> pd.DataFrame:
        t0 = self.last["t0"]
        idx = self.times[t0 - hours:t0].tz_convert(LOCAL_TZ)
        return pd.DataFrame(self.data.raw[t0 - hours:t0], index=idx, columns=self.data.stations)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="24 h PM2.5 forecast from a trained model.")
    p.add_argument("--run", default=None, help="run folder (default: most recent in runs/)")
    p.add_argument("--date", default=None,
                   help='forecast start, Brussels time, e.g. "2025-01-15 08:00" '
                        "(default: right after the last hour in the data)")
    p.add_argument("--plot", nargs="*", default=None, metavar="STATION",
                   help="save a figure for these stations (no names = 4 most polluted)")
    args = p.parse_args()

    run = Path(args.run) if args.run else latest_run()
    fc = Forecaster(run)
    table = fc.forecast(args.date)
    obs = fc.observed(table)
    print(f"\nModel: {run.name}")
    print(f"Forecast from {table.index[0]:%Y-%m-%d %H:%M} to {table.index[-1]:%Y-%m-%d %H:%M} "
          f"(Brussels time), {table.shape[1]} stations, ug/m3\n")

    # Compact view: network-wide average and the most polluted stations
    top = table.mean().sort_values(ascending=False).index[:4]
    view = pd.DataFrame({"mean (all stations)": table.mean(axis=1)})
    if obs is not None:
        view["observed mean"] = obs.mean(axis=1)
    view = pd.concat([view, table[top].add_prefix("fc ")], axis=1)
    print(view.iloc[::3].round(1).to_string())  # every 3rd hour

    if obs is not None:
        err = (table - obs).to_numpy()
        print(f"\nCompared with observations: RMSE {np.sqrt(np.nanmean(err ** 2)):.2f} ug/m3, "
              f"bias {np.nanmean(err):+.2f} ug/m3")
    else:
        print("\n(This is a genuine forecast: these hours are not in the data yet.)")

    out_dir = run / "forecasts"
    out_dir.mkdir(exist_ok=True)
    stamp = f"{table.index[0]:%Y%m%d_%H%M}"
    table.round(2).to_csv(out_dir / f"forecast_{stamp}.csv")
    print(f"Saved {out_dir / f'forecast_{stamp}.csv'}")

    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stations = args.plot or list(top)
        missing = [s for s in stations if s not in table.columns]
        if missing:
            raise SystemExit(f"Unknown station(s) {missing}. Valid codes: {list(table.columns)}")
        hist = fc.history(72)
        fig, axes = plt.subplots(len(stations), 1, figsize=(10, 2.6 * len(stations)), sharex=True)
        for ax, s in zip(np.atleast_1d(axes), stations):
            ax.plot(hist.index, hist[s], color="0.4", lw=1.2, label="past 72 h (observed)")
            ax.plot(table.index, table[s], color="tab:blue", lw=2, label="forecast")
            if obs is not None:
                ax.plot(obs.index, obs[s], color="0.1", ls="--", lw=1.2, label="observed")
            ax.axvline(table.index[0], color="0.7", lw=1)
            ax.set_ylabel("PM2.5 (ug/m3)")
            ax.set_title(s, loc="left", fontsize=10)
        np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"forecast_{stamp}.png", dpi=130)
        print(f"Saved {out_dir / f'forecast_{stamp}.png'}")


if __name__ == "__main__":
    main()

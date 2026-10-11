"""
prepare_weather.py
==================
Turn the downloaded weather (download_weather.py) into model features aligned with the
processed PM2.5 data (preprocess.py).

    python prepare_weather.py
    python prepare_weather.py --max-gap 6        # interpolate gaps up to 6 h (default)

Features (one value per hour and station)
-----------------------------------------
    wind_u, wind_v   wind vector (m/s) TOWARDS which the air moves:
                     u = -speed * sin(dir), v = -speed * cos(dir), dir = direction it comes FROM.
                     u/v instead of degrees: no jump between 359 and 0 deg, and the vector
                     can be rotated (useful for the rotation-invariant GNN of Phase 2).
    temperature      2 m air temperature (degC)
    humidity         2 m relative humidity (%)
    precip           log(1 + precipitation in mm over the preceding hour)   (skewed -> log)
    pressure         surface pressure (hPa)
    blh              log(boundary layer height in m): the height of the mixing layer. Low BLH
                     = pollution trapped near the ground (winter inversions).
Only features whose raw variables were downloaded are built.

Normalization: z-score with statistics of the TRAINING years only (no leakage). wind_u and
wind_v share one scale and are NOT centered, so the direction of the vector is preserved.

Outputs (next to the PM2.5 data, data/processed/)
    weather.npz         features (T, N, F) normalized, 0 where missing
                        raw (T, N, 3) = wind_u, wind_v (m/s), precipitation (mm), NaN missing
    weather_meta.json   feature names, normalization statistics, sources, coverage
    stations.csv        station coordinates and metadata (static node features for Phase 2)

Note on the "future" weather
----------------------------
During training and evaluation the model receives the weather of the 24 hours it predicts.
That weather comes from the reanalysis, i.e. what really happened: a PERFECT weather forecast.
Real forecasts are worse, so scores obtained this way are slightly optimistic. For a strict
evaluation, use archived forecasts instead (Open-Meteo Historical Forecast API). predict.py
uses the real Forecast API for genuine forecasts beyond the data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FOR_FEATURE = {                        # feature -> raw Open-Meteo variables it needs
    "wind_u": ("wind_speed_10m", "wind_direction_10m"),
    "wind_v": ("wind_speed_10m", "wind_direction_10m"),
    "temperature": ("temperature_2m",),
    "humidity": ("relative_humidity_2m",),
    "precip": ("precipitation",),
    "pressure": ("surface_pressure",),
    "blh": ("boundary_layer_height",),
}
WIND = ("wind_u", "wind_v")
RAW_DIAGNOSTICS = ("wind_u", "wind_v", "precipitation_mm")   # kept in physical units


# --------------------------------------------------------------------------- #
# Feature engineering (shared with predict.py)
# --------------------------------------------------------------------------- #
def engineer_features(raw: dict[str, np.ndarray],
                      only: list[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """raw: Open-Meteo variable -> array of any shape S (e.g. (T, N)).
    Returns (feature names, features (*S, F) in physical units, diagnostics (*S, 3))."""
    shape = next(iter(raw.values())).shape
    nan = np.full(shape, np.nan, dtype=np.float64)
    get = lambda k: np.asarray(raw.get(k, nan), dtype=np.float64)   # noqa: E731

    speed, direction = get("wind_speed_10m"), np.deg2rad(get("wind_direction_10m"))
    u, v = -speed * np.sin(direction), -speed * np.cos(direction)
    precip = np.clip(get("precipitation"), 0, None)
    built = {
        "wind_u": u, "wind_v": v,
        "temperature": get("temperature_2m"),
        "humidity": get("relative_humidity_2m"),
        "precip": np.log1p(precip),
        "pressure": get("surface_pressure"),
        "blh": np.log(np.clip(get("boundary_layer_height"), 10, None)),
    }
    names = only or [f for f, need in RAW_FOR_FEATURE.items() if all(r in raw for r in need)]
    feats = np.stack([built[n] for n in names], axis=-1)
    diag = np.stack([u, v, precip], axis=-1)
    return names, feats.astype(np.float32), diag.astype(np.float32)


def fit_normalization(feats: np.ndarray, names: list[str]) -> dict:
    """feats (T, N, F) of the TRAINING period -> {feature: [mean, std]}."""
    stats = {}
    for i, n in enumerate(names):
        x = feats[..., i]
        x = x[np.isfinite(x)]
        stats[n] = [float(x.mean()), float(x.std() + 1e-6)]
    present = [n for n in WIND if n in names]
    if present:                                  # u, v: one shared scale, no centering
        idx = [names.index(n) for n in present]
        w = feats[..., idx]
        rms = float(np.sqrt(np.nanmean(w ** 2)) + 1e-6)
        for n in present:
            stats[n] = [0.0, rms]
    return stats


def normalize(feats: np.ndarray, names: list[str], stats: dict) -> np.ndarray:
    """Z-score with the given statistics; missing values become 0 (= the training average)."""
    mean = np.array([stats[n][0] for n in names], dtype=np.float32)
    std = np.array([stats[n][1] for n in names], dtype=np.float32)
    z = (feats - mean) / std
    return np.nan_to_num(z, nan=0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Align and normalize weather features.")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--weather", type=Path, default=Path("data/raw/weather"))
    p.add_argument("--max-gap", type=int, default=6, help="interpolate gaps up to this many hours")
    p.add_argument("--min-coverage", type=float, default=0.9,
                   help="drop a feature available for less than this fraction of training hours")
    args = p.parse_args()

    from dataset import find_processed_dir
    proc = find_processed_dir(args.processed)
    meta = json.loads((proc / "meta.json").read_text())
    stations = meta["stations"]
    ts = np.load(proc / "dataset.npz")["timestamps"]
    unit = "ns" if ts.max() > 1e17 else "us"
    times = pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, utc=True)).as_unit("ns")
    T, N = len(times), len(stations)
    print(f"PM2.5 data: {N} stations, {T:,} hours ({times[0]} -> {times[-1]} UTC)")

    st = pd.read_csv(args.weather / "stations.csv")
    missing = sorted(set(stations) - set(st["station"]))
    if missing:
        raise SystemExit(f"No weather cell for {missing[:10]}. Re-run download_weather.py.")
    st = st.set_index("station").loc[stations].reset_index()
    cells = pd.read_parquet(args.weather / "weather_cells.parquet")
    cells["time"] = pd.to_datetime(cells["time"], utc=True).dt.as_unit("ns")
    if cells["time"].min().year < 1990:
        raise SystemExit("weather_cells.parquet has wrong timestamps (1970). Delete "
                         "data/raw/weather/cache and re-run download_weather.py.")
    raw_vars = [c for c in cells.columns if c not in ("time", "cell", "source")]
    print(f"Weather: {cells['cell'].nunique()} cells, variables {raw_vars}")

    # (T, n_cells) per variable on the PM2.5 timeline, short gaps interpolated
    raw: dict[str, np.ndarray] = {}
    cell_ids = st["cell"].tolist()
    for v in raw_vars:
        wide = cells.pivot_table(index="time", columns="cell", values=v, aggfunc="first")
        wide = wide.reindex(times)
        if args.max_gap > 0:
            wide = wide.interpolate(limit=args.max_gap, limit_area="inside")
        wide = wide.reindex(columns=sorted(set(cell_ids)))
        raw[v] = wide[cell_ids].to_numpy()       # (T, N): each station takes its cell
    names, feats, diag = engineer_features(raw)

    # Coverage on the training period; drop features that are mostly missing
    a, z = meta["splits"]["train"]
    keep, coverage = [], {}
    for i, n in enumerate(names):
        coverage[n] = float(np.isfinite(feats[a:z, :, i]).mean())
        if coverage[n] >= args.min_coverage:
            keep.append(i)
        else:
            print(f"  ! Dropping '{n}': only {coverage[n]:.0%} of training hours available")
    names = [names[i] for i in keep]
    feats = feats[..., keep]
    if not names:
        raise SystemExit("No usable weather feature.")

    stats = fit_normalization(feats[a:z], names)
    feats_n = normalize(feats, names, stats)
    missing_frac = {n: float(1 - np.isfinite(feats[..., i]).mean()) for i, n in enumerate(names)}

    np.savez_compressed(proc / "weather.npz", features=feats_n, raw=diag)
    sources = cells.groupby("source")["time"].agg(["min", "max"]).astype(str).to_dict("index")
    wmeta = {
        "features": names, "normalization": stats, "raw_diagnostics": list(RAW_DIAGNOSTICS),
        "stations": stations, "n_times": T,
        "first_timestamp": int(ts[0]), "last_timestamp": int(ts[-1]),
        "missing_fraction": missing_frac, "train_coverage": coverage,
        "sources": sources, "max_gap_interpolated_h": args.max_gap,
        "station_coords": st[["station", "lat", "lon"]].to_dict("records"),
        "raw_variables": raw_vars,
    }
    (proc / "weather_meta.json").write_text(json.dumps(wmeta, indent=2))
    shutil.copy(args.weather / "stations.csv", proc / "stations.csv")

    print(f"\n{'feature':<12}{'mean':>10}{'std':>9}{'missing':>9}   (physical units)")
    for i, n in enumerate(names):
        print(f"{n:<12}{stats[n][0]:>10.2f}{stats[n][1]:>9.2f}{missing_frac[n]:>9.2%}")
    print(f"\nSaved {proc / 'weather.npz'} ({T:,} x {N} x {len(names)}), weather_meta.json, "
          "stations.csv")
    print("Train with weather (the default when weather.npz exists):  python train.py")


if __name__ == "__main__":
    main()

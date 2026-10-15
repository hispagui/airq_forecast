"""
prepare_weather.py
==================
Turn the ERA5 files (download_weather.py) into weather features for every PM2.5 station and
hour, saved next to the PM2.5 data so that train.py uses them automatically.

    python prepare_weather.py
    python prepare_weather.py --max-gap 6        # interpolate gaps up to 6 h (default)

What it does
------------
1. Reads every monthly file data/raw/era5/era5_YYYY_MM.nc.
2. Interpolates each variable from the 0.25 deg grid to the exact station location
   (bilinear: weighted average of the 4 surrounding grid points).
3. Puts everything on the PM2.5 timeline (same UTC hours, same station order).
4. Builds the features below and normalizes them with TRAINING-year statistics only.

Features (one value per hour and station)
-----------------------------------------
    wind_u, wind_v   wind towards the east / north (m/s). A vector instead of an angle: no
                     jump between 359 and 0 deg, and it can be rotated (Phase-2 GNN).
    temperature      2 m temperature (degC)
    humidity         2 m relative humidity (%), computed from temperature and dew point
    precip           log(1 + precipitation in mm over the preceding hour)    (skewed -> log)
    pressure         surface pressure (hPa)
    blh              log(boundary layer height in m): thickness of the layer in which
                     pollution mixes. Low BLH = pollution trapped near the ground.
    radiation        incoming solar radiation (W/m2), mean over the preceding hour
Only features whose variables were downloaded are built.

wind_u and wind_v share one scale and are NOT centered, so the direction is preserved.

Outputs (in data/processed/, next to the PM2.5 data)
    weather.npz         features (T, N, F) normalized, 0 where missing
                        raw (T, N, 3) = wind_u, wind_v (m/s), precipitation (mm), NaN if missing
                        valid (T, N) True where all weather features are real values
    weather_meta.json   feature names, normalization statistics, coverage
    stations.csv        station coordinates and metadata (static node features for Phase 2)

Note on the "future" weather
----------------------------
During training and evaluation, the model receives the weather of the 24 hours it predicts.
ERA5 is what really happened, i.e. a PERFECT weather forecast, which real forecasts are not.
Scores with weather are therefore slightly optimistic: say so in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Common physical variables (any weather source is converted to these first)
#   u10, v10 m/s | t2m degC | rh % | sp hPa | tp mm per hour | blh m | ssrd W/m2
FEATURE_NEEDS = {
    "wind_u": ("u10",), "wind_v": ("v10",), "temperature": ("t2m",), "humidity": ("rh",),
    "precip": ("tp",), "pressure": ("sp",), "blh": ("blh",), "radiation": ("ssrd",),
}
WIND = ("wind_u", "wind_v")
RAW_DIAGNOSTICS = ("wind_u", "wind_v", "precipitation_mm")


# --------------------------------------------------------------------------- #
# Units: source -> common physical variables
# --------------------------------------------------------------------------- #
def relative_humidity(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature and dew point (degC), Magnus formula."""
    es = lambda t: 6.112 * np.exp(17.62 * t / (243.12 + t))   # noqa: E731
    return np.clip(100 * es(td_c) / es(t_c), 0, 100)


def from_era5(v: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ERA5 short names and units -> common variables."""
    out = {}
    if "u10" in v and "v10" in v:
        out["u10"], out["v10"] = v["u10"], v["v10"]
    if "t2m" in v:
        out["t2m"] = v["t2m"] - 273.15
        if "d2m" in v:
            out["rh"] = relative_humidity(out["t2m"], v["d2m"] - 273.15)
    if "sp" in v:
        out["sp"] = v["sp"] / 100.0
    if "tp" in v:
        out["tp"] = np.clip(v["tp"] * 1000.0, 0, None)            # m -> mm
    if "blh" in v:
        out["blh"] = v["blh"]
    if "ssrd" in v:
        out["ssrd"] = np.clip(v["ssrd"] / 3600.0, 0, None)        # J/m2 per hour -> W/m2
    return out


# --------------------------------------------------------------------------- #
# Feature engineering and normalization (shared with predict.py)
# --------------------------------------------------------------------------- #
def engineer_features(common: dict[str, np.ndarray], only: list[str] | None = None):
    """Common variables -> (names, features (*S, F), diagnostics (*S, 3)), physical units."""
    shape = next(iter(common.values())).shape
    get = lambda k: np.asarray(common.get(k, np.full(shape, np.nan)), dtype=np.float64)  # noqa
    built = {
        "wind_u": get("u10"), "wind_v": get("v10"),
        "temperature": get("t2m"), "humidity": get("rh"),
        "precip": np.log1p(np.clip(get("tp"), 0, None)),
        "pressure": get("sp"),
        "blh": np.log(np.clip(get("blh"), 10, None)),
        "radiation": get("ssrd"),
    }
    names = only or [f for f, need in FEATURE_NEEDS.items() if all(k in common for k in need)]
    feats = np.stack([built[n] for n in names], axis=-1).astype(np.float32)
    diag = np.stack([get("u10"), get("v10"), np.clip(get("tp"), 0, None)], -1).astype(np.float32)
    return names, feats, diag


def fit_normalization(feats: np.ndarray, names: list[str]) -> dict:
    """feats (T, N, F) of the TRAINING period -> {feature: [mean, std]}."""
    stats = {}
    for i, n in enumerate(names):
        x = feats[..., i]
        x = x[np.isfinite(x)]
        stats[n] = [float(x.mean()), float(x.std() + 1e-6)]
    present = [n for n in WIND if n in names]
    if present:                                  # u, v: one shared scale, no centering
        w = feats[..., [names.index(n) for n in present]]
        rms = float(np.sqrt(np.nanmean(w ** 2)) + 1e-6)
        for n in present:
            stats[n] = [0.0, rms]
    return stats


def normalize(feats: np.ndarray, names: list[str], stats: dict) -> np.ndarray:
    """Z-score with the given statistics; missing values become 0 (= the training average)."""
    mean = np.array([stats[n][0] for n in names], dtype=np.float32)
    std = np.array([stats[n][1] for n in names], dtype=np.float32)
    return np.nan_to_num((feats - mean) / std, nan=0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reading ERA5 NetCDF files
# --------------------------------------------------------------------------- #
def _bilinear(grid: np.ndarray, x: np.ndarray):
    """Indices of the two grid points around each x, and the weight of the second one."""
    order = np.argsort(grid)
    g = grid[order]
    i = np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2)
    w = np.clip((x - g[i]) / (g[i + 1] - g[i]), 0.0, 1.0)
    return order[i], order[i + 1], w


def read_era5_month(path: Path, lats: np.ndarray, lons: np.ndarray):
    """One ERA5 file -> (UTC times, {short name: (T, N) values at the stations})."""
    import xarray as xr
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    if "expver" in ds.dims:                   # older format: final ERA5 and preliminary ERA5T
        ds = ds.max("expver", skipna=True)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    glat, glon = ds[lat_name].values, ds[lon_name].values
    a0, a1, wa = _bilinear(glat, lats)
    o0, o1, wo = _bilinear(glon, lons)

    times = pd.DatetimeIndex(pd.to_datetime(ds[tdim].values, utc=True)).as_unit("ns")
    out = {}
    for name, da in ds.data_vars.items():
        if tdim not in da.dims or lat_name not in da.dims:
            continue
        arr = da.transpose(tdim, lat_name, lon_name).values.astype(np.float64)  # (T, ny, nx)
        out[name] = ((1 - wa) * (1 - wo) * arr[:, a0, o0] + (1 - wa) * wo * arr[:, a0, o1]
                     + wa * (1 - wo) * arr[:, a1, o0] + wa * wo * arr[:, a1, o1])
    ds.close()
    return times, out


def read_era5(folder: Path, lats: np.ndarray, lons: np.ndarray) -> dict[str, pd.DataFrame]:
    """All monthly files -> {short name: DataFrame (time x station index)}."""
    files = sorted(folder.glob("era5_*.nc"),
                   key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.stem)[:2]))
    if not files:
        raise SystemExit(f"No ERA5 files in {folder}. Run download_weather.py first.")
    parts: dict[str, list[pd.DataFrame]] = {}
    for i, f in enumerate(files):
        times, vals = read_era5_month(f, lats, lons)
        for name, arr in vals.items():
            parts.setdefault(name, []).append(pd.DataFrame(arr, index=times))
        if (i + 1) % 12 == 0 or i + 1 == len(files):
            print(f"  read {i + 1}/{len(files)} files")
    out = {}
    for name, frames in parts.items():
        df = pd.concat(frames)
        out[name] = df[~df.index.duplicated(keep="last")].sort_index()
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Build weather features from ERA5 for the stations.")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--era5", type=Path, default=Path("data/raw/era5"))
    p.add_argument("--max-gap", type=int, default=6, help="interpolate gaps up to this many hours")
    p.add_argument("--min-coverage", type=float, default=0.9,
                   help="drop a feature available for less than this fraction of training hours")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from features.dataset import find_processed_dir
    proc = find_processed_dir(args.processed)
    meta = json.loads((proc / "meta.json").read_text())
    stations = meta["stations"]
    ts = np.load(proc / "dataset.npz")["timestamps"]
    unit = "ns" if ts.max() > 1e17 else "us"
    times = pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, utc=True)).as_unit("ns")
    T, N = len(times), len(stations)
    print(f"PM2.5 data: {N} stations, {T:,} hours ({times[0]} -> {times[-1]} UTC)")

    st = pd.read_csv(args.era5 / "stations.csv")
    missing = sorted(set(stations) - set(st["station"]))
    if missing:
        raise SystemExit(f"No coordinates for {missing[:10]}. Re-run download_weather.py.")
    st = st.set_index("station").loc[stations].reset_index()

    print("Reading ERA5 and interpolating to the stations...")
    series = read_era5(args.era5, st["lat"].to_numpy(), st["lon"].to_numpy())
    print(f"ERA5 variables found: {sorted(series)}")
    era5_end = min(df.index.max() for df in series.values())

    raw = {}
    for name, df in series.items():           # PM2.5 timeline, short gaps interpolated
        df = df.reindex(times)
        if args.max_gap > 0:
            df = df.interpolate(limit=args.max_gap, limit_area="inside")
        raw[name] = df.to_numpy()             # (T, N)
    names, feats, diag = engineer_features(from_era5(raw))

    a, z = meta["splits"]["train"]
    keep, coverage = [], {}
    for i, n in enumerate(names):
        coverage[n] = float(np.isfinite(feats[a:z, :, i]).mean())
        if coverage[n] >= args.min_coverage:
            keep.append(i)
        else:
            print(f"  ! Dropping '{n}': only {coverage[n]:.0%} of training hours available")
    names, feats = [names[i] for i in keep], feats[..., keep]
    if not names:
        raise SystemExit("No usable weather feature.")

    valid = np.isfinite(feats).all(axis=-1)                       # (T, N)
    stats = fit_normalization(feats[a:z], names)
    feats_n = normalize(feats, names, stats)
    missing_frac = {n: float(1 - np.isfinite(feats[..., i]).mean()) for i, n in enumerate(names)}

    np.savez_compressed(proc / "weather.npz", features=feats_n, raw=diag, valid=valid)
    wmeta = {
        "source": "ERA5 (Copernicus CDS), bilinear interpolation to the stations",
        "features": names, "normalization": stats, "raw_diagnostics": list(RAW_DIAGNOSTICS),
        "stations": stations, "n_times": T,
        "first_timestamp": int(ts[0]), "last_timestamp": int(ts[-1]),
        "era5_last_hour": str(era5_end), "missing_fraction": missing_frac,
        "train_coverage": coverage, "max_gap_interpolated_h": args.max_gap,
        "station_coords": st[["station", "lat", "lon"]].to_dict("records"),
    }
    (proc / "weather_meta.json").write_text(json.dumps(wmeta, indent=2))
    shutil.copy(args.era5 / "stations.csv", proc / "stations.csv")

    print(f"\n{'feature':<12}{'mean':>10}{'std':>9}{'missing':>9}   (physical units)")
    for i, n in enumerate(names):
        print(f"{n:<12}{stats[n][0]:>10.2f}{stats[n][1]:>9.2f}{missing_frac[n]:>9.2%}")
    if era5_end < times[-1]:
        hours = int((times[-1] - era5_end) / pd.Timedelta(hours=1))
        print(f"\nNote: ERA5 stops at {era5_end}, {hours} h before the end of the PM2.5 data. "
              "Those hours get the training average; predict.py fetches a weather forecast "
              "for them when you forecast from the end of the data.")
    print(f"\nSaved {proc / 'weather.npz'} ({T:,} hours x {N} stations x {len(names)} features)")
    print("Next: python train.py   (weather is used automatically)")


if __name__ == "__main__":
    main()

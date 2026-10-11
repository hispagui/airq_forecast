"""
preprocess.py

Turn the raw EEA download into a clean, model-ready dataset.

Pipeline
--------
 1. Load the combined raw table produced by download_eea.py
 2. Quality filter: keep valid flags only, drop physically impossible values
 3. Convert timestamps to UTC (EEA reports in a fixed UTC+1 offset by default; see --source-tz)
 4. Map sampling points -> stations, attach coordinates from the metadata file
 5. Pivot to a complete hourly grid:  time x station
 6. Split by YEAR into train / val / test (chronological, no shuffling!)
 7. Select stations with enough data coverage in every split
 8. Fill only SHORT gaps (<= --max-gap hours) by interpolation; longer gaps stay missing
    and are tracked by a mask (the loss will ignore them later)
 9. Transform: log1p (PM2.5 is very right-skewed), then z-score using TRAIN statistics only (NORMALIZE)
10. Build calendar features in local Brussels time (traffic and heating follow local clocks)
11. Save everything to data/processed/

Why one file with the full timeline instead of three separate files?
A validation sample at 1 January needs the last 72 h of December as input history.
Keeping one continuous timeline + split boundaries makes that trivial and leak-free:
the *targets* of a split never leave that split, only the input history can look back.

Usage
-----
    python preprocess.py                                   # defaults below
    python preprocess.py --val-year 2024 --test-year 2025 --train-start 2016
    python preprocess.py --norm station                    # per-station z-score

Outputs (data/processed/)
-------
    dataset.npz          arrays: values (T,N) normalized, mask (T,N), raw (T,N) in ug/m3,
                         time_features (T,F), timestamps (T,) int64 ns UTC
    stations.csv         one row per kept station: id, lat, lon, altitude, type, area, coverage
    meta.json            split boundaries, normalization stats, feature names, settings
    pm25_hourly_clean.parquet   human-readable wide table (UTC index, ug/m3), for plotting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VALUE_MIN, VALUE_MAX = 0.0, 1000.0  # ug/m3; above 1000 is a sensor error, not Belgian air


# --------------------------------------------------------------------------- #
# 1-3. Load, clean, fix time
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} raw rows, {df['Samplingpoint'].nunique()} sampling points")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    # Validity: 1 = valid, 2/3 = valid but below detection limit; <= 0 means invalid
    if "Validity" in df.columns:
        df = df[pd.to_numeric(df["Validity"], errors="coerce") >= 1]
    if "Unit" in df.columns:
        units = df["Unit"].astype(str).str.lower().unique()
        if any("ug" not in u and "µg" not in u for u in units):
            print(f"  ! Unexpected units found: {units}, check them before continuing")
    # Some files store Value as text/decimal: force float, otherwise interpolation fails
    df = df.assign(Value=pd.to_numeric(df["Value"], errors="coerce")).dropna(subset=["Value"])
    df = df[(df["Value"] >= VALUE_MIN) & (df["Value"] <= VALUE_MAX)]
    print(f"Quality filter kept {len(df):,} / {n0:,} rows ({len(df) / max(n0, 1):.1%})")
    return df


def check_dst(start: pd.Series) -> None:
    """If timestamps were local time WITH daylight saving, the 02:00 hour on the last
    Sunday of March would never exist. We check that, so a timezone mistake can't slip by."""
    s = pd.to_datetime(start)
    if s.dt.tz is not None:
        return
    hits, expected = 0, 0
    for year in sorted(s.dt.year.unique()):
        march_end = pd.Timestamp(year=year, month=3, day=31)
        last_sunday = march_end - pd.Timedelta(days=(march_end.weekday() + 1) % 7)
        at_2am = s == last_sunday + pd.Timedelta(hours=2)
        hits += int(at_2am.sum())
        expected += int((s == last_sunday + pd.Timedelta(hours=1)).sum())
    if expected > 0 and hits < 0.2 * expected:
        print("  ! Almost no data at 02:00 on DST-switch days: timestamps look like local "
              "time with daylight saving. Re-run with --source-tz Europe/Brussels")
    else:
        print("  DST check OK: timestamps look like a fixed offset (no DST gaps)")


def to_utc(df: pd.DataFrame, source_tz: str) -> pd.DataFrame:
    start = pd.to_datetime(df["Start"])
    if start.dt.tz is None:
        check_dst(start)
        start = start.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    df = df.assign(time=start.dt.tz_convert("UTC")).dropna(subset=["time"])
    return df


# --------------------------------------------------------------------------- #
# 4. Stations
# --------------------------------------------------------------------------- #
def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


def _strip_country(sp: pd.Series) -> pd.Series:
    """'BE/SPO-BETR001_...' -> 'SPO-BETR001_...' (the metadata file has no prefix)."""
    return sp.astype(str).str.replace(r"^[A-Z]{2}/", "", regex=True)


def load_station_table(meta_path: Path, sampling_points: pd.Series) -> pd.DataFrame:
    """Return one row per sampling point with station id + coordinates.
    Falls back to 'sampling point = station, no coordinates' if metadata is unusable."""
    sp = pd.Series(sampling_points.unique(), name="Samplingpoint")
    fallback = pd.DataFrame({"Samplingpoint": sp, "station": _strip_country(sp)})
    if not meta_path.exists():
        print(f"  ! No metadata at {meta_path}; stations will have no coordinates")
        return fallback

    meta = pd.read_csv(meta_path, low_memory=False)
    c_sp = _find_col(meta, "Sampling Point Id", "SamplingPointId", "Samplingpoint",
                     "SamplingPoint", "Sampling Point")
    c_st = _find_col(meta, "Air Quality Station EoI Code", "AirQualityStationEoICode",
                     "Station EoI Code", "Air Quality Station", "AirQualityStation")
    if c_sp is None or c_st is None:
        print(f"  ! Metadata columns not recognised: {list(meta.columns)[:15]}...")
        return fallback

    cols = {
        "lat": _find_col(meta, "Latitude"),
        "lon": _find_col(meta, "Longitude"),
        "altitude": _find_col(meta, "Altitude"),
        "station_type": _find_col(meta, "Air Quality Station Type", "AirQualityStationType"),
        "station_area": _find_col(meta, "Air Quality Station Area", "AirQualityStationArea"),
        "station_name": _find_col(meta, "Air Quality Station Name", "AirQualityStationName"),
    }
    keep = meta[[c_sp, c_st] + [c for c in cols.values() if c]].copy()
    keep = keep.rename(columns={c_sp: "sp_key", c_st: "station",
                                **{v: k for k, v in cols.items() if v}})
    keep["sp_key"] = _strip_country(keep["sp_key"])
    keep = keep.drop_duplicates("sp_key")

    table = fallback.assign(sp_key=_strip_country(sp)).drop(columns="station")
    table = table.merge(keep, on="sp_key", how="left")
    missing = table["station"].isna()
    if missing.any():
        print(f"  ! {missing.sum()} sampling points not found in metadata (kept, no coords)")
        print(f"    e.g. data id {table.loc[missing, 'sp_key'].iloc[0]!r} vs "
              f"metadata ids {keep['sp_key'].head(3).tolist()}")
        table.loc[missing, "station"] = table.loc[missing, "sp_key"]
    print(f"Matched {(~missing).sum()} / {len(table)} sampling points to stations")
    return table.drop(columns="sp_key")


# --------------------------------------------------------------------------- #
# 5. Wide hourly grid
# --------------------------------------------------------------------------- #
def to_wide(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(stations[["Samplingpoint", "station"]], on="Samplingpoint", how="left")
    # A station can have several PM2.5 instruments: average them per hour
    wide = df.pivot_table(index="time", columns="station", values="Value",
                          aggfunc="mean").astype("float64")
    full_index = pd.date_range(wide.index.min().floor("h"), wide.index.max().ceil("h"),
                               freq="h", tz="UTC")
    wide = wide.reindex(full_index)
    wide.index.name = "time_utc"
    print(f"Hourly grid: {wide.shape[0]:,} hours x {wide.shape[1]} stations, "
          f"{wide.notna().mean().mean():.1%} filled")
    return wide


# --------------------------------------------------------------------------- #
# 6-7. Splits and station selection
# --------------------------------------------------------------------------- #
def split_bounds(index: pd.DatetimeIndex, train_start: int, val_year: int, test_year: int):
    def pos(ts):  # first index position >= ts
        return int(index.searchsorted(pd.Timestamp(ts, tz="UTC")))
    b = {
        "train": (pos(f"{train_start}-01-01"), pos(f"{val_year}-01-01")),
        "val": (pos(f"{val_year}-01-01"), pos(f"{test_year}-01-01")),
        "test": (pos(f"{test_year}-01-01"), pos(f"{test_year + 1}-01-01")),
    }
    for name, (a, z) in b.items():
        if z - a <= 0:
            raise ValueError(f"Split '{name}' is empty. Data covers "
                             f"{index.min():%Y-%m-%d} -> {index.max():%Y-%m-%d}; adjust the years.")
    return b


def select_stations(wide: pd.DataFrame, bounds: dict, min_cov: dict) -> pd.DataFrame:
    cov = pd.DataFrame({name: wide.iloc[a:z].notna().mean() for name, (a, z) in bounds.items()})
    ok = (cov["train"] >= min_cov["train"]) & (cov["val"] >= min_cov["eval"]) & \
         (cov["test"] >= min_cov["eval"])
    print("\nCoverage per split (fraction of hours with data):")
    print(cov.describe().loc[["mean", "min", "50%", "max"]].round(2).to_string())
    print(f"Keeping {ok.sum()} / {len(ok)} stations "
          f"(train >= {min_cov['train']:.0%}, val/test >= {min_cov['eval']:.0%})")
    dropped = cov.index[~ok].tolist()
    if dropped:
        print(f"  dropped: {dropped}")
    return cov[ok]


# --------------------------------------------------------------------------- #
# 8-10. Gaps, normalization, features
# --------------------------------------------------------------------------- #
def fill_short_gaps(wide: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Linear interpolation only for gaps of at most `max_gap` consecutive hours."""
    filled = wide.copy()
    for col in wide.columns:
        s = wide[col]
        is_na = s.isna()
        run_id = (~is_na).cumsum()
        run_len = is_na.groupby(run_id).transform("sum")
        interp = s.interpolate(method="linear", limit_area="inside")
        short = is_na & (run_len <= max_gap)
        filled.loc[short, col] = interp[short]
    return filled


def normalize(wide: pd.DataFrame, train_slice: slice, mode: str):
    logged = np.log1p(wide)
    train = logged.iloc[train_slice]
    if mode == "global":
        mean, std = float(np.nanmean(train.values)), float(np.nanstd(train.values))
        normed = (logged - mean) / std
        stats = {"mode": "global", "transform": "log1p", "mean": mean, "std": std}
    else:  # per-station
        mean, std = train.mean(), train.std().replace(0, 1.0)
        normed = (logged - mean) / std
        stats = {"mode": "station", "transform": "log1p",
                 "mean": mean.to_dict(), "std": std.to_dict()}
    return normed, stats


def time_features(index: pd.DatetimeIndex, local_tz: str = "Europe/Brussels") -> pd.DataFrame:
    local = index.tz_convert(local_tz)
    hour = local.hour + local.minute / 60
    doy = local.dayofyear - 1
    dow = local.dayofweek
    feats = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(float),
    }, index=index)
    return feats


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess EEA PM2.5 data.")
    p.add_argument("--raw", type=Path, default=None,
                   help="combined raw parquet (default: first *_hourly.parquet in data/raw/eea)")
    p.add_argument("--metadata", type=Path, default=Path("data/raw/eea/stations_metadata.csv"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--source-tz", default="Etc/GMT-1",
                   help="timezone of raw timestamps. 'Etc/GMT-1' means fixed UTC+1 (EEA default)")
    p.add_argument("--train-start", type=int, default=2016)
    p.add_argument("--val-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--min-train-coverage", type=float, default=0.70)
    p.add_argument("--min-eval-coverage", type=float, default=0.50)
    p.add_argument("--max-gap", type=int, default=3, help="max hours to interpolate")
    p.add_argument("--norm", choices=["global", "station"], default="global")
    p.add_argument("--keep-recent", action="store_true",
                   help="also keep the data AFTER the test year (e.g. 2026 so far), so predict.py "
                        "can forecast recent dates. Splits, stations and normalization are unchanged, "
                        "so models already trained stay valid.")
    args = p.parse_args()

    raw_path = args.raw or next(Path("data/raw/eea").glob("*_hourly.parquet"), None)
    if raw_path is None or not raw_path.exists():
        raise FileNotFoundError("Raw parquet not found. Run download_eea.py first or pass --raw.")

    # 1-3
    df = to_utc(clean(load_raw(raw_path)), args.source_tz)
    # 4
    stations = load_station_table(args.metadata, df["Samplingpoint"])
    # 5
    wide = to_wide(df, stations)
    # 6-7
    bounds = split_bounds(wide.index, args.train_start, args.val_year, args.test_year)
    end = len(wide) if args.keep_recent else bounds["test"][1]
    wide = wide.iloc[bounds["train"][0]:end]                          # trim to used period
    bounds = split_bounds(wide.index, args.train_start, args.val_year, args.test_year)
    cov = select_stations(wide, bounds,
                          {"train": args.min_train_coverage, "eval": args.min_eval_coverage})
    if cov.empty:
        raise RuntimeError("No station passes the coverage thresholds; lower them.")
    wide = wide[cov.index]
    # 8
    observed = wide.notna()
    filled = fill_short_gaps(wide, args.max_gap)
    mask = filled.notna()
    print(f"\nObserved: {observed.values.mean():.1%} | after short-gap filling: "
          f"{mask.values.mean():.1%} (the rest stays masked)")
    # 9
    tr = slice(*bounds["train"])
    normed, stats = normalize(filled, tr, args.norm)
    # 10
    feats = time_features(filled.index)

    # 11. Save
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "dataset.npz",
        values=normed.fillna(0.0).to_numpy(np.float32),   # 0 = train mean; masked anyway
        mask=mask.to_numpy(bool),
        raw=filled.to_numpy(np.float32),                  # ug/m3, NaN where missing
        time_features=feats.to_numpy(np.float32),
        timestamps=filled.index.as_unit("ns").asi8,  # always nanoseconds
    )
    st = (stations.drop_duplicates("station").set_index("station")
          .drop(columns="Samplingpoint").reindex(cov.index))
    st = st.join(cov.add_prefix("coverage_"))
    st.index.name = "station"
    st.to_csv(args.out / "stations.csv")
    filled.to_parquet(args.out / "pm25_hourly_clean.parquet")

    meta = {
        "stations": cov.index.tolist(),
        "splits": {k: list(v) for k, v in bounds.items()},
        "split_years": {"train": [args.train_start, args.val_year - 1],
                        "val": args.val_year, "test": args.test_year},
        "normalization": stats,
        "time_features": feats.columns.tolist(),
        "start_utc": str(filled.index[0]), "end_utc": str(filled.index[-1]),
        "settings": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to {args.out}/")
    if args.keep_recent and len(filled) > bounds["test"][1]:
        a = bounds["test"][1]
        print(f"  recent (not used for training/eval): {filled.index[a]:%Y-%m-%d} -> "
              f"{filled.index[-1]:%Y-%m-%d %H:%M} UTC, available to predict.py")
    for name, (a, z) in bounds.items():
        print(f"  {name:>5}: {filled.index[a]:%Y-%m-%d} -> {filled.index[z - 1]:%Y-%m-%d} "
              f"({z - a:,} hours, {mask.iloc[a:z].values.mean():.1%} valid)")
    print(f"  stations: {len(cov)} | features per hour: {feats.shape[1]}")
    print("  PM2.5 (ug/m3) train stats: "
          f"mean {np.nanmean(filled.iloc[tr].values):.1f}, "
          f"p95 {np.nanpercentile(filled.iloc[tr].values, 95):.1f}")


if __name__ == "__main__":
    main()
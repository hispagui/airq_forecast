"""
download_weather.py
===================
Download hourly ERA5 weather over the PM2.5 stations from the Copernicus Climate Data Store
(CDS), one file per month.

ERA5 = ECMWF's reanalysis: the best estimate of past weather, every hour, on a 0.25 deg grid
(~25 km), from 1940 to about one week ago.

One-time setup (5 minutes)
--------------------------
1. Log in on https://cds.climate.copernicus.eu and open your profile page: copy your
   "Personal Access Token". (Your password is NEVER used by the API. Never put it in code.)
2. Create the file ~/.cdsapirc  (Windows: C:\\Users\\<you>\\.cdsapirc) containing:
       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-access-token>
3. Accept the licence once: open the dataset page "ERA5 hourly data on single levels from
   1940 to present", tab "Download", scroll to "Terms of use" and click Accept.
4. pip install cdsapi xarray netCDF4

Usage
-----
    python download_weather.py                    # stations + years from data/processed
    python download_weather.py --start 2016-01 --end 2025-12
    python download_weather.py --dry-run          # show the plan, download nothing
    python download_weather.py --workers 2        # requests in the queue at once (default 1)

Requests are queued on the CDS servers: each month can take from ~1 minute to much longer
when the queue is busy. The CDS also limits how many requests one user may have waiting;
when it answers "temporarily limited", the script waits and tries again by itself. Ten years = ~120 requests, often a few hours in total. Every month is
saved as soon as it arrives, so you can stop the script and run it again: it resumes.
Track your requests at https://cds.climate.copernicus.eu/requests

Variables (ERA5 name -> short name in the files)
------------------------------------------------
    10m_u_component_of_wind              u10    wind towards the east (m/s)
    10m_v_component_of_wind              v10    wind towards the north (m/s)
    2m_temperature                       t2m    K
    2m_dewpoint_temperature              d2m    K   (-> relative humidity)
    surface_pressure                     sp     Pa
    total_precipitation                  tp     m, over the preceding hour
    boundary_layer_height                blh    m   (low = pollution trapped near the ground)
    surface_solar_radiation_downwards    ssrd   J/m2 over the preceding hour

Outputs: data/raw/era5/
    era5_YYYY_MM.nc        one file per month, the whole box around the stations
    stations.csv           station coordinates (+ altitude, type, area when known)

Then run:  python prepare_weather.py
Citation: Hersbach et al. (2023), ERA5 hourly data on single levels from 1940 to present,
Copernicus Climate Change Service (C3S) Climate Data Store, doi:10.24381/cds.adbb2d47
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature",
    "surface_pressure", "total_precipitation",
    "boundary_layer_height", "surface_solar_radiation_downwards",
]
ERA5_DELAY_DAYS = 6          # ERA5 (preliminary "ERA5T") appears ~5 days after real time
MARGIN_DEG = 0.5             # box around the stations, so interpolation has neighbours
GRID = 0.25


# --------------------------------------------------------------------------- #
# Station coordinates
# --------------------------------------------------------------------------- #
def _find_col(df: pd.DataFrame, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
    for c in df.columns:
        low = c.lower()
        if all(n in low for n in needles) and not any(x in low for x in exclude):
            return c
    return None


def resolve_coordinates(stations: list[str], metadata_csv: Path,
                        coords_csv: Path | None) -> pd.DataFrame:
    """station -> lat, lon (+ altitude, station_type, area_type when the EEA file has them)."""
    if coords_csv:
        df = pd.read_csv(coords_csv)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        missing = sorted(set(stations) - set(df["station"]))
        if missing:
            raise SystemExit(f"{coords_csv} lacks coordinates for {len(missing)} stations: {missing[:10]}")
        return df.set_index("station").loc[stations].reset_index()

    if not metadata_csv.exists():
        raise SystemExit(f"No station metadata at {metadata_csv}. Run `python download_eea.py "
                         "--metadata-only`, or pass --coords file.csv (columns station, lat, lon).")
    meta = pd.read_csv(metadata_csv, low_memory=False)
    lat_c, lon_c = _find_col(meta, "latitude"), _find_col(meta, "longitude")
    if lat_c is None or lon_c is None:
        raise SystemExit(f"No latitude/longitude column in {metadata_csv}: {list(meta.columns)}")
    alt_c = _find_col(meta, "altitude")
    type_c = _find_col(meta, "station", "type") or _find_col(meta, "type", exclude=("pollutant",))
    area_c = _find_col(meta, "area")
    text_cols = [c for c in meta.columns if not pd.api.types.is_numeric_dtype(meta[c])]
    upper = {c: meta[c].astype(str).str.strip().str.upper() for c in text_cols}

    rows, unmatched = [], []
    for s in stations:
        code = s.strip().upper()
        hit = pd.Series(False, index=meta.index)
        for c in text_cols:                         # 1) exact match (EoI code, e.g. BETR801)
            hit |= upper[c] == code
        if not hit.any():                           # 2) inside a sampling-point / station id
            for c in text_cols:
                if "sampl" in c.lower() or "station" in c.lower():
                    hit |= upper[c].str.contains(code, regex=False)
        sel = meta[hit]
        sel = sel[pd.to_numeric(sel[lat_c], errors="coerce").notna()]
        if sel.empty:
            unmatched.append(s)
            continue
        row = {"station": s, "lat": float(pd.to_numeric(sel[lat_c]).median()),
               "lon": float(pd.to_numeric(sel[lon_c]).median())}
        if alt_c:
            row["altitude"] = float(pd.to_numeric(sel[alt_c], errors="coerce").median())
        if type_c:
            row["station_type"] = sel[type_c].astype(str).mode().iloc[0]
        if area_c:
            row["area_type"] = sel[area_c].astype(str).mode().iloc[0]
        rows.append(row)
    if unmatched:
        raise SystemExit(f"No coordinates found for {len(unmatched)} station(s): {unmatched[:10]}. "
                         "Pass --coords file.csv (columns station, lat, lon) for those.")
    return pd.DataFrame(rows)


def bounding_box(coords: pd.DataFrame, margin: float = MARGIN_DEG) -> list[float]:
    """[North, West, South, East] on the 0.25 deg grid, with a margin around the stations."""
    snap_up = lambda x: float(np.ceil(x / GRID) * GRID)      # noqa: E731
    snap_dn = lambda x: float(np.floor(x / GRID) * GRID)     # noqa: E731
    return [snap_up(coords["lat"].max() + margin), snap_dn(coords["lon"].min() - margin),
            snap_dn(coords["lat"].min() - margin), snap_up(coords["lon"].max() + margin)]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def month_list(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def days_in(year: int, month: int, last_day: date) -> list[int]:
    first = date(year, month, 1)
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    end = min(nxt - timedelta(days=1), last_day)
    return list(range(1, end.day + 1)) if end >= first else []


def month_file(out: Path, year: int, month: int, n_days: int, complete: bool) -> Path:
    # An incomplete (current) month gets the number of days in its name, so a later run
    # downloads it again once more days are available.
    tag = "" if complete else f"_partial{n_days:02d}"
    return out / f"era5_{year}_{month:02d}{tag}.nc"


def build_request(year: int, month: int, days: list[int], area: list[float],
                  variables: list[str]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,                                   # North, West, South, East
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def normalize_download(path: Path) -> None:
    """The CDS sometimes sends a zip of several NetCDF files (instantaneous variables and
    accumulated ones separately), even when asked for 'unarchived'. Merge them into one file."""
    if not zipfile.is_zipfile(path):
        return
    import xarray as xr
    tmp = path.with_suffix(".parts")
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp)
    parts = sorted(tmp.rglob("*.nc"))
    ds = xr.merge([xr.open_dataset(p).load() for p in parts], compat="override")
    path.unlink()
    ds.to_netcdf(path)
    for p in parts:
        p.unlink()
    for d in sorted(tmp.rglob("*"), reverse=True):
        d.rmdir()
    tmp.rmdir()


_print_lock = threading.Lock()
_netcdf_lock = threading.Lock()   # the HDF5 library behind NetCDF is not thread-safe


def say(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def download_month(client, year: int, month: int, days: list[int], area, variables,
                   target: Path) -> Path:
    part = target.with_suffix(".download")
    request = build_request(year, month, days, area, variables)
    if part.exists():                     # downloaded by an earlier run, not yet processed
        try:
            with _netcdf_lock:
                normalize_download(part)
            part.rename(target)
            say(f"  saved {target.name} (reused an earlier download)")
            return target
        except Exception as e:
            if "numpy" in str(e).lower() or "np." in str(e):
                raise                     # a library problem, not a bad file: keep the file
            say(f"  ! earlier download of {year}-{month:02d} unreadable ({e}); downloading again")
            part.unlink()
    for attempt in range(1, 61):
        say(f"  requesting {year}-{month:02d} ({len(days)} days)"
            + (f", attempt {attempt}" if attempt > 1 else "") + "...")
        try:
            client.retrieve(DATASET, request, str(part))
            break
        except Exception as e:
            msg = str(e).lower()
            if "temporarily limited" not in msg and "queued requests" not in msg:
                raise
            wait = min(60 * attempt, 600)          # 1, 2, ... up to 10 minutes
            say(f"  ... CDS queue limit reached for {year}-{month:02d}; retrying in "
                f"{wait // 60} min")
            time.sleep(wait)
    else:
        raise RuntimeError("CDS queue still limited after many retries; run again later.")
    with _netcdf_lock:                    # downloads run in parallel, file handling does not
        normalize_download(part)
    part.rename(target)
    # Remove older partial versions of this month
    for old in target.parent.glob(f"era5_{year}_{month:02d}_partial*.nc"):
        if old != target:
            old.unlink()
    say(f"  saved {target.name}")
    return target


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def processed_info(processed: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from features.dataset import find_processed_dir
    d = find_processed_dir(processed)
    meta = json.loads((d / "meta.json").read_text())
    ts = np.load(d / "dataset.npz")["timestamps"]
    unit = "ns" if ts.max() > 1e17 else "us"
    times = pd.to_datetime(ts[[0, -1]], unit=unit, utc=True)
    return meta["stations"], times[0].date(), times[1].date()


def main() -> None:
    p = argparse.ArgumentParser(description="Download ERA5 hourly weather from the CDS.")
    p.add_argument("--processed", default="data/processed",
                   help="processed PM2.5 data (gives the stations and the period)")
    p.add_argument("--metadata", type=Path, default=Path("data/raw/eea/stations_metadata.csv"))
    p.add_argument("--coords", type=Path, default=None,
                   help="optional CSV with columns station, lat, lon")
    p.add_argument("--start", default=None, help="YYYY-MM (default: start of the PM2.5 data)")
    p.add_argument("--end", default=None, help="YYYY-MM (default: end of the PM2.5 data)")
    p.add_argument("--variables", nargs="+", default=VARIABLES, help="ERA5 variable names")
    p.add_argument("--workers", type=int, default=1,
                   help="requests in the CDS queue at once (the CDS limits this per user)")
    p.add_argument("--out", type=Path, default=Path("data/raw/era5"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stations, d0, d1 = processed_info(args.processed)
    start = (datetime.strptime(args.start, "%Y-%m").date() if args.start
             else d0 - timedelta(days=1))
    end = d1 + timedelta(days=1)
    if args.end:
        e = datetime.strptime(args.end, "%Y-%m").date()
        end = (date(e.year + 1, 1, 1) if e.month == 12 else date(e.year, e.month + 1, 1)) \
            - timedelta(days=1)
    last_available = datetime.now(timezone.utc).date() - timedelta(days=ERA5_DELAY_DAYS)
    end = min(end, last_available)

    coords = resolve_coordinates(stations, args.metadata, args.coords)
    area = bounding_box(coords)
    args.out.mkdir(parents=True, exist_ok=True)
    coords.to_csv(args.out / "stations.csv", index=False)

    jobs = []
    for y, m in month_list(start, end):
        days = days_in(y, m, end)
        if not days:
            continue
        complete = len(days) == days_in(y, m, date(9999, 1, 1))[-1]
        target = month_file(args.out, y, m, len(days), complete)
        if not target.exists():
            jobs.append((y, m, days, target))
    n_months = len(month_list(start, end))
    print(f"{len(stations)} stations | {start} -> {end} | box N/W/S/E = {area}")
    print(f"{len(args.variables)} variables | {n_months} months, {n_months - len(jobs)} already "
          f"downloaded, {len(jobs)} to request")
    if args.dry_run or not jobs:
        print("Nothing to do." if not jobs else "(dry run: nothing requested)")
        return

    try:
        import cdsapi
    except ImportError:
        raise SystemExit("pip install cdsapi xarray netCDF4")
    try:
        client = cdsapi.Client(quiet=True, progress=False)
    except Exception as e:
        raise SystemExit(f"Could not start the CDS client ({e}).\nCreate ~/.cdsapirc with your "
                         "personal access token (see the top of this file).")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_month, client, y, m, days, area, args.variables, t):
                   (y, m) for y, m, days, t in jobs}
        for fut in as_completed(futures):
            y, m = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append((y, m))
                msg = str(e)
                if "numpy" in msg.lower() or "np." in msg:
                    msg += ("\n    -> outdated library. Run: pip install -U xarray netCDF4 cftime"
                            "\n       (the downloaded file is kept and reused next time)")
                if "licence" in msg.lower() or "license" in msg.lower():
                    msg += ("\n    -> accept the ERA5 licence on the dataset page "
                            "(tab Download, 'Terms of use').")
                if "401" in msg or "authentication" in msg.lower():
                    msg += "\n    -> check the token in ~/.cdsapirc."
                say(f"  ! {y}-{m:02d} failed: {msg}")
    print(f"\nDone: {len(jobs) - len(failed)} months downloaded, {len(failed)} failed.")
    if failed:
        print("Run the same command again to retry the failed months.")
    else:
        print("Next: python prepare_weather.py")


if __name__ == "__main__":
    main()
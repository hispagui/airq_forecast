"""
download_weather.py
===================
Download hourly weather at every PM2.5 station from Open-Meteo (no API key needed).

Default variables (hourly):
    temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m, wind_direction_10m,
    surface_pressure, boundary_layer_height
Default source: ERA5 reanalysis (0.25 deg, consistent from 1940 to ~a week ago). The last
days, which ERA5 does not have yet, are filled with Open-Meteo's "best_match" archive
(ECMWF IFS 9 km, no delay).

Usage
-----
    pip install requests pandas pyarrow
    python download_weather.py                      # stations + period from data/processed
    python download_weather.py --start 2016-01-01 --end 2026-10-10
    python download_weather.py --dry-run            # show the plan and the API budget only
    python download_weather.py --variables temperature_2m precipitation wind_speed_10m wind_direction_10m

Then run:  python prepare_weather.py     (aligns, engineers and normalizes the features)

Station coordinates
-------------------
Looked up, in order, from
  1. --coords my_coords.csv           (columns: station, lat, lon)
  2. data/raw/eea/stations_metadata.csv   (written by download_eea.py)
and saved, with altitude / station type / area type when available, to
  data/raw/weather/stations.csv       (also useful later as static node features for the GNN)

Free-tier budget (non-commercial)
---------------------------------
Open-Meteo allows 600 calls/min, 5 000/hour and 10 000/day. A request counts as
(locations) x (days / 14) x (variables / 10, at least 1) calls, so 10 years at one location
costs ~260 calls. To stay small the script
  * snaps stations to the ERA5 grid (0.25 deg): stations in the same cell share one download,
  * caches every (cell, year) file, so an interrupted run resumes where it stopped,
  * paces itself under 80 % of each limit, using a ledger shared by successive runs.
If the daily budget runs out it stops cleanly: run the same command again tomorrow.

Outputs: data/raw/weather/
    stations.csv              one row per station: coordinates, ERA5 cell, metadata
    cache/<model>/*.parquet   raw API responses, one file per (cell, period)
    weather_cells.parquet     every cell x hour (UTC), all variables, column 'source'

Licence: Open-Meteo data is CC BY 4.0. Cite Open-Meteo and ERA5 (Hersbach et al. 2023).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DEFAULT_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "boundary_layer_height",
]
ERA5_DELAY_DAYS = 7          # ERA5 is published with ~5 days delay; keep a margin
FREE_LIMITS = ((60, 600), (3600, 5_000), (86_400, 10_000))   # (window in s, calls)
SAFETY = 0.8                 # use at most 80 % of each limit
MAX_WEIGHT_PER_REQUEST = 200.0


class DailyLimitReached(RuntimeError):
    pass


class ApiError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Call budget: counts calls the way Open-Meteo does and waits when needed
# --------------------------------------------------------------------------- #
def call_weight(n_locations: int, n_days: float, n_variables: int) -> float:
    return n_locations * max(1.0, n_days / 14) * max(1.0, n_variables / 10)


class CallBudget:
    """Sliding-window limiter over the free-tier limits. The ledger (time, weight) is saved
    to disk so that two runs in the same hour/day do not exceed the limits together."""

    def __init__(self, ledger_path: Path | None = None, safety: float = SAFETY):
        self.safety, self.path = safety, ledger_path
        self.events: deque[tuple[float, float]] = deque()
        if ledger_path and ledger_path.exists():
            try:
                now = time.time()
                self.events.extend((t, w) for t, w in json.loads(ledger_path.read_text())
                                   if t > now - 86_400)
            except (ValueError, TypeError):
                pass

    def _used(self, now: float, window: float) -> float:
        return sum(w for t, w in self.events if t > now - window)

    def wait(self, weight: float) -> None:
        while True:
            now = time.time()
            while self.events and self.events[0][0] <= now - 86_400:
                self.events.popleft()
            sleep = 0.0
            for window, limit in FREE_LIMITS:
                cap = limit * self.safety
                used = self._used(now, window)
                if used + weight <= cap:
                    continue
                if window == 86_400:
                    raise DailyLimitReached(
                        f"Daily budget reached ({used:.0f} of {cap:.0f} calls used in 24 h).")
                # Wait until enough old calls leave the window
                freed = 0.0
                for t, w in self.events:
                    if t <= now - window:
                        continue
                    freed += w
                    if used - freed + weight <= cap:
                        sleep = max(sleep, t + window - now + 1)
                        break
            if sleep <= 0:
                return
            print(f"  ... pacing: waiting {sleep / 60:.1f} min to stay under the free-tier limits")
            time.sleep(sleep)

    def record(self, weight: float) -> None:
        self.events.append((time.time(), weight))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(list(self.events)))


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
_session = requests.Session()


def api_get(url: str, params: dict, budget: CallBudget | None, weight: float,
            retries: int = 6) -> list[dict]:
    """GET with pacing and retries. Always returns a list (one dict per location)."""
    for attempt in range(retries):
        if budget:
            budget.wait(weight)
        try:
            r = _session.get(url, params=params, timeout=180)
        except requests.RequestException as e:
            wait = 10 * 2 ** attempt
            print(f"  ! {e.__class__.__name__}; retrying in {wait}s")
            time.sleep(wait)
            continue
        if budget:
            budget.record(weight)
        if r.status_code == 200:
            out = r.json()
            return out if isinstance(out, list) else [out]
        try:
            reason = r.json().get("reason", r.text[:300])
        except ValueError:
            reason = r.text[:300]
        if r.status_code == 429:
            low = reason.lower()
            if "daily" in low:
                raise DailyLimitReached(reason)
            wait = 65 if "minut" in low else 15 * 60
            print(f"  ! Rate limited ({reason}); waiting {wait // 60} min")
            time.sleep(wait)
            continue
        if r.status_code == 400:
            raise ApiError(reason)
        wait = 10 * 2 ** attempt
        print(f"  ! HTTP {r.status_code} ({reason}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {retries} attempts.")


def parse_location(loc: dict, variables: list[str]) -> pd.DataFrame:
    """One location of an Open-Meteo JSON response -> DataFrame indexed by UTC hour."""
    hourly = loc.get("hourly", {})
    # nanosecond resolution: second-resolution timestamps do not survive a parquet round-trip
    # with every pandas/pyarrow version
    secs = np.asarray(hourly.get("time", []), dtype="int64")
    times = pd.DatetimeIndex(pd.to_datetime(secs * 1_000_000_000, unit="ns", utc=True))
    df = pd.DataFrame({v: np.asarray(hourly.get(v, [np.nan] * len(times)), dtype="float64")
                       for v in variables}, index=times)
    df.index.name = "time"
    return df


_REJECTED: dict[tuple[str, str], set[str]] = {}   # (url, model) -> variables the API refused


def fetch_with_fallback(url: str, base_params: dict, variables: list[str],
                        budget: CallBudget | None, weight_fn) -> tuple[list[dict], list[str]]:
    """Request `variables`; if the API rejects one (e.g. not available in this model),
    drop it, remember it for the next requests, and try again.
    Returns (response, variables actually obtained)."""
    rejected = _REJECTED.setdefault((url, base_params.get("models", "")), set())
    variables = [v for v in variables if v not in rejected]
    while variables:
        params = {**base_params, "hourly": ",".join(variables)}
        try:
            return api_get(url, params, budget, weight_fn(len(variables))), variables
        except ApiError as e:
            bad = [v for v in variables if v in str(e)]
            if not bad:
                raise
            print(f"  ! The API rejected {bad} ({e}). Continuing without it.")
            rejected.update(bad)
            variables = [v for v in variables if v not in bad]
    raise ApiError("No variable left to request.")


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


def assign_cells(coords: pd.DataFrame, grid: float) -> pd.DataFrame:
    """Snap each station to the model grid so stations in the same cell share one download."""
    out = coords.copy()
    if grid > 0:
        out["cell_lat"] = (np.round(out["lat"] / grid) * grid).round(4)
        out["cell_lon"] = (np.round(out["lon"] / grid) * grid).round(4)
    else:
        out["cell_lat"], out["cell_lon"] = out["lat"].round(4), out["lon"].round(4)
    out["cell"] = [f"{a:.4f}_{b:.4f}" for a, b in zip(out["cell_lat"], out["cell_lon"])]
    return out


# --------------------------------------------------------------------------- #
# Download plan
# --------------------------------------------------------------------------- #
def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks, y = [], start.year
    while y <= end.year:
        a, z = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
        chunks.append((a, z))
        y += 1
    return chunks


def chunk_path(cache: Path, model: str, cell: str, a: date, z: date) -> Path:
    return cache / model / f"{cell}_{a:%Y%m%d}_{z:%Y%m%d}.parquet"


def download_archive(cells: pd.DataFrame, start: date, end: date, model: str,
                     variables: list[str], cache: Path, budget: CallBudget,
                     dry_run: bool = False) -> list[Path]:
    """Download every (cell, year) chunk that is not cached yet. Returns all chunk paths."""
    jobs = []                                       # (a, z, cell, lat, lon, path)
    for a, z in year_chunks(start, end):
        for _, c in cells.iterrows():
            jobs.append((a, z, c["cell"], c["cell_lat"], c["cell_lon"],
                         chunk_path(cache, model, c["cell"], a, z)))
    todo = [j for j in jobs if not j[-1].exists()]
    cost = sum(call_weight(1, (j[1] - j[0]).days + 1, len(variables)) for j in todo)
    print(f"[{model}] {start} -> {end}: {len(jobs)} chunks, {len(jobs) - len(todo)} cached, "
          f"{len(todo)} to download (~{cost:,.0f} API calls)")
    if cost > FREE_LIMITS[2][1] * SAFETY:
        print(f"  Note: more than one day of free budget. The script stops when the daily budget "
              f"is used; re-run it tomorrow and it resumes ({math.ceil(cost / 8000)} days).")
    if dry_run or not todo:
        return [j[-1] for j in jobs]

    # Group cells of the same period into multi-location requests
    by_period: dict[tuple[date, date], list] = {}
    for j in todo:
        by_period.setdefault((j[0], j[1]), []).append(j)
    done = 0
    for (a, z), group in by_period.items():
        days = (z - a).days + 1
        per_req = max(1, int(MAX_WEIGHT_PER_REQUEST // call_weight(1, days, len(variables))))
        for i in range(0, len(group), per_req):
            batch = group[i:i + per_req]
            params = {
                "latitude": ",".join(f"{j[3]:.4f}" for j in batch),
                "longitude": ",".join(f"{j[4]:.4f}" for j in batch),
                "start_date": f"{a:%Y-%m-%d}", "end_date": f"{z:%Y-%m-%d}",
                "models": model, "wind_speed_unit": "ms", "timeformat": "unixtime",
                "timezone": "GMT", "cell_selection": "nearest",
            }
            resp, got = fetch_with_fallback(
                ARCHIVE_URL, params, variables, budget,
                lambda nv, n=len(batch), d=days: call_weight(n, d, nv))
            if len(resp) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} locations, got {len(resp)}.")
            for j, loc in zip(batch, resp):
                df = parse_location(loc, got)
                j[-1].parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(j[-1])
            done += len(batch)
            print(f"  {a.year}: {done}/{len(todo)} chunks done")
    return [j[-1] for j in jobs]


# --------------------------------------------------------------------------- #
# Forecast weather (used by predict.py for genuine forecasts beyond the data)
# --------------------------------------------------------------------------- #
def fetch_forecast(lats, lons, times_utc: pd.DatetimeIndex,
                   variables: list[str] = DEFAULT_VARIABLES) -> pd.DataFrame:
    """Hourly forecast (or recent past) weather at given points from the Forecast API.
    Returns a long DataFrame: point (index into lats), time (UTC), one column per variable.
    Variables the API does not return come back as NaN."""
    now = pd.Timestamp.now(tz="UTC")
    past_days = int(np.clip(math.ceil((now - times_utc.min()) / pd.Timedelta(days=1)) + 1, 0, 92))
    fc_days = int(np.clip(math.ceil((times_utc.max() - now) / pd.Timedelta(days=1)) + 1, 1, 16))
    params = {"latitude": ",".join(f"{x:.4f}" for x in lats),
              "longitude": ",".join(f"{x:.4f}" for x in lons),
              "past_days": past_days, "forecast_days": fc_days,
              "wind_speed_unit": "ms", "timeformat": "unixtime", "timezone": "GMT"}
    resp, got = fetch_with_fallback(FORECAST_URL, params, variables, None, lambda nv: 1.0)
    frames = []
    for i, loc in enumerate(resp):
        df = parse_location(loc, got).reindex(times_utc)
        for v in variables:
            if v not in df:
                df[v] = np.nan
        df["point"] = i
        frames.append(df.rename_axis("time").reset_index())
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def processed_period_and_stations(processed: str):
    """Stations and time range of the PM2.5 data (so the weather covers exactly that)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import find_processed_dir
    d = find_processed_dir(processed)
    meta = json.loads((d / "meta.json").read_text())
    ts = np.load(d / "dataset.npz")["timestamps"]
    unit = "ns" if ts.max() > 1e17 else "us"
    times = pd.to_datetime(ts[[0, -1]], unit=unit, utc=True)
    return meta["stations"], times[0].date(), times[1].date(), d


def main() -> None:
    p = argparse.ArgumentParser(description="Download hourly weather at the PM2.5 stations.")
    p.add_argument("--processed", default="data/processed",
                   help="processed PM2.5 data (gives the stations and the period)")
    p.add_argument("--metadata", type=Path, default=Path("data/raw/eea/stations_metadata.csv"))
    p.add_argument("--coords", type=Path, default=None,
                   help="optional CSV with columns station, lat, lon (overrides the EEA metadata)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: start of the PM2.5 data)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: end of the PM2.5 data)")
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--model", default="era5",
                   help="Open-Meteo reanalysis: era5 (default), era5_seamless, era5_land, "
                        "ecmwf_ifs, best_match")
    p.add_argument("--grid", type=float, default=None,
                   help="snap stations to this grid in degrees (default 0.25 for era5*, "
                        "0.1 for era5_land, 0 = no snapping)")
    p.add_argument("--no-recent", action="store_true",
                   help="do not fill the last days (not yet in ERA5) with best_match")
    p.add_argument("--out", type=Path, default=Path("data/raw/weather"))
    p.add_argument("--dry-run", action="store_true", help="print the plan and the cost only")
    args = p.parse_args()

    stations, d0, d1, _ = processed_period_and_stations(args.processed)
    start = date.fromisoformat(args.start) if args.start else d0 - timedelta(days=1)
    end = date.fromisoformat(args.end) if args.end else d1 + timedelta(days=1)
    today = datetime.now(timezone.utc).date()
    end = min(end, today)
    print(f"{len(stations)} stations | {start} -> {end} | model {args.model}")
    print(f"Variables: {args.variables}")

    coords = resolve_coordinates(stations, args.metadata, args.coords)
    grid = args.grid if args.grid is not None else (
        0.1 if args.model == "era5_land" else 0.25 if args.model.startswith("era5") else 0.0)
    coords = assign_cells(coords, grid)
    cells = coords.drop_duplicates("cell")[["cell", "cell_lat", "cell_lon"]].reset_index(drop=True)
    print(f"{len(stations)} stations fall into {len(cells)} distinct grid cells "
          f"(grid {grid or 'none'} deg)")
    args.out.mkdir(parents=True, exist_ok=True)
    coords.to_csv(args.out / "stations.csv", index=False)

    cache = args.out / "cache"
    budget = CallBudget(cache / "call_ledger.json")
    is_era5 = args.model.startswith("era5")
    main_end = min(end, today - timedelta(days=ERA5_DELAY_DAYS)) if is_era5 else end
    tasks = [(args.model, start, main_end)]
    if is_era5 and not args.no_recent and end > main_end:
        tasks.append(("best_match", main_end + timedelta(days=1), end))

    paths: list[tuple[str, Path]] = []
    try:
        for model, a, z in tasks:
            for path in download_archive(cells, a, z, model, args.variables, cache, budget,
                                         args.dry_run):
                paths.append((model, path))
    except DailyLimitReached as e:
        print(f"\n{e}\nEverything downloaded so far is cached. Run the same command tomorrow "
              "to continue.")
        sys.exit(2)
    if args.dry_run:
        return

    # Combine all chunks: one row per (cell, hour). ERA5 wins over best_match on overlaps.
    frames = []
    for model, path in paths:
        df = pd.read_parquet(path)
        df["cell"] = path.name.rsplit("_", 2)[0]
        df["source"] = model
        frames.append(df.reset_index())
    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.as_unit("ns")
    data["_prio"] = (data["source"] != args.model).astype(int)
    data = (data.sort_values(["_prio"]).drop_duplicates(["cell", "time"])
                .drop(columns="_prio").sort_values(["cell", "time"]).reset_index(drop=True))
    out_file = args.out / "weather_cells.parquet"
    data.to_parquet(out_file, index=False)

    print(f"\nSaved {len(data):,} rows ({data['cell'].nunique()} cells) -> {out_file}")
    print(f"Time range (UTC): {data['time'].min()} -> {data['time'].max()}")
    print("Missing values per variable:")
    for v in args.variables:
        if v in data:
            print(f"  {v:<24} {data[v].isna().mean():6.2%}")
        else:
            print(f"  {v:<24} not available")
    print("\nNext: python prepare_weather.py")


if __name__ == "__main__":
    main()

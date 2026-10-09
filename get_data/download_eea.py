"""
download_eea.py
===============
Download hourly air-quality measurements from the European Environment Agency
(EEA) Air Quality Download Service, for every monitoring station in a country.

Default: PM2.5, hourly, all Belgian stations, verified (E1a) + up-to-date (E2a) data.

How the EEA service works (API docs: https://eeadmz1-downloads-api-appservice.azurewebsites.net/swagger/index.html)
---------------------------------------------------------------------------------
1. GET  /Pollutant          -> maps pollutant names ("PM2.5") to vocabulary URIs
2. POST /DownloadSummary    -> how many files / MB a request represents
3. POST /ParquetFile/urls   -> list of Parquet file URLs (one file per sampling point)
4. GET  <each url>          -> the actual Parquet file
Station coordinates/types come from a separate metadata CSV (all of Europe).

Datasets:
    1 = "unverified" (E2a, up-to-date, recent data, not yet validated)
    2 = "verified"   (E1a, officially validated, published each year with ~1 year delay)
    3 = "historical" (Airbase, 2002-2012)

Usage
-----
    pip install requests pandas pyarrow tqdm
    python download_eea.py                                  # Belgium, PM2.5
    python download_eea.py --pollutants PM2.5 PM10 NO2      # several pollutants
    python download_eea.py --countries BE NL LU FR          # Benelux + France later
    python download_eea.py --summary-only                   # just check the size

Output
------
    data/raw/eea/<dataset>/<COUNTRY>/*.parquet     raw files, as delivered
    data/raw/eea/stations_metadata.csv             station metadata (filtered to your countries)
    data/raw/eea/<COUNTRIES>_<POLLUTANTS>_hourly.parquet   one combined table
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kwargs):
        return it

API_BASE = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"
METADATA_URL = (
    "https://discomap.eea.europa.eu/App/AQViewer/download"
    "?fqn=Airquality_Dissem.b2g.measurements&f=csv"
)
DATASETS = {"unverified": 1, "verified": 2, "historical": 3}

# Columns we keep from the raw Parquet files
KEEP_COLUMNS = [
    "Samplingpoint", "Pollutant", "Start", "End", "Value", "Unit",
    "AggType", "Validity", "Verification",
]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _request(method: str, url: str, retries: int = 4, **kwargs) -> requests.Response:
    """HTTP request with simple exponential back-off (the EEA server is sometimes slow)."""
    kwargs.setdefault("timeout", 120)
    for attempt in range(retries):
        try:
            r = requests.request(method, url, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  ! {e.__class__.__name__} on {url[:80]}... retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def get_pollutant_uris(names: list[str]) -> list[str]:
    """Translate pollutant notations (e.g. 'PM2.5') into the URIs the API expects."""
    vocab = _request("GET", f"{API_BASE}/Pollutant").json()
    lookup = {p["notation"].lower(): p["id"] for p in vocab}
    uris = []
    for name in names:
        uri = lookup.get(name.lower())
        if uri is None:
            close = [p["notation"] for p in vocab if name.lower() in p["notation"].lower()]
            raise ValueError(f"Unknown pollutant '{name}'. Did you mean one of: {close[:10]}?")
        uris.append(uri)
    return uris


def build_payload(country: str, pollutant_uris: list[str], dataset: int) -> dict:
    return {
        "countries": [country],
        "cities": [],
        "pollutants": pollutant_uris,
        "dataset": dataset,
        "aggregationType": "hour",
        "source": "API",
    }


def get_summary(payload: dict) -> dict:
    return _request("POST", f"{API_BASE}/DownloadSummary", json=payload).json()


def get_file_urls(payload: dict) -> list[str]:
    """The endpoint returns plain text: a header line then one URL per line."""
    text = _request("POST", f"{API_BASE}/ParquetFile/urls", json=payload).text
    return [line.strip() for line in text.splitlines() if line.strip().endswith(".parquet")]


def download_file(url: str, dest_dir: Path, overwrite: bool = False) -> Path:
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    r = _request("GET", url, timeout=300)
    dest.write_bytes(r.content)
    return dest


# --------------------------------------------------------------------------- #
# Station metadata
# --------------------------------------------------------------------------- #
COUNTRY_NAMES = {"BE": "belgium", "NL": "netherlands", "LU": "luxembourg", "FR": "france",
                 "DE": "germany"}


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read a CSV whose delimiter/encoding we don't know in advance."""
    head = content[:4096]
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError("The metadata URL returned an HTML page instead of a CSV.")
    text_head = head.decode("utf-8-sig", errors="replace").splitlines()[0]
    sep = max([",", ";", "\t"], key=text_head.count)
    return pd.read_csv(io.BytesIO(content), sep=sep, encoding="utf-8-sig", low_memory=False)


def filter_metadata(meta: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Keep rows of our countries, whatever the column layout of this release is."""
    codes = {c.upper() for c in countries}
    names = {COUNTRY_NAMES.get(c, c).lower() for c in codes}
    mask = pd.Series(False, index=meta.index)
    for col in meta.columns:
        if pd.api.types.is_numeric_dtype(meta[col]):
            continue
        s = meta[col].astype(str).str.strip()
        low = col.lower()
        if "country" in low or low in ("iso", "countrycode"):
            mask |= s.str.upper().isin(codes) | s.str.lower().isin(names)
        if "sampling" in low or "station" in low:
            # ids like 'BE/SPO-BELAL01_06001_100' or 'SPO-BELAL01_...' or 'BELAL01'
            pat = "|".join(rf"^{c}/|^SPO-{c}|^{c}[A-Z0-9]" for c in codes)
            mask |= s.str.upper().str.contains(pat, regex=True)
    return meta[mask]


def download_metadata(out_csv: Path, countries: list[str], overwrite: bool = False) -> Path:
    """Download the Europe-wide station metadata, keep a raw copy, and save our countries."""
    raw_csv = out_csv.with_name("stations_metadata_europe.csv")
    if out_csv.exists() and out_csv.stat().st_size > 1000 and not overwrite:
        print(f"Metadata already present: {out_csv}")
        return out_csv
    if raw_csv.exists() and not overwrite:
        meta = pd.read_csv(raw_csv, low_memory=False)
    else:
        print("Downloading station metadata (all of Europe, can take a minute)...")
        content = _request("GET", METADATA_URL, timeout=600).content
        if zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                content = zf.read(csv_name)
        meta = _read_csv_bytes(content)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(raw_csv, index=False)
    print(f"Europe-wide metadata: {len(meta):,} rows. Columns: {list(meta.columns)}")

    ours = filter_metadata(meta, countries)
    if ours.empty:
        print("  ! Could not identify our countries' rows; saving the full European file instead. "
              "preprocess.py will still match stations by sampling point id.")
        ours = meta
    ours.to_csv(out_csv, index=False)
    print(f"Saved {len(ours):,} metadata rows -> {out_csv}")
    return out_csv


# --------------------------------------------------------------------------- #
# Combine raw files into one table
# --------------------------------------------------------------------------- #
def combine_parquet_files(files: list[tuple[Path, str, str]], out_path: Path) -> pd.DataFrame:
    """Read every downloaded file, keep hourly rows, tag with dataset/country, save one file."""
    frames = []
    for path, dataset_name, country in tqdm(files, desc="Combining", unit="file"):
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # corrupted/empty file
            print(f"  ! Skipping unreadable file {path.name}: {e}")
            continue
        df = df[[c for c in KEEP_COLUMNS if c in df.columns]]
        if "AggType" in df.columns:
            df = df[df["AggType"].astype(str).str.lower() == "hour"]
        df["dataset"] = dataset_name
        df["country"] = country
        frames.append(df)

    if not frames:
        raise RuntimeError("No data could be read from the downloaded files.")

    data = pd.concat(frames, ignore_index=True)
    # Verified data supersedes unverified data for the same sampling point and hour.
    priority = {"verified": 0, "unverified": 1, "historical": 2}
    data["_prio"] = data["dataset"].map(priority)
    data = (data.sort_values("_prio")
                .drop_duplicates(subset=["Samplingpoint", "Pollutant", "Start"], keep="first")
                .drop(columns="_prio")
                .sort_values(["Samplingpoint", "Start"])
                .reset_index(drop=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out_path, index=False)
    return data


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Download EEA hourly air-quality data.")
    p.add_argument("--countries", nargs="+", default=["BE"], help="ISO-2 codes, e.g. BE NL LU FR")
    p.add_argument("--pollutants", nargs="+", default=["PM2.5"], help="e.g. PM2.5 PM10 NO2 O3")
    p.add_argument("--datasets", nargs="+", default=["verified", "unverified"],
                   choices=list(DATASETS), help="which EEA datasets to fetch")
    p.add_argument("--out", type=Path, default=Path("data/raw/eea"), help="output folder")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    p.add_argument("--summary-only", action="store_true", help="only print download size")
    p.add_argument("--no-metadata", action="store_true", help="skip station metadata")
    p.add_argument("--overwrite", action="store_true", help="re-download existing files")
    p.add_argument("--metadata-only", action="store_true",
                   help="only (re)download the station metadata, not the measurements")
    args = p.parse_args()

    if args.metadata_only:
        download_metadata(args.out / "stations_metadata.csv",
                          [c.upper() for c in args.countries], overwrite=True)
        return

    countries = [c.upper() for c in args.countries]
    print(f"Countries: {countries} | Pollutants: {args.pollutants} | Datasets: {args.datasets}")
    pollutant_uris = get_pollutant_uris(args.pollutants)

    # 1) Collect file URLs for every (dataset, country) pair
    jobs: list[tuple[str, Path, str, str]] = []  # (url, dest_dir, dataset, country)
    for ds_name in args.datasets:
        for country in countries:
            payload = build_payload(country, pollutant_uris, DATASETS[ds_name])
            summary = get_summary(payload)
            print(f"[{ds_name:>10} | {country}] {summary.get('numberFiles', '?')} files, "
                  f"~{summary.get('size', '?')} MB")
            if args.summary_only:
                continue
            urls = get_file_urls(payload)
            dest = args.out / ds_name / country
            dest.mkdir(parents=True, exist_ok=True)
            jobs += [(u, dest, ds_name, country) for u in urls]

    if args.summary_only:
        return
    if not jobs:
        print("Nothing to download (no files matched your request).")
        return

    # 2) Download in parallel
    downloaded: list[tuple[Path, str, str]] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, u, d, args.overwrite): (u, ds, c)
                   for u, d, ds, c in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            url, ds, c = futures[fut]
            try:
                downloaded.append((fut.result(), ds, c))
            except Exception as e:
                failed.append(url)
                print(f"  ! Failed: {url} ({e})")
    print(f"Downloaded {len(downloaded)} files, {len(failed)} failures.")

    # 3) Station metadata (coordinates, station type, altitude...)
    if not args.no_metadata:
        try:
            download_metadata(args.out / "stations_metadata.csv", countries, args.overwrite)
        except Exception as e:
            print(f"  ! Metadata download failed ({e}). You can re-run with --no-metadata later.")

    # 4) Combine everything into a single table
    tag = f"{'-'.join(countries)}_{'-'.join(args.pollutants)}_hourly".replace(".", "")
    out_file = args.out / f"{tag}.parquet"
    data = combine_parquet_files(downloaded, out_file)
    print(f"\nCombined table: {len(data):,} rows, "
          f"{data['Samplingpoint'].nunique()} sampling points -> {out_file}")
    print(f"Time range: {data['Start'].min()}  ->  {data['Start'].max()}")
    print(data.head())


if __name__ == "__main__":
    sys.exit(main())
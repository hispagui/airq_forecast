"""
forecast_weather.py
===================
Weather FORECAST for the stations, used only by predict.py when it forecasts beyond the end
of the data. ERA5 describes the past; tomorrow's weather has to come from a weather forecast.

Source: the Open-Meteo Forecast API (free, no account, one request per forecast). The values
are converted to the same physical variables and units as ERA5 (see prepare_weather.py), so
the model receives features built exactly like during training.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo name -> common variable of prepare_weather.py
OPEN_METEO = {
    "temperature_2m": "t2m", "relative_humidity_2m": "rh", "surface_pressure": "sp",
    "precipitation": "tp", "boundary_layer_height": "blh", "shortwave_radiation": "ssrd",
    "wind_speed_10m": None, "wind_direction_10m": None,      # -> u10, v10
}


def fetch_forecast(lats, lons, times_utc: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Common variables (u10, v10, t2m, rh, sp, tp, blh, ssrd), each (len(times), n_points).
    Works for the next 16 days and the past 92 days. Missing values are NaN."""
    now = pd.Timestamp.now(tz="UTC")
    past = int(np.clip(math.ceil((now - times_utc.min()) / pd.Timedelta(days=1)) + 1, 0, 92))
    ahead = int(np.clip(math.ceil((times_utc.max() - now) / pd.Timedelta(days=1)) + 1, 1, 16))
    variables = list(OPEN_METEO)
    params = {"latitude": ",".join(f"{x:.4f}" for x in lats),
              "longitude": ",".join(f"{x:.4f}" for x in lons),
              "hourly": ",".join(variables), "past_days": past, "forecast_days": ahead,
              "wind_speed_unit": "ms", "timeformat": "unixtime", "timezone": "GMT"}
    r = requests.get(FORECAST_URL, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo forecast request failed ({r.status_code}): {r.text[:300]}")
    resp = r.json()
    resp = resp if isinstance(resp, list) else [resp]

    cols: dict[str, list[np.ndarray]] = {v: [] for v in variables}
    for loc in resp:
        h = loc.get("hourly", {})
        t = pd.DatetimeIndex(pd.to_datetime(np.asarray(h["time"], dtype="int64") * 10**9,
                                            unit="ns", utc=True))
        for v in variables:
            s = pd.Series(np.asarray(h.get(v, [np.nan] * len(t)), dtype="float64"), index=t)
            cols[v].append(s.reindex(times_utc).to_numpy())
    arr = {v: np.stack(c, axis=1) for v, c in cols.items()}         # (T, n_points)

    speed, direction = arr["wind_speed_10m"], np.deg2rad(arr["wind_direction_10m"])
    out = {"u10": -speed * np.sin(direction), "v10": -speed * np.cos(direction)}
    for v, common in OPEN_METEO.items():
        if common:
            out[common] = arr[v]
    return out

"""
dataset.py
==========
PyTorch datasets and dataloaders built from the output of preprocess.py.

One sample = one forecast issued at time t0:

    history (inputs)                     |  future (targets)
    t0-L ... t0-1                        |  t0 ... t0+H-1
    ------------------------------------>|------------------------->
    past PM2.5 at all stations           |  PM2.5 to predict at all stations
    + "was it observed?" flag            |  + mask (which targets are real)
    + calendar features                  |  + calendar features (known in advance)

Every model in the project (MLP, LSTM, GNN, hybrid quantum) gets the SAME batch:

    x            (B, L, N, F_in)  inputs per hour and station
    x_future     (B, H, F_time)   calendar features of the hours to predict
    y            (B, H, N)        normalized targets (log1p + z-score)
    y_mask       (B, H, N)        True where the target was observed (use it in the loss!)
    t0           (B,)             index of the first predicted hour (for plotting / debugging)

with B = batch, L = history length, H = horizon, N = number of stations.
Input features F_in = [pm25_normalized, observed_flag, 7 calendar features] = 9.

Leakage rules
-------------
* A sample belongs to the split that contains ALL of its TARGET hours.
* Its input history may reach back into the previous split (e.g. the first validation
  sample on 1 Jan 2024 uses the last 72 h of 2023 as input). That is not leakage: in
  real life you always know the past. What we never do is train on target hours that
  belong to validation or test.

Usage
-----
    from dataset import make_dataloaders
    loaders, info = make_dataloaders("data/processed", history=72, horizon=24, batch_size=64)
    batch = next(iter(loaders["train"]))

    python dataset.py            # sanity check: shapes, sample counts, round-trip test
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Loading the preprocessed arrays
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedData:
    values: torch.Tensor         # (T, N) float32, normalized, 0 where missing
    mask: torch.Tensor           # (T, N) bool, True where a (possibly short-gap-filled) value exists
    raw: np.ndarray              # (T, N) float32, ug/m3, NaN where missing
    time_features: torch.Tensor  # (T, F_time) float32
    timestamps: np.ndarray       # (T,) int64, ns since epoch, UTC
    meta: dict
    norm_mean: np.ndarray        # (N,) mean of log1p(PM2.5) used for the z-score
    norm_std: np.ndarray         # (N,)

    @property
    def n_nodes(self) -> int:
        return self.values.shape[1]

    @property
    def stations(self) -> list[str]:
        return self.meta["stations"]


def find_processed_dir(processed_dir: str | Path = "data/processed") -> Path:
    """Find the processed data wherever you run the script from.
    A relative path like 'data/processed' is tried from the current folder, then from each
    parent folder (so running from model/ or src/ still finds <repo>/data/processed)."""
    p = Path(processed_dir)
    if p.is_absolute() or (p / "dataset.npz").exists():
        return p
    here = Path(__file__).resolve().parent
    for base in [Path.cwd(), *Path.cwd().parents, here, *here.parents]:
        if (base / p / "dataset.npz").exists():
            return base / p
    raise FileNotFoundError(
        f"Could not find '{processed_dir}/dataset.npz' from {Path.cwd()} or its parent folders. "
        "Run preprocess.py first, or pass the full path.")


def load_processed(processed_dir: str | Path) -> ProcessedData:
    processed_dir = find_processed_dir(processed_dir)
    print(f"Loading processed data from {processed_dir}")
    arrays = np.load(processed_dir / "dataset.npz")
    meta = json.loads((processed_dir / "meta.json").read_text())
    stations = meta["stations"]
    norm = meta["normalization"]

    # Global or per-station statistics -> always (N,) arrays, so the code below is the same
    if norm["mode"] == "global":
        mean = np.full(len(stations), norm["mean"], dtype=np.float32)
        std = np.full(len(stations), norm["std"], dtype=np.float32)
    else:
        mean = np.array([norm["mean"][s] for s in stations], dtype=np.float32)
        std = np.array([norm["std"][s] for s in stations], dtype=np.float32)

    return ProcessedData(
        values=torch.from_numpy(arrays["values"].astype(np.float32)),
        mask=torch.from_numpy(arrays["mask"].astype(bool)),
        raw=arrays["raw"],
        time_features=torch.from_numpy(arrays["time_features"].astype(np.float32)),
        timestamps=arrays["timestamps"],
        meta=meta,
        norm_mean=mean,
        norm_std=std,
    )


# --------------------------------------------------------------------------- #
# Normalization helpers (you will need them for metrics in ug/m3)
# --------------------------------------------------------------------------- #
class Normalizer:
    """Converts between ug/m3 and the normalized space the model works in.

    forward transform:  z = (log1p(x) - mean) / std
    inverse transform:  x = expm1(z * std + mean)
    Station axis must be the LAST axis of the tensor (true for x[..., 0], y, predictions).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def to(self, device) -> "Normalizer":
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self

    def normalize(self, x_ugm3: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(x_ugm3) - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        # clamp: a wild prediction must not overflow expm1
        return torch.expm1((z * self.std + self.mean).clamp(max=10.0)).clamp(min=0.0)


# --------------------------------------------------------------------------- #
# The Dataset
# --------------------------------------------------------------------------- #
class AirQualityWindows(Dataset):
    """Sliding windows over the full timeline, restricted to one split's target hours."""

    def __init__(
        self,
        data: ProcessedData,
        split: str,
        history: int = 72,
        horizon: int = 24,
        stride: int = 1,
        min_history_valid: float = 0.5,
        min_target_valid: float = 0.5,
    ):
        assert split in data.meta["splits"], f"unknown split {split}"
        self.data = data
        self.split = split
        self.history = history
        self.horizon = horizon

        # Inputs per time step and station: value, observed flag (+ calendar features broadcast)
        self.node_inputs = torch.stack([data.values, data.mask.float()], dim=-1)  # (T, N, 2)

        split_start, split_end = data.meta["splits"][split]
        # t0 = first target hour. Targets [t0, t0+H) must be inside the split,
        # history [t0-L, t0) must exist in the timeline (may belong to the previous split).
        first = max(split_start, history)
        last = split_end - horizon  # inclusive
        candidates = np.arange(first, last + 1, stride)

        # Drop windows with too little real data (a station broken for a week, etc.)
        mask_np = data.mask.numpy()
        cum = np.concatenate([np.zeros((1,), dtype=np.int64),
                              np.cumsum(mask_np.sum(axis=1), dtype=np.int64)])
        n = data.n_nodes
        hist_valid = (cum[candidates] - cum[candidates - history]) / (history * n)
        targ_valid = (cum[candidates + horizon] - cum[candidates]) / (horizon * n)
        keep = (hist_valid >= min_history_valid) & (targ_valid >= min_target_valid)
        self.t0s = candidates[keep]
        self.n_dropped = int((~keep).sum())

    def __len__(self) -> int:
        return len(self.t0s)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t0 = int(self.t0s[i])
        L, H = self.history, self.horizon
        hist = slice(t0 - L, t0)
        fut = slice(t0, t0 + H)

        node_part = self.node_inputs[hist]                                   # (L, N, 2)
        time_part = self.data.time_features[hist]                            # (L, F_time)
        time_part = time_part.unsqueeze(1).expand(-1, self.data.n_nodes, -1)  # (L, N, F_time)

        return {
            "x": torch.cat([node_part, time_part], dim=-1),   # (L, N, F_in)
            "x_future": self.data.time_features[fut],        # (H, F_time)
            "y": self.data.values[fut],                      # (H, N)
            "y_mask": self.data.mask[fut],                   # (H, N)
            "t0": torch.tensor(t0),
        }

    def timestamp(self, t0: int) -> np.datetime64:
        ts = int(self.data.timestamps[t0])
        unit = "ns" if abs(ts) > 1e17 else "us"  # older preprocess runs may have saved microseconds
        return np.datetime64(ts, unit)


# --------------------------------------------------------------------------- #
# Convenience: build all three loaders at once
# --------------------------------------------------------------------------- #
def make_dataloaders(
    processed_dir: str | Path = "data/processed",
    history: int = 72,
    horizon: int = 24,
    batch_size: int = 64,
    train_stride: int = 1,
    eval_stride: int = 1,
    num_workers: int = 0,
    min_history_valid: float = 0.5,
    min_target_valid: float = 0.5,
) -> tuple[dict[str, DataLoader], dict]:
    data = load_processed(processed_dir)
    loaders, sizes = {}, {}
    for split in ("train", "val", "test"):
        ds = AirQualityWindows(
            data, split, history, horizon,
            stride=train_stride if split == "train" else eval_stride,
            min_history_valid=min_history_valid, min_target_valid=min_target_valid,
        )
        sizes[split] = len(ds)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False, pin_memory=torch.cuda.is_available(),
        )

    sample = loaders["train"].dataset[0]
    info = {
        "n_nodes": data.n_nodes,
        "n_features": sample["x"].shape[-1],
        "n_time_features": sample["x_future"].shape[-1],
        "history": history,
        "horizon": horizon,
        "stations": data.stations,
        "sizes": sizes,
        "normalizer": Normalizer(data.norm_mean, data.norm_std),
        "data": data,
    }
    return loaders, info


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--history", type=int, default=72)
    p.add_argument("--horizon", type=int, default=24)
    args = p.parse_args()

    loaders, info = make_dataloaders(args.processed, args.history, args.horizon, batch_size=32)
    data = info["data"]
    print(f"Stations: {info['n_nodes']} | input features: {info['n_features']} "
          f"| history {args.history} h -> horizon {args.horizon} h")
    for split, loader in loaders.items():
        ds = loader.dataset
        print(f"  {split:>5}: {len(ds):>7,} samples ({ds.n_dropped:,} dropped for missing data), "
              f"first forecast {ds.timestamp(ds.t0s[0])}, last {ds.timestamp(ds.t0s[-1])}")

    batch = next(iter(loaders["train"]))
    print("\nOne training batch:")
    for k, v in batch.items():
        print(f"  {k:>9}: {tuple(v.shape)} {v.dtype}")

    # 1) Split boundaries: no target hour may leave its split
    for split, loader in loaders.items():
        a, z = data.meta["splits"][split]
        t0s = loader.dataset.t0s
        assert t0s.min() >= a and t0s.max() + args.horizon <= z, f"leak in {split}"
    print("\nCheck OK: every target window stays inside its split.")

    # 2) Normalization round-trip: denormalized targets must equal raw ug/m3 values
    norm = info["normalizer"]
    y_ugm3 = norm.denormalize(batch["y"])
    raw = torch.stack([torch.from_numpy(data.raw[t:t + args.horizon]) for t in batch["t0"].tolist()])
    m = batch["y_mask"]
    err = (y_ugm3[m] - raw[m]).abs().max().item()
    print(f"Check OK: denormalize(y) matches raw ug/m3 (max abs diff {err:.2e})."
          if err < 1e-2 else f"! Round-trip error {err:.3f} ug/m3, check meta.json")

    # 3) Persistence baseline, to have a first number to beat
    #    forecast for t0+h = value at the same hour yesterday (t0+h-24); works for horizon <= 24
    if args.horizon <= 24 and args.history >= 24:
        se, cnt = 0.0, 0
        for b in loaders["val"]:
            last_day = b["x"][:, -24:, :, 0][:, :args.horizon]              # (B, H, N) normalized
            pred = norm.denormalize(last_day)
            hist_ok = b["x"][:, -24:, :, 1][:, :args.horizon].bool()
            true = norm.denormalize(b["y"])
            ok = b["y_mask"] & hist_ok
            se += ((pred - true)[ok] ** 2).sum().item()
            cnt += ok.sum().item()
        print(f"\nPersistence baseline on validation (same hour yesterday): "
              f"RMSE = {np.sqrt(se / cnt):.2f} ug/m3  <- your models must beat this")
"""
holdout.py
==========
Spatial holdout: can a model forecast PM2.5 at a station it NEVER saw during training?
This is where graph models should shine: an unseen station has to rely on its neighbours.

    python holdout.py train --model gru                               # fold 0, scenario "history"
    python holdout.py train --model gnn --model-args graph=wind --mode virtual
    python holdout.py train --model gnn --model-args graph=wind --fold all     # all 5 folds
    python holdout.py folds                                           # show which stations are in each fold
    python holdout.py compare runs/holdout_*                          # one table for several runs

Folds
-----
The stations are split into 5 FOLDS (about 11 stations each), stratified by station type
(traffic / background / industrial) and spread from west to east, so each fold is a mix of
the whole country. Training on fold k hides the stations of fold k; `--fold all` trains 5
models, one per fold, so EVERY station is forecast once as an unseen station.
The split is saved in data/processed/holdout_folds.json.

Two scenarios (--mode)
----------------------
    history   A new station is installed. It was never seen in training, but at forecast
              time its last 72 h of PM2.5 are available. ("generalize to a new place")
    virtual   A VIRTUAL SENSOR: no PM2.5 measurement at all at that place, only the weather,
              the calendar, its static features and the other stations. ("kriging")
              Training randomly hides the PM2.5 inputs of some training stations
              (--mask-prob, default 0.2) so the model learns to forecast without them.

No leakage
----------
    * hidden stations are never in the training loss, nor in the validation loss used for
      early stopping;
    * their PM2.5 inputs are hidden during training and validation (and also at test time
      in the "virtual" scenario);
    * PM2.5 is re-normalized with statistics of the TRAINING stations and years only
      (per-station statistics would leak the unseen station's average level);
    * the learned station embedding is replaced by static features (coordinates, altitude,
      station and area type): an unseen station has no trained identity, but it has these.

Reference forecasts for the hidden stations
-------------------------------------------
    idw_last_value   inverse-distance-weighted average of the last observed value of the 5
                     nearest TRAINING stations (spatial interpolation, no model)
    idw_model        same weighting applied to the model's own forecasts at those 5
                     training stations: "forecast the neighbours, then interpolate"
                     -> if the model beats this, it learned more than interpolation
    last_value, persistence   (scenario "history" only: they need the station's own history)

Outputs: runs/holdout_<name>/
    results.json          metrics on the hidden stations (and on the training stations)
    stations.csv          per-station RMSE, with the distance to the nearest training station
    fold<k>.pt            the trained model of each fold
These checkpoints use their own normalization: evaluate them with this script, not with
evaluate.py / predict.py.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _import(name: str):
    """Import a project module whether it lives at the root or in features/."""
    try:
        return importlib.import_module(f"features.{name}")
    except ModuleNotFoundError as e:
        if e.name not in ("features", f"features.{name}"):
            raise
        return importlib.import_module(name)


D = _import("dataset")
M = _import("metrics")
G = _import("graph")
from architectures import MODELS, build_model  # noqa: E402

THRESHOLDS = (15.0, 25.0, 50.0)
IDW_NEIGHBOURS = 5


# =========================================================================== #
# Folds and static features
# =========================================================================== #
def make_folds(st: pd.DataFrame, n_folds: int) -> list[int]:
    """Stratified by station type, spread west -> east: fold of each station (same order)."""
    types = st["station_type"].fillna("unknown") if "station_type" in st else pd.Series(["all"] * len(st))
    order = []
    for _, grp in st.assign(_t=types.values).groupby("_t", sort=True):
        order += list(grp.sort_values("lon").index)
    folds = np.zeros(len(st), dtype=int)
    for rank, idx in enumerate(order):
        folds[st.index.get_loc(idx)] = rank % n_folds
    return folds.tolist()


def load_folds(proc: Path, st: pd.DataFrame, n_folds: int) -> list[int]:
    path = proc / "holdout_folds.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["stations"] == st["station"].tolist() and saved["n_folds"] == n_folds:
            return saved["folds"]
    folds = make_folds(st, n_folds)
    path.write_text(json.dumps({"stations": st["station"].tolist(), "n_folds": n_folds,
                                "folds": folds}, indent=2))
    return folds


def static_features(st: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    """(N, S) features known for ANY station, measured or not."""
    cols, names = [], []
    for c in ("lat", "lon"):
        v = st[c].to_numpy(float)
        cols.append((v - v.mean()) / (v.std() + 1e-9)); names.append(c)
    if "altitude" in st and st["altitude"].notna().any():
        v = np.log1p(np.clip(st["altitude"].fillna(st["altitude"].median()).to_numpy(float), 0, None))
        cols.append((v - v.mean()) / (v.std() + 1e-9)); names.append("log_altitude")
    for c in ("station_type", "area_type"):
        if c in st:
            for val in sorted(st[c].fillna("unknown").astype(str).str.lower().unique()):
                cols.append((st[c].fillna("unknown").astype(str).str.lower() == val).to_numpy(float))
                names.append(f"{c}={val}")
    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32), names


# =========================================================================== #
# Data: global re-normalization on training stations, input masking
# =========================================================================== #
def renormalize(data, train_nodes: np.ndarray):
    """Rewrite PM2.5 in a normalization computed on training stations x training years."""
    a, z = data.meta["splits"]["train"]
    mean_n, std_n = data.norm_mean[None, :], data.norm_std[None, :]
    log_ugm3 = data.values.numpy() * std_n + mean_n                      # log1p(ug/m3), (T, N)
    m = data.mask.numpy()
    sel = log_ugm3[a:z][:, train_nodes][m[a:z][:, train_nodes]]
    mu, sigma = float(sel.mean()), float(sel.std() + 1e-6)
    values = np.where(m, (log_ugm3 - mu) / sigma, 0.0).astype(np.float32)
    N = data.n_nodes
    return dataclasses.replace(data, values=torch.from_numpy(values),
                               norm_mean=np.full(N, mu, np.float32),
                               norm_std=np.full(N, sigma, np.float32))


def hide_pm25(x: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Remove the PM2.5 value and 'observed' flag of some stations from the inputs.
    nodes: (N,) bool for every sample, or (B, N) bool per sample."""
    x = x.clone()
    m = nodes.view(1, 1, -1) if nodes.dim() == 1 else nodes[:, None, :]
    m = m.expand(x.shape[0], x.shape[1], -1)
    x[..., 0] = x[..., 0].masked_fill(m, 0.0)
    x[..., 1] = x[..., 1].masked_fill(m, 0.0)
    return x


def prepare_batch(batch, held: torch.Tensor, device, mask_prob: float = 0.0,
                  hide_held: bool = True, gen: torch.Generator | None = None):
    x, xf, y, ym = (batch[k].to(device) for k in ("x", "x_future", "y", "y_mask"))
    hide = held.unsqueeze(0).expand(x.shape[0], -1) if hide_held else torch.zeros(
        x.shape[0], held.numel(), dtype=torch.bool, device=device)
    if mask_prob > 0:      # random extra stations without PM2.5 input (targets kept)
        rnd = torch.rand(hide.shape, generator=gen).to(device) < mask_prob
        hide = hide | (rnd & ~held)
    if hide.any():
        x = hide_pm25(x, hide)
    return x, xf, y, ym


# =========================================================================== #
# Training one fold
# =========================================================================== #
def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def input_names(data, use_weather: bool):
    """Names of x features and x_future features (same order as dataset.window_inputs)."""
    weather = list(data.weather_meta["features"]) if (use_weather and data.weather_meta) else []
    cal = [f"calendar_{i}" for i in range(data.time_features.shape[1])]
    return ["pm25", "observed"] + weather + cal, weather + cal


def train_fold(args, data, held_np, static, device, log):
    N = data.n_nodes
    held = torch.tensor(held_np, device=device)
    use_weather = data.weather is not None and not args.no_weather
    loaders = {}
    for split in ("train", "val", "test"):
        ds = D.AirQualityWindows(data, split, args.history, args.horizon,
                                 stride=args.train_stride if split == "train" else 1,
                                 use_weather=use_weather)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"))
    in_names, fut_names = input_names(data, use_weather)
    info = {"n_nodes": N, "n_features": len(in_names), "n_time_features": data.time_features.shape[1],
            "n_future_features": len(fut_names), "history": args.history, "horizon": args.horizon,
            "feature_names": in_names, "processed": str(args.proc)}
    model = build_model(args.model, info, out_size=M.output_size(args.loss), hidden=args.hidden,
                        layers=args.layers, dropout=args.dropout, residual=True, **args.model_args)
    if not args.station_embedding:
        model.head.use_static_features(static)
    model.to(device)
    loss_fn = M.get_loss(args.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    gen = torch.Generator().manual_seed(args.seed)
    train_only = ~held                                                      # loss mask

    def run(split, train: bool, max_batches=None):
        model.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for i, batch in enumerate(loaders[split]):
                if max_batches and i >= max_batches:
                    break
                x, xf, y, ym = prepare_batch(batch, held, device,
                                             args.mask_prob if train else 0.0, True, gen)
                ym = ym & train_only.view(1, 1, -1)          # never learn / select on held
                pred = model(x, xf)
                loss = loss_fn(pred, y, ym)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += loss.item(); n += 1
        return tot / max(n, 1)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        tr = run("train", True, args.max_train_batches)
        va = run("val", False, args.max_eval_batches)
        sched.step(va)
        better = va < best
        if better:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        log(f"  epoch {epoch:>2}  train {tr:.4f}  val (training stations) {va:.4f}  "
            f"{time.time() - t:4.0f}s" + ("  *best" if better else ""))
        if bad >= args.patience:
            log(f"  early stopping after {epoch} epochs")
            break
    model.load_state_dict(best_state)
    return model, loaders["test"], info


# =========================================================================== #
# Evaluation of one fold on the test year
# =========================================================================== #
@torch.no_grad()
def predict_test(model, loader, held_np, mode, device, loss_name, max_batches=None):
    """Model forecasts (S, H, N) in ug/m3 and the t0 of each sample."""
    model.eval()
    held = torch.tensor(held_np, device=device)
    norm = D.Normalizer(loader.dataset.data.norm_mean, loader.dataset.data.norm_std).to(device)
    preds, t0s = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, xf, _, _ = prepare_batch(batch, held, device, 0.0, hide_held=(mode == "virtual"))
        p = M.point_forecast(model(x, xf), loss_name)
        preds.append(norm.denormalize(p).cpu().numpy())
        t0s.append(batch["t0"].numpy())
    return np.concatenate(preds).astype(np.float32), np.concatenate(t0s)


def references(data, t0s, H, L, held_np, dist, model_pred):
    """Reference forecasts (S, H, N) in ug/m3, NaN where undefined."""
    raw, mask = data.raw, data.mask.numpy()
    T, N = raw.shape
    S = len(t0s)
    # last observed value of every station within the history window
    idx = np.where(mask, np.arange(T)[:, None], -1)
    last_idx = np.maximum.accumulate(idx, axis=0)[t0s - 1]                 # (S, N)
    last = raw[np.clip(last_idx, 0, None), np.arange(N)[None, :]]
    last = np.where(last_idx >= t0s[:, None] - L, last, np.nan)

    train_nodes = np.flatnonzero(~held_np)
    W = np.zeros((N, N))                                                   # IDW weights
    for i in np.flatnonzero(held_np):
        nb = train_nodes[np.argsort(dist[i, train_nodes])[:IDW_NEIGHBOURS]]
        W[i, nb] = 1.0 / np.maximum(dist[i, nb], 1.0) ** 2

    def idw(values):                       # values (S, [H,] N): average over available nbs
        v = np.nan_to_num(values)
        ok = (~np.isnan(values)).astype(float)
        num = np.einsum("...j,ij->...i", v, W)
        den = np.einsum("...j,ij->...i", ok, W)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    refs = {"idw_last_value": np.repeat(idw(last)[:, None, :], H, axis=1),
            "idw_model": idw(model_pred),
            "last_value": np.repeat(last[:, None, :], H, axis=1)}
    src = t0s[:, None] + np.arange(H)[None, :] - 24 * (1 + np.arange(H)[None, :] // 24)
    refs["persistence"] = np.where(mask[src], raw[src], np.nan)
    return refs


def score(pred, true):
    """Metrics on the entries where true is finite. pred/true: (S, H, n)."""
    ok = np.isfinite(true) & np.isfinite(pred)
    p, t = pred[ok], true[ok]
    e = p - t
    out = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
           "bias": float(np.mean(e)), "n": int(ok.sum()),
           "corr": float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else float("nan")}
    se = np.where(ok, (pred - true) ** 2, np.nan)
    out["rmse_per_horizon"] = np.sqrt(np.nanmean(se, axis=(0, 2))).round(4).tolist()
    for thr in THRESHOLDS:
        pa, ta = p >= thr, t >= thr
        tp, fp, fn = int((pa & ta).sum()), int((pa & ~ta).sum()), int((~pa & ta).sum())
        out[f"precision@{thr:g}"] = tp / (tp + fp) if tp + fp else float("nan")
        out[f"recall@{thr:g}"] = tp / (tp + fn) if tp + fn else float("nan")
        out[f"f1@{thr:g}"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return out


# =========================================================================== #
# Commands
# =========================================================================== #
def cmd_folds(args):
    st = G.load_coordinates(args.processed)
    folds = load_folds(D.find_processed_dir(args.processed), st, args.n_folds)
    for k in range(args.n_folds):
        sel = st[np.array(folds) == k]
        types = sel["station_type"].value_counts().to_dict() if "station_type" in sel else {}
        print(f"fold {k}: {len(sel)} stations {types}\n   {', '.join(sel['station'])}")


def cmd_train(args):
    args.model_args = parse_kv(args.model_args)
    if args.mask_prob is None:
        args.mask_prob = 0.2 if args.mode == "virtual" else 0.0
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.proc = D.find_processed_dir(args.processed)
    data0 = D.load_processed(args.proc)
    st = G.load_coordinates(args.proc)
    folds = np.array(load_folds(args.proc, st, args.n_folds))
    static, static_names = static_features(st)
    dist, _ = G.geometry(G.project_km(st["lat"].to_numpy(), st["lon"].to_numpy()))
    norm_mode = data0.meta["normalization"]["mode"]

    name = args.run_name or (f"{args.model}_{'-'.join(f'{k}{v}' for k, v in args.model_args.items())}"
                             f"_{args.mode}_{datetime.now():%Y%m%d_%H%M}").replace("__", "_")
    run_dir = Path(args.runs_dir) / f"holdout_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    fold_list = list(range(args.n_folds)) if args.fold == "all" else [int(args.fold)]
    log(f"Spatial holdout | model {args.model} {args.model_args} | scenario {args.mode} | "
        f"folds {fold_list} | device {device}")
    log(f"Static features: {static_names}")
    if norm_mode != "global":
        log("PM2.5 was normalized per station in preprocess.py: re-normalizing with training "
            "stations only (no leakage of the unseen stations' average level).")

    S_all = None
    pred_full, true_full = None, None
    per_fold, train_scores = [], []
    for k in fold_list:
        held = folds == k
        log(f"\n=== fold {k}: hiding {held.sum()} stations: {', '.join(st['station'][held])}")
        data = renormalize(data0, np.flatnonzero(~held))
        model, test_loader, info = train_fold(args, data, held, static, device, log)
        torch.save({"model": model.state_dict(), "fold": k, "held": held.tolist(),
                    "config": {**{a: v for a, v in vars(args).items() if a != "proc"},
                               "static_features": static_names}},
                   run_dir / f"fold{k}.pt")
        pred, t0s = predict_test(model, test_loader, held, args.mode, device, args.loss,
                                 args.max_eval_batches)
        H, L = args.horizon, args.history
        tidx = t0s[:, None] + np.arange(H)[None, :]
        true = np.where(data.mask.numpy()[tidx], data.raw[tidx], np.nan).astype(np.float32)
        refs = references(data, t0s, H, L, held, dist, pred)
        if pred_full is None:
            S_all = t0s
            pred_full = np.full_like(true, np.nan)
            ref_full = {r: np.full_like(true, np.nan) for r in refs}
            true_full = true
        assert np.array_equal(S_all, t0s), "test windows differ between folds"
        pred_full[..., held] = pred[..., held]
        for r in refs:
            ref_full[r][..., held] = refs[r][..., held]
        f = score(pred[..., held], true[..., held])
        train_scores.append(score(pred[..., ~held], true[..., ~held])["rmse"])
        per_fold.append({"fold": k, "stations": st["station"][held].tolist(), "rmse": f["rmse"],
                         "idw_model_rmse": score(refs["idw_model"][..., held], true[..., held])["rmse"]})
        log(f"  fold {k}: RMSE on hidden stations {f['rmse']:.3f} | idw_model "
            f"{per_fold[-1]['idw_model_rmse']:.3f} | RMSE on training stations {train_scores[-1]:.3f}")

    # ---- results over every station that was hidden --------------------------
    hidden = np.isin(folds, fold_list)
    refs_used = ["idw_last_value", "idw_model"] + (["last_value", "persistence"] if args.mode == "history" else [])
    common = np.isfinite(true_full) & np.isfinite(pred_full)
    for r in refs_used:
        common &= np.isfinite(ref_full[r])
    tf = np.where(common, true_full, np.nan)
    results = {"model": score(pred_full[..., hidden], tf[..., hidden])}
    for r in refs_used:
        results[r] = score(ref_full[r][..., hidden], tf[..., hidden])
    for r in refs_used:
        results["model"][f"skill_vs_{r}"] = 1 - results["model"]["rmse"] / results[r]["rmse"]

    rows = []
    for i in np.flatnonzero(hidden):
        others = np.flatnonzero(folds != folds[i])
        row = {"station": st["station"][i], "fold": int(folds[i]),
               "nearest_training_station_km": round(float(dist[i, others].min()), 1),
               "station_type": st["station_type"][i] if "station_type" in st else "",
               "rmse_model": score(pred_full[..., [i]], tf[..., [i]])["rmse"]}
        for r in refs_used:
            row[f"rmse_{r}"] = score(ref_full[r][..., [i]], tf[..., [i]])["rmse"]
        rows.append(row)
    pd.DataFrame(rows).sort_values("nearest_training_station_km").to_csv(run_dir / "stations.csv", index=False)

    out = {"model": args.model, "model_args": args.model_args, "mode": args.mode,
           "folds": fold_list, "n_hidden_stations": int(hidden.sum()),
           "rmse_on_training_stations": float(np.mean(train_scores)),
           "per_fold": per_fold, "results": results}
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, default=float))
    (run_dir / "log.txt").write_text("\n".join(log_lines))
    log("")
    print_table({name: out})
    log(f"\nSaved to {run_dir}/")


def print_table(runs: dict):
    first = next(iter(runs.values()))
    hz = [h for h in (1, 3, 6, 12, 24) if h <= len(first["results"]["model"]["rmse_per_horizon"])]
    print(f"Hidden stations, test year | lower RMSE is better")
    print(f"{'forecast':<44}{'RMSE':>7}{'MAE':>7}{'bias':>7}" + "".join(f"{'h+'+str(h):>7}" for h in hz)
          + f"{'F1@25':>7}{'F1@50':>7}")
    shown_refs = set()
    for name, out in runs.items():
        for key, r in out["results"].items():
            if key != "model":
                if key in shown_refs:
                    continue
                shown_refs.add(key)
                label = f"  [reference] {key}"
            else:
                label = f"{name}"[:43]
            rh = r["rmse_per_horizon"]
            print(f"{label:<44}{r['rmse']:>7.2f}{r['mae']:>7.2f}{r['bias']:>+7.2f}"
                  + "".join(f"{rh[h - 1]:>7.2f}" for h in hz)
                  + f"{r['f1@25']:>7.2f}{r['f1@50']:>7.2f}")
    print("(references are scored on the hidden stations of the first run listed; compare runs "
          "of the same scenario and folds)")


def cmd_compare(args):
    runs = {}
    for d in args.runs:
        p = Path(d) / "results.json"
        if p.exists():
            out = json.loads(p.read_text())
            runs[Path(d).name.replace("holdout_", "")] = out
    if not runs:
        raise SystemExit("No results.json found in the given folders.")
    modes = {o["mode"] for o in runs.values()}
    folds = {tuple(o["folds"]) for o in runs.values()}
    if len(modes) > 1 or len(folds) > 1:
        print(f"! Runs mix scenarios {modes} or folds {folds}: compare like with like.\n")
    print_table(runs)
    print("\nModel RMSE on the TRAINING stations (for reference):")
    for n, o in runs.items():
        print(f"  {n:<44}{o['rmse_on_training_stations']:.2f}")


def parse_kv(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description="Spatial holdout experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and evaluate on hidden stations")
    t.add_argument("--model", default="gru", choices=sorted(MODELS))
    t.add_argument("--model-args", nargs="*", default=[], metavar="KEY=VALUE")
    t.add_argument("--mode", default="history", choices=["history", "virtual"])
    t.add_argument("--fold", default="0", help="fold number, or 'all'")
    t.add_argument("--n-folds", type=int, default=5)
    t.add_argument("--mask-prob", type=float, default=None,
                   help="fraction of training stations whose PM2.5 inputs are hidden at random "
                        "during training (default 0.2 for virtual, 0 for history)")
    t.add_argument("--station-embedding", action="store_true",
                   help="keep the learned station embedding instead of static features (ablation)")
    t.add_argument("--no-weather", action="store_true")
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--loss", default="mse", choices=sorted(M.LOSSES))
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--train-stride", type=int, default=2)
    t.add_argument("--history", type=int, default=72)
    t.add_argument("--horizon", type=int, default=24)
    t.add_argument("--max-train-batches", type=int, default=None)
    t.add_argument("--max-eval-batches", type=int, default=None)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--processed", default="data/processed")
    t.add_argument("--runs-dir", default="runs")
    t.add_argument("--run-name", default=None)

    f = sub.add_parser("folds", help="show the stations of each fold")
    f.add_argument("--processed", default="data/processed")
    f.add_argument("--n-folds", type=int, default=5)

    c = sub.add_parser("compare", help="compare several holdout runs")
    c.add_argument("runs", nargs="+")

    args = p.parse_args()
    {"train": cmd_train, "folds": cmd_folds, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()

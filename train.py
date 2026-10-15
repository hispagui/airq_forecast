"""
train.py
========
Train a forecaster on 2016-2023, select it on 2024, report it once on 2025.

    python train.py                                  # LSTM, MSE loss, default settings
    python train.py --model mlp                      # the MLP baseline
    python train.py --model gru                      # also: tcn, transformer
    python train.py --model tcn --model-args kernel=5          # model-specific settings
    python train.py --model transformer --model-args patch=4 heads=8
    python train.py --loss quantile                  # 10/50/90 % quantiles
    python train.py --epochs 3 --max-train-batches 50    # quick smoke test (~1 min)
    python train.py --no-weather                     # ablation: PM2.5 + calendar only

Weather features (wind, temperature, precipitation...) are used automatically when
data/processed/weather.npz exists (download_weather.py + prepare_weather.py).

What happens
------------
1. Build the dataloaders (dataset.py) and the model (architectures/).
2. Each epoch:
     train  : gradient steps on 2016-2023 with the masked loss (metrics.py)
     val    : loss + RMSE (ug/m3) + skill vs "last observed value" on 2024
   The checkpoint with the lowest validation loss is kept (early stopping).
3. At the end, the BEST checkpoint is evaluated ONCE on the 2025 test year.
   Never tune settings by looking at test results: that is what validation is for.

Outputs: runs/<run_name>/
    best.pt                 model weights + config
    history.csv             one line per epoch (plot it to see over/under-fitting)
    results.json            final validation and test metrics
    test_predictions.npz    predictions and observations on 2025 (ug/m3), for plots
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from architectures import MODELS, build_model
from features.dataset import make_dataloaders
from features.metrics import LOSSES, Evaluator, format_results, get_loss, output_size, point_forecast


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def to_jsonable(d: dict) -> dict:
    return {k: (np.asarray(v).round(4).tolist() if not np.isscalar(v) else
                (None if v != v else round(float(v), 5))) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# One pass over a loader
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, loss_fn, optimizer, device, max_batches=None, clip=1.0):
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        pred = model(batch["x"], batch["x_future"])
        loss = loss_fn(pred, batch["y"], batch["y_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # avoids rare exploding steps
        optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, loss_name, normalizer, device, threshold,
             max_batches=None, keep_predictions=False):
    model.eval()
    ev = Evaluator(normalizer, threshold=threshold, skill_baseline="last_value")
    total, n, t0s = 0.0, 0, []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        pred = model(batch["x"], batch["x_future"])
        total += loss_fn(pred, batch["y"], batch["y_mask"]).item()
        n += 1
        ev.update(point_forecast(pred, loss_name), batch)
        if keep_predictions:
            t0s.append(batch["t0"].cpu().numpy())
    results = ev.compute()
    results["loss"] = total / max(n, 1)
    if keep_predictions:
        results["_arrays"] = {"pred": np.concatenate(ev._pred), "true": np.concatenate(ev._true),
                              "t0": np.concatenate(t0s)}
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train a PM2.5 forecaster.")
    # data
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--history", type=int, default=72, help="hours of input")
    p.add_argument("--horizon", type=int, default=24, help="hours to predict")
    p.add_argument("--no-weather", action="store_true",
                   help="ignore weather features even if weather.npz exists")
    p.add_argument("--train-stride", type=int, default=2,
                   help="use every k-th training window (consecutive windows overlap a lot)")
    # model
    p.add_argument("--model", default="lstm", choices=sorted(MODELS))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--model-args", nargs="*", default=[], metavar="KEY=VALUE",
                   help="model-specific settings, e.g. kernel=5 (tcn), patch=4 heads=8 "
                        "(transformer)")
    p.add_argument("--no-residual", action="store_true",
                   help="predict the value directly instead of a correction to the last value")
    # optimization
    p.add_argument("--loss", default="mse", choices=sorted(LOSSES))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs)")
    p.add_argument("--max-train-batches", type=int, default=None, help="for quick tests")
    p.add_argument("--max-eval-batches", type=int, default=None, help="for quick tests")
    # misc
    p.add_argument("--threshold", type=float, default=25.0, help="alert threshold, ug/m3")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default="runs")
    return p.parse_args()


def parse_model_args(items: list[str]) -> dict:
    """['kernel=5', 'patch=4'] -> {'kernel': 5, 'patch': 4} (numbers converted)."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--model-args expects KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        out[k] = v
    return out


def main():
    args = parse_args()
    args.model_args = parse_model_args(args.model_args)
    set_seed(args.seed)
    device = pick_device(args.device)
    run_name = args.run_name or f"{args.model}_{args.loss}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -------------------------------------------------------------
    loaders, info = make_dataloaders(
        args.processed, history=args.history, horizon=args.horizon,
        batch_size=args.batch_size, train_stride=args.train_stride, num_workers=args.num_workers,
        use_weather=not args.no_weather)
    normalizer = info["normalizer"]
    print(f"Data: {info['n_nodes']} stations, samples {info['sizes']} "
          f"(train stride {args.train_stride})")
    print(f"Inputs: {info['n_features']} features per hour and station | weather: "
          f"{info['weather_features'] or 'none'}")

    # ---- model ------------------------------------------------------------
    model = build_model(args.model, info, out_size=output_size(args.loss), hidden=args.hidden,
                        layers=args.layers, dropout=args.dropout,
                        residual=not args.no_residual, **args.model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | {n_params:,} parameters | loss: {args.loss} | device: {device}")

    loss_fn = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    config = {**vars(args), "run_name": run_name, "n_params": n_params,
              "stations": info["stations"], "n_features": info["n_features"],
              "n_time_features": info["n_time_features"],
              "n_future_features": info["n_future_features"],
              "use_weather": info["use_weather"], "weather_features": info["weather_features"]}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- training loop ----------------------------------------------------
    history_path = run_dir / "history.csv"
    fields = ["epoch", "train_loss", "val_loss", "val_rmse", "val_skill", "lr", "seconds"]
    with history_path.open("w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_val, best_epoch, bad_epochs = float("inf"), 0, 0
    print(f"\n{'epoch':>5} {'train loss':>11} {'val loss':>9} {'val RMSE':>9} "
          f"{'skill':>7} {'lr':>8} {'time':>6}")
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device,
                                     args.max_train_batches)
        val = evaluate(model, loaders["val"], loss_fn, args.loss, normalizer, device,
                       args.threshold, args.max_eval_batches)
        scheduler.step(val["loss"])
        lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t

        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val["loss"], epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1

        print(f"{epoch:>5} {train_loss:>11.4f} {val['loss']:>9.4f} {val['rmse']:>9.3f} "
              f"{val['skill_rmse']:>7.3f} {lr:>8.1e} {secs:>5.0f}s" + ("  *best" if improved else ""))
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val["loss"], val["rmse"],
                                    val["skill_rmse"], lr, round(secs, 1)])

        if bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    # ---- final evaluation with the best checkpoint ------------------------
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\nBest epoch: {best_epoch} (val loss {best_val:.4f})")

    final = {}
    for split in ("val", "test"):
        res = evaluate(model, loaders[split], loss_fn, args.loss, normalizer, device,
                       args.threshold, args.max_eval_batches, keep_predictions=(split == "test"))
        arrays = res.pop("_arrays", None)
        final[split] = res
        print(f"\n=== {split.upper()} ({'2024' if split == 'val' else '2025'}) ===")
        print(format_results(res, prefix="  "))
        if arrays is not None:
            np.savez_compressed(run_dir / "test_predictions.npz", **arrays,
                                stations=np.array(info["stations"]))

    # Side-by-side per lead time: model vs references, on the test year
    t = final["test"]
    print("\nTest RMSE (ug/m3) per lead time:")
    print(f"  {'':<14}" + "".join(f"{f'h+{h}':>8}" for h in (1, 3, 6, 12, 18, 24)))
    for label, key in (("model", "rmse_per_horizon"), ("last value", "last_value_rmse_per_horizon"),
                       ("persistence", "persistence_rmse_per_horizon")):
        arr = np.asarray(t[key])
        print(f"  {label:<14}" + "".join(f"{arr[h - 1]:>8.2f}" for h in (1, 3, 6, 12, 18, 24)
                                          if h <= len(arr)))

    (run_dir / "results.json").write_text(json.dumps(
        {"best_epoch": best_epoch, **{s: to_jsonable(r) for s, r in final.items()}}, indent=2))
    print(f"\nSaved to {run_dir}/")


if __name__ == "__main__":
    main()

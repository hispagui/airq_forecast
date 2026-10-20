"""
Phase 3 training: put a quantum (or matched classical) head on top of your best Phase 2 model.

Stage A  (main experiment)  --stage head
    Freeze the trained Phase 2 encoder, precompute its node embeddings once,
    and train only  bottleneck -> head -> readout.  Fast, and isolates the head.

Stage B  (optional)         --stage finetune
    Start from a Stage A checkpoint, unfreeze the encoder, fine-tune everything
    with a small encoder learning rate.

Heads:  quantum | classical (parameter-matched) | linear (no head, ablation)

Examples
    python train_quantum.py --demo --head quantum   --k 6 --seed 0
    python train_quantum.py --head quantum   --k 6 --seed 0
    python train_quantum.py --head classical --k 6 --seed 0
    python train_quantum.py --head linear    --k 6 --seed 0
    python train_quantum.py --head quantum   --k 6 --seed 0 --stage finetune
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features.quantum_head import ClassicalHead, QuantumHead, count_params


# =============================================================================
# ADAPT THESE THREE THINGS TO YOUR PROJECT
# =============================================================================
def load_encoder(device) -> nn.Module:
    """Return your trained Phase 2 model *without its final output layer*.

    It must map x (B, T, N, F) -> node embeddings (B, N, D) and expose `.out_dim = D`.
    Easiest: add an `encode(x)` method to your GNN+GRU class that returns the hidden
    state just before the final Linear(D -> 24), then wrap it:

        model = MyGraphGRU(**cfg); model.load_state_dict(torch.load("best_phase2.pt"))
        return EncoderWrapper(model, out_dim=model.hidden_dim)
    """
    raise NotImplementedError("Point this at your best Phase 2 checkpoint")


def get_loaders(batch_size: int):
    """Return (train, val, test) DataLoaders yielding (x, y):
    x (B, T, N, F) normalised with train stats, y (B, H, N) normalised PM2.5.
    Reuse exactly the Dataset / splits from Phases 1-2.
    """
    raise NotImplementedError("Reuse your Phase 1/2 Dataset and chronological splits")


# Train-set statistics of the PM2.5 target, used to report metrics in µg/m³.
TARGET_MEAN, TARGET_STD = 0.0, 1.0
# =============================================================================


class EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, out_dim: int):
        super().__init__()
        self.model, self.out_dim = model, out_dim

    def forward(self, x):
        return self.model.encode(x)


class HeadModel(nn.Module):
    """Embeddings (B, N, D) -> forecasts (B, H, N). The only part that differs between runs."""

    def __init__(self, D: int, head: str, k: int, H: int, n_modes=None, n_photons=None):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.bottleneck = nn.Linear(D, k)
        q = QuantumHead(k=k, n_modes=n_modes, n_photons=n_photons)
        if head == "quantum":
            self.head = q
        elif head == "classical":
            self.head = ClassicalHead(k, q.out_dim, target_params=count_params(q))
        elif head == "linear":
            self.head = None
        else:
            raise ValueError(head)
        # Same readout width for quantum and classical; the linear ablation reads the k angles.
        self.readout = nn.Linear(q.out_dim if self.head is not None else k, H)

    def forward(self, e):
        B, N, _ = e.shape
        z = torch.pi * torch.sigmoid(self.bottleneck(self.norm(e)))   # phases in [0, pi]
        z = z.reshape(B * N, -1)                                       # nodes -> batch
        h = self.head(z) if self.head is not None else z
        return self.readout(h).reshape(B, N, -1).permute(0, 2, 1)     # (B, H, N)


class FullModel(nn.Module):
    def __init__(self, encoder, head_model):
        super().__init__()
        self.encoder, self.head_model = encoder, head_model

    def forward(self, x):
        return self.head_model(self.encoder(x))


# ----------------------------------------------------------------------------- utils
@torch.no_grad()
def precompute(encoder, loader, device):
    """Run the frozen encoder once over a split; returns a TensorDataset(emb, y)."""
    encoder.eval()
    E, Y = [], []
    for x, y in loader:
        E.append(encoder(x.to(device)).cpu())
        Y.append(y)
    return TensorDataset(torch.cat(E), torch.cat(Y))


@torch.no_grad()
def evaluate(model, loader, device):
    """RMSE / MAE per horizon in µg/m³ + WHO daily-mean exceedance precision/recall."""
    model.eval()
    P, Y = [], []
    for inp, y in loader:
        P.append(model(inp.to(device)).cpu())
        Y.append(y)
    p = torch.cat(P) * TARGET_STD + TARGET_MEAN
    y = torch.cat(Y) * TARGET_STD + TARGET_MEAN
    mask = ~torch.isnan(y)
    err = torch.where(mask, p - y, torch.zeros_like(p))
    n = mask.sum(dim=(0, 2)).clamp(min=1)
    rmse_h = (err.pow(2).sum(dim=(0, 2)) / n).sqrt()
    mae_h = err.abs().sum(dim=(0, 2)) / n
    # exceedance of WHO daily guideline: mean over the 24 h horizon > 15 µg/m³
    pe, ye = p.nanmean(1) > 15, y.nanmean(1) > 15
    tp = (pe & ye).sum().item()
    prec = tp / max(pe.sum().item(), 1)
    rec = tp / max(ye.sum().item(), 1)
    return {"rmse": rmse_h.mean().item(), "mae": mae_h.mean().item(),
            "rmse_per_h": rmse_h.tolist(), "mae_per_h": mae_h.tolist(),
            "exceed_precision": prec, "exceed_recall": rec}


def masked_mse(p, y):
    m = ~torch.isnan(y)
    return ((p - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)


def fit(model, param_groups, train_dl, val_dl, device, epochs, patience, ckpt):
    opt = torch.optim.Adam(param_groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 3))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.perf_counter(), 0.0, 0
        for inp, y in train_dl:
            inp, y = inp.to(device), y.to(device)
            loss = masked_mse(model(inp), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        val = evaluate(model, val_dl, device)["rmse"]
        sched.step(val)
        flag = ""
        if val < best - 1e-4:
            best, best_state, bad, flag = val, copy.deepcopy(model.state_dict()), 0, "  *"
            torch.save(best_state, ckpt)
        else:
            bad += 1
        print(f"epoch {ep:3d}  train mse {tot/nb:.4f}  val rmse {val:.3f}  "
              f"({time.perf_counter()-t0:.1f}s){flag}")
        if bad >= patience:
            print("early stopping")
            break
    model.load_state_dict(best_state)
    return model


# ----------------------------------------------------------------------------- demo data
def demo_setup(device):
    """Synthetic stand-in so you can check the pipeline runs before wiring in real data."""
    global TARGET_MEAN, TARGET_STD
    TARGET_MEAN, TARGET_STD = 12.0, 6.0
    from quantum_head import GRUEncoder
    torch.manual_seed(123)
    S, T, N, F, H = 600, 72, 30, 10, 24
    x = torch.randn(S, T, N, F)
    w = torch.randn(F, H) * 0.3
    y = torch.tanh(x[:, -6:].mean(1) @ w).permute(0, 2, 1) + 0.1 * torch.randn(S, H, N)
    ds = TensorDataset(x, y)
    tr, va, te = torch.utils.data.random_split(ds, [400, 100, 100])  # demo only, NOT chronological
    enc = GRUEncoder(F, 32).to(device)
    return enc, tr, va, te


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["quantum", "classical", "linear"], default="quantum")
    ap.add_argument("--stage", choices=["head", "finetune"], default="head")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--modes", type=int, default=None)
    ap.add_argument("--photons", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="bottleneck/readout lr")
    ap.add_argument("--lr-head", type=float, default=1e-2, help="quantum phases / classical head lr")
    ap.add_argument("--lr-encoder", type=float, default=1e-4, help="stage finetune only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results_phase3")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(exist_ok=True)
    tag = f"{a.head}_k{a.k}_m{a.modes or a.k}_s{a.seed}"
    dev = torch.device(a.device)

    if a.demo:
        encoder, tr, va, te = demo_setup(dev)
        mk = lambda d, sh: DataLoader(d, batch_size=a.batch, shuffle=sh)
        train_raw, val_raw, test_raw = mk(tr, True), mk(va, False), mk(te, False)
        a.epochs = min(a.epochs, 15)
    else:
        encoder = load_encoder(dev).to(dev)
        train_raw, val_raw, test_raw = get_loaders(a.batch)

    torch.manual_seed(a.seed)
    head_model = HeadModel(encoder.out_dim, a.head, a.k, a.horizon, a.modes, a.photons).to(dev)
    head_p = list(head_model.head.parameters()) if head_model.head is not None else []
    other_p = [p for n, p in head_model.named_parameters() if not n.startswith("head.")]
    print(f"[{tag}] head params: {sum(p.numel() for p in head_p)}   "
          f"trainable total: {count_params(head_model)}")

    if a.stage == "head":
        # ---- Stage A: frozen encoder, precomputed embeddings --------------------
        for p in encoder.parameters():
            p.requires_grad_(False)
        cache = out / "embeddings.pt"
        if cache.exists() and not a.demo:
            tr_e, va_e, te_e = torch.load(cache)
        else:
            t0 = time.perf_counter()
            tr_e, va_e, te_e = (precompute(encoder, d, dev) for d in (train_raw, val_raw, test_raw))
            print(f"embeddings precomputed in {time.perf_counter()-t0:.1f}s")
            if not a.demo:
                torch.save((tr_e, va_e, te_e), cache)
        g = torch.Generator().manual_seed(a.seed)
        train_dl = DataLoader(tr_e, batch_size=a.batch, shuffle=True, generator=g)
        val_dl = DataLoader(va_e, batch_size=256)
        test_dl = DataLoader(te_e, batch_size=256)
        groups = [{"params": other_p, "lr": a.lr}]
        if head_p:
            groups.append({"params": head_p, "lr": a.lr_head})
        model = fit(head_model, groups, train_dl, val_dl, dev, a.epochs, a.patience,
                    out / f"{tag}_head.pt")
    else:
        # ---- Stage B: end-to-end fine-tune from the Stage A checkpoint -----------
        head_model.load_state_dict(torch.load(out / f"{tag}_head.pt"))
        model = FullModel(encoder, head_model).to(dev)
        groups = [{"params": encoder.parameters(), "lr": a.lr_encoder},
                  {"params": other_p, "lr": a.lr * 0.3}]
        if head_p:
            groups.append({"params": head_p, "lr": a.lr_head * 0.3})
        model = fit(model, groups, train_raw, val_raw, dev, a.epochs, a.patience,
                    out / f"{tag}_finetune.pt")
        test_dl = test_raw

    res = evaluate(model, test_dl, dev)
    res.update(vars(a))
    res["head_params"] = sum(p.numel() for p in head_p)
    res["total_trainable"] = count_params(model)
    with open(out / f"{tag}_{a.stage}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"TEST  rmse {res['rmse']:.3f}  mae {res['mae']:.3f}  "
          f"exceed P/R {res['exceed_precision']:.2f}/{res['exceed_recall']:.2f}")


if __name__ == "__main__":
    main()
"""
graph.py
========
Station graphs for Phase 2. 
Stations are NODES; an EDGE from station j to station i means "information at j is useful to forecast i". 
The weight A[i, j] says how much.

Graphs (choose with  train.py --model gnn --model-args graph=<name>)
---------------------------------------------------------------------
    distance    A[i,j] = exp(-d_ij^2 / sigma^2), weights below `threshold` removed.
                Nearby stations are strongly linked, far ones not at all.
    knn         each station is linked to its k nearest stations (then symmetrized).
    wind        directed and different for EVERY forecast: 
                the weight of j -> i grows when the wind blows from j towards i (j is upwind of i). 
                Built inside the model (architectures/gnn.py) from the wind in the inputs; 
                this file only gives it the geometry (distances, directions, candidate neighbours).
    random      the distance graph with the station labels shuffled: 
                same number of edges and same weights, but the WRONG geography. 
                Control experiment: if it scores as well as `distance`, the gain did not come from geography.
    none        no edges at all: the graph layers then only transform each station on its
                own. Control experiment: same model and parameters, zero spatial information.

Conventions
-----------
    rows = receivers, columns = senders:  message to i = sum_j A[i, j] * h_j
    no self-loops (each station keeps its own information through a residual connection)
    row-normalized (rows sum to 1): a station AVERAGES its neighbours

Coordinates come from data/processed/stations.csv (written by prepare_weather.py).
Distances use a local flat projection in km (error < 0.5 % at the scale of Belgium).

Check and draw the graphs:
    python graph.py                     statistics of every graph
    python graph.py --plot              + data/processed/graphs.png (map of the edges)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.dataset import find_processed_dir

EARTH_RADIUS_KM = 6371.0
STATIC_GRAPHS = ("distance", "knn", "random", "none")
ALL_GRAPHS = STATIC_GRAPHS + ("wind",)


# --------------------------------------------------------------------------- #
# Coordinates and geometry
# --------------------------------------------------------------------------- #
def load_coordinates(processed: str | Path = "data/processed") -> pd.DataFrame:
    """Station coordinates in the SAME order as the processed PM2.5 data."""
    #from dataset import find_processed_dir
    proc = find_processed_dir(processed)
    stations = json.loads((proc / "meta.json").read_text())["stations"]
    if (proc / "stations.csv").exists():
        st = pd.read_csv(proc / "stations.csv")
    elif (proc / "weather_meta.json").exists():
        st = pd.DataFrame(json.loads((proc / "weather_meta.json").read_text())["station_coords"])
    else:
        raise FileNotFoundError(f"No station coordinates in {proc}. Run prepare_weather.py.")
    missing = sorted(set(stations) - set(st["station"]))
    if missing:
        raise ValueError(f"No coordinates for {missing[:10]}. Re-run prepare_weather.py.")
    return st.set_index("station").loc[stations].reset_index()


def project_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(N, 2) positions in km on a local flat map: x towards the east, y towards the north."""
    lat0 = np.deg2rad(np.mean(lat))
    x = EARTH_RADIUS_KM * np.deg2rad(lon) * np.cos(lat0)
    y = EARTH_RADIUS_KM * np.deg2rad(lat)
    pos = np.stack([x, y], axis=1)
    return pos - pos.mean(axis=0)


def geometry(pos_km: np.ndarray):
    """Distances (N, N) in km and unit vectors (N, N, 2): unit[i, j] points FROM j TO i."""
    disp = pos_km[:, None, :] - pos_km[None, :, :]          # disp[i, j] = pos_i - pos_j
    dist = np.linalg.norm(disp, axis=-1)
    unit = disp / np.maximum(dist, 1e-9)[..., None]
    return dist, unit


# --------------------------------------------------------------------------- #
# Static graphs
# --------------------------------------------------------------------------- #
def distance_graph(dist: np.ndarray, sigma: float | None = None,
                   threshold: float = 0.1) -> np.ndarray:
    """Gaussian kernel on distances. sigma defaults to the std of all pairwise distances."""
    n = len(dist)
    off = ~np.eye(n, dtype=bool)
    sigma = float(sigma) if sigma else float(dist[off].std())
    A = np.exp(-(dist / sigma) ** 2)
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return A


def knn_mask(dist: np.ndarray, k: int) -> np.ndarray:
    """mask[i, j] = True if j is one of the k nearest stations of i (directed, no self)."""
    n = len(dist)
    k = int(min(k, n - 1))
    d = dist + np.diag(np.full(n, np.inf))
    nearest = np.argsort(d, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    mask[np.arange(n)[:, None], nearest] = True
    return mask


def knn_graph(dist: np.ndarray, k: int = 8) -> np.ndarray:
    """Symmetric k-nearest-neighbour graph with weights 1."""
    m = knn_mask(dist, k)
    return (m | m.T).astype(np.float64)


def random_graph(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same graph, station labels shuffled: P A P^T. Keeps every weight, breaks geography."""
    perm = np.random.default_rng(seed).permutation(len(A))
    return A[np.ix_(perm, perm)]


def row_normalize(A: np.ndarray) -> np.ndarray:
    """Each row sums to 1 (a station averages its neighbours); empty rows stay 0."""
    s = A.sum(axis=1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A, dtype=np.float64), where=s > 0)


def build_graph(name: str, dist: np.ndarray, k: int = 8, sigma: float | None = None,
                threshold: float = 0.1, seed: int = 0) -> np.ndarray:
    """Row-normalized (N, N) adjacency of a static graph."""
    if name == "distance":
        A = distance_graph(dist, sigma, threshold)
    elif name == "knn":
        A = knn_graph(dist, k)
    elif name == "random":
        A = random_graph(distance_graph(dist, sigma, threshold), seed)
    elif name == "none":
        A = np.zeros_like(dist)
    else:
        raise ValueError(f"Unknown static graph '{name}'. Choose from {STATIC_GRAPHS}.")
    return row_normalize(A)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def describe(name: str, A: np.ndarray, dist: np.ndarray) -> str:
    edges = A > 0
    n_edges = int(edges.sum())
    deg = edges.sum(axis=1)
    mean_d = float(dist[edges].mean()) if n_edges else float("nan")
    return (f"{name:<9} {n_edges:>6} edges | neighbours per station: mean {deg.mean():4.1f}, "
            f"min {deg.min()}, max {deg.max()} | isolated: {int((deg == 0).sum())} | "
            f"mean edge length {mean_d:5.1f} km")


def plot_graphs(st: pd.DataFrame, graphs: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(graphs), figsize=(5.2 * len(graphs), 5), squeeze=False)
    lon, lat = st["lon"].to_numpy(), st["lat"].to_numpy()
    for ax, (name, A) in zip(axes[0], graphs.items()):
        wmax = A.max() if A.max() > 0 else 1
        for i, j in zip(*np.nonzero(A)):
            ax.plot([lon[j], lon[i]], [lat[j], lat[i]], color="#2a78d6",
                    alpha=0.15 + 0.6 * A[i, j] / wmax, lw=0.8, zorder=1)
        ax.scatter(lon, lat, s=14, color="#eb6834", zorder=2)
        ax.set_title(name, loc="left")
        ax.set_xlabel("longitude")
        ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    axes[0][0].set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser(description="Build and inspect the station graphs.")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--sigma", type=float, default=None, help="km (default: std of distances)")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    st = load_coordinates(args.processed)
    dist, _ = geometry(project_km(st["lat"].to_numpy(), st["lon"].to_numpy()))
    off = ~np.eye(len(dist), dtype=bool)
    print(f"{len(st)} stations | pairwise distances: min {dist[off].min():.1f} km, "
          f"median {np.median(dist[off]):.1f} km, max {dist.max():.1f} km\n")
    graphs = {}
    for name in STATIC_GRAPHS:
        graphs[name] = build_graph(name, dist, args.k, args.sigma, args.threshold)
        print(describe(name, graphs[name], dist))
    wind_candidates = knn_mask(dist, args.k).astype(float)
    print(describe("wind", wind_candidates, dist).replace("edges", "candidate edges") +
          "  (weights depend on the wind of each forecast)")
    if args.plot:
        from dataset import find_processed_dir
        plot_graphs(st, {"distance": graphs["distance"], "knn": graphs["knn"],
                         "random (control)": graphs["random"]},
                    find_processed_dir(args.processed) / "graphs.png")


if __name__ == "__main__":
    main()

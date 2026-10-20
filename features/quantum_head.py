"""
Phase 3 — hybrid photonic quantum heads for the PM2.5 forecaster.

Every model here keeps the project contract:
    input  (B, T, N, F)  ->  output (B, H, N)

Design
------
    x (B,T,N,F)
      -> encoder (your Phase 1/2 GRU or GNN+GRU, returns per-node embeddings (B, N, D))
      -> bottleneck  Linear(D -> k) + bounded activation        (k = 4..8)
      -> HEAD  (quantum or classical, applied per node, nodes folded into batch)
      -> readout Linear(head_out -> H)
      -> (B, H, N)

Heads
-----
    QuantumHead   : k angles -> photonic interferometer (MerLin QuantumLayer) -> Fock probabilities
    ClassicalHead : k -> MLP -> same output width, with a parameter count matched to QuantumHead

Tested against merlinquantum 0.4.1 / perceval-quandela 1.3.1 / torch 2.12.
"""
from __future__ import annotations

import math
from math import comb

import torch
import torch.nn as nn
import merlin as ML


# --------------------------------------------------------------------------
# Quantum head
# --------------------------------------------------------------------------
def build_photonic_circuit(k: int, n_modes: int) -> ML.CircuitBuilder:
    """Trainable interferometer -> angle encoding of k features -> trainable interferometer.

    This "sandwich" (U2 · S(x) · U1) is the standard photonic VQC ansatz. Data are encoded
    as phases; the trainable unitaries are meshes of Mach-Zehnder interferometers.
    """
    b = ML.CircuitBuilder(n_modes=n_modes)
    b.add_entangling_layer(trainable=True, name="U1")
    b.add_angle_encoding(modes=list(range(k)), name="x")  # one phase shifter per feature
    b.add_entangling_layer(trainable=True, name="U2")
    return b


class QuantumHead(nn.Module):
    """k bounded features -> photonic quantum layer -> probability vector.

    Output width = C(m, n) in the unbunched space (m modes, n photons).
    E.g. m=6, n=3 -> 20 outputs; m=8, n=4 -> 70 outputs.
    """

    def __init__(self, k: int = 6, n_modes: int | None = None, n_photons: int | None = None,
                 computation_space=ML.ComputationSpace.UNBUNCHED):
        super().__init__()
        self.k = k
        self.n_modes = n_modes or k              # at least k modes to host k phase shifters
        self.n_photons = n_photons or self.n_modes // 2
        builder = build_photonic_circuit(k, self.n_modes)
        self.qlayer = ML.QuantumLayer(
            input_size=k,
            builder=builder,
            n_photons=self.n_photons,
            measurement_strategy=ML.MeasurementStrategy.probs(computation_space=computation_space),
        )
        self.out_dim = self.qlayer.output_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # z: (M, k) in [0, pi]
        return self.qlayer(z)


# --------------------------------------------------------------------------
# Classical head with matched parameter count (the fair baseline)
# --------------------------------------------------------------------------
class ClassicalHead(nn.Module):
    """k -> hidden -> out_dim (softmax so the output also lives on the probability simplex).

    `hidden` is chosen so the trainable-parameter count matches a target (the quantum head).
    """

    def __init__(self, k: int, out_dim: int, target_params: int, simplex_output: bool = True):
        super().__init__()
        # params = (k+1)*h + (h+1)*out  ->  h = (target - out) / (k + 1 + out)
        h = max(1, round((target_params - out_dim) / (k + 1 + out_dim)))
        self.net = nn.Sequential(nn.Linear(k, h), nn.Tanh(), nn.Linear(h, out_dim))
        self.simplex_output = simplex_output
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        return torch.softmax(y, dim=-1) if self.simplex_output else y


# --------------------------------------------------------------------------
# Full forecaster: encoder + bottleneck + head + readout
# --------------------------------------------------------------------------
class GRUEncoder(nn.Module):
    """Minimal stand-in for your Phase 1/2 encoder: shared GRU per station.

    Replace with your GNN+GRU; it only needs to map (B,T,N,F) -> (B,N,D).
    """

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
        _, h = self.gru(x)                       # (1, B*N, hidden)
        return h[-1].reshape(B, N, -1)


class HybridForecaster(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module, k: int, horizon: int = 24,
                 residual_linear: bool = False):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = nn.Linear(encoder.out_dim, k)
        self.head = head
        self.readout = nn.Linear(head.out_dim, horizon)
        # Optional "skip" path: lets you measure how much the head adds on top of a linear model.
        self.skip = nn.Linear(k, horizon) if residual_linear else None

    def encode_angles(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(self.encoder(x))     # (B, N, k)
        return math.pi * torch.sigmoid(z)        # phases in [0, pi]: bounded, no wrap-around

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x.shape
        z = self.encode_angles(x).reshape(B * N, -1)   # fold nodes into the batch
        y = self.readout(self.head(z))                  # (B*N, H)
        if self.skip is not None:
            y = y + self.skip(z)
        return y.reshape(B, N, -1).permute(0, 2, 1)     # (B, H, N)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def make_pair(n_features: int, k: int = 6, horizon: int = 24, hidden: int = 64,
              n_modes: int | None = None, n_photons: int | None = None, seed: int = 0):
    """Build a (quantum, classical) pair that differ ONLY in the head, with matched head params."""
    torch.manual_seed(seed)
    q_head = QuantumHead(k=k, n_modes=n_modes, n_photons=n_photons)
    c_head = ClassicalHead(k=k, out_dim=q_head.out_dim, target_params=count_params(q_head))

    torch.manual_seed(seed)
    q_model = HybridForecaster(GRUEncoder(n_features, hidden), q_head, k, horizon)
    torch.manual_seed(seed)
    c_model = HybridForecaster(GRUEncoder(n_features, hidden), c_head, k, horizon)
    return q_model, c_model


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    B, T, N, F, H = 8, 72, 40, 12, 24
    x = torch.randn(B, T, N, F)
    y = torch.randn(B, H, N)

    for k, m, n in [(4, 4, 2), (6, 6, 3), (8, 8, 4)]:
        q, c = make_pair(F, k=k, n_modes=m, n_photons=n, horizon=H)
        print(f"\nk={k}, modes={m}, photons={n}  -> head output dim {q.head.out_dim}")
        print(f"  head params   quantum={count_params(q.head):5d}   classical={count_params(c.head):5d}")
        print(f"  total params  quantum={count_params(q):5d}   classical={count_params(c):5d}")
        for name, model in [("quantum", q), ("classical", c)]:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            t0 = time.perf_counter()
            out = model(x)
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            opt.step()
            dt = time.perf_counter() - t0
            assert out.shape == (B, H, N), out.shape
            grads_ok = all(p.grad is not None for p in model.head.parameters())
            print(f"  {name:9s} out {tuple(out.shape)}  loss {loss.item():.3f}  "
                  f"fwd+bwd {dt*1000:6.1f} ms  head grads {'ok' if grads_ok else 'MISSING'}")
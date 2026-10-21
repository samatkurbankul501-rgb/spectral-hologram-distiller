"""Quantify the lattice warm-up window: where must a rare regime appear to be seen?"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller


def build(X, y, M=300, tq=84.0):
    d = SpectralHologramDistiller(crystal_size=M, high_band_ratio=0.26, sketch_dim=256,
                                  tail_quantile=tq, phase_kappa=3.2, random_state=0)
    for i in range(0, len(X), 2500):
        d.partial_fit(X[i:i+2500], y[i:i+2500])
    d.finalize_crystal()
    return d


rng = np.random.default_rng(0)
N, d = 60_000, 6
print(f"stream N={N}, M=300 -> lattice window = max(4500, 12*300) = 4500 rows "
      f"({4500/N:.1%} of the stream)")
print(f"\n{'regime position':>18s} | {'crystal rows in regime':>24s} | {'expected':>9s}")
print("-" * 62)
for frac in (0.0, 0.02, 0.05, 0.075, 0.10, 0.25, 0.5, 0.75, 0.90):
    X = rng.normal(size=(N, d))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
    lo = int(frac * N)
    hi = lo + int(0.05 * N)
    X[lo:hi] += 3.0
    y[lo:hi] += 12.0
    dist = build(X, y)
    captured = int(np.sum((dist.crystal_Y >= 12.0)))
    print(f"{frac:17.1%} | {captured:24d} | {int(300*0.05):9d}")

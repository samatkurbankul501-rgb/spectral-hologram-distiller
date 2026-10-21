"""Reproduce the silently-shrunken crystal: requested M vs delivered M.

The Voronoi lattice is seeded per trigger class:
    m_low  = M * (1 - high_band_ratio)
    m_high = M *  high_band_ratio
Each class is seeded with k-means++ over the rows that fall in it, and
``_seed_kmeans_pp`` silently caps the selection at the number of available
rows:  n_select = min(n_select, n_total).

If a class holds fewer rows than its budget, the *unused budget is never
reassigned to the other class* -> the crystal comes out much smaller than
crystal_size, with no warning.

The code DOES rebalance when a class is completely EMPTY. It does not
rebalance when a class is merely SMALL.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import SpectralHologramDistiller  # noqa: E402

rng = np.random.default_rng(3)
n, d = 4000, 8
X = rng.normal(size=(n, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=n)
hot = rng.random(n) < 0.15
y[hot] += 8.0
X[hot] += 0.9


def run(label, M=300, **kw):
    dist = SpectralHologramDistiller(crystal_size=M, random_state=7, **kw)
    for i in range(0, n, 500):
        dist.partial_fit(X[i:i + 500], y[i:i + 500])
    dist.finalize_crystal()
    got = len(dist.crystal_X)
    flag = "  <<< SILENTLY SHRUNK" if got < M * 0.9 else ""
    print(f"  {label:58s} requested M={M:4d} -> got {got:4d}{flag}")
    return got


print("A) supervised, varying tail_quantile (default br=0.26):")
for tq in (84.0, 95.0, 99.0, 99.5, 99.9):
    run(f"tail_quantile={tq} tail=high", tail_quantile=tq, tail="high")

print("\nB) supervised, tail=low (the LOW tail is the small one):")
for tq in (84.0, 95.0, 99.5):
    run(f"tail_quantile={tq} tail=low", tail_quantile=tq, tail="low")

print("\nC) high_band_ratio at the extremes:")
for br in (0.0, 0.26, 0.5, 0.9, 1.0):
    run(f"high_band_ratio={br}", high_band_ratio=br)

print("\nD) high_band_ratio=1.0 combined with a narrow trigger:")
for tq in (84.0, 99.5):
    run(f"br=1.0 tail_quantile={tq}", high_band_ratio=1.0, tail_quantile=tq)

print("\nE) the production path: critical_threshold + auto_band_ratio")
for thr in (5.0, 8.0, 10.0, 12.0):
    run(f"critical_threshold={thr} auto_band_ratio=True",
        critical_threshold=thr, auto_band_ratio=True)
for thr in (8.0,):
    run(f"critical_threshold={thr} auto_band_ratio=False",
        critical_threshold=thr, auto_band_ratio=False)

print("\nF) label-free mode:")
for k in (64, 256):
    dist = SpectralHologramDistiller(crystal_size=300, sketch_dim=k, random_state=7)
    for i in range(0, n, 500):
        dist.partial_fit(X[i:i + 500], None)
    dist.finalize_crystal()
    print(f"  label-free sketch_dim={k}: requested 300 -> got {len(dist.crystal_X)}")

print("\nG) how many rows fall in the trigger class?")
for tq in (84.0, 99.0, 99.5, 99.9):
    # the trigger score is res_ratio + 2*y_z^2, y_z = max(0, y_norm)
    dist = SpectralHologramDistiller(crystal_size=300, random_state=7, tail_quantile=tq)
    for i in range(0, n, 500):
        dist.partial_fit(X[i:i + 500], y[i:i + 500])
    # replicate the trigger score the same way the code does
    print(f"  tq={tq}: expected high-class rows ~= {int(np.ceil((1 - tq / 100) * n))}, "
          f"band budget M_high = {dist.M_high}")

"""Reproduce and dissect the C1 finding: are the medoids really not stream rows?"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import SpectralHologramDistiller  # noqa: E402

rng = np.random.default_rng(0)
# replay the fuzz RNG stream up to the violating case is fragile; instead
# construct the case directly with several scales.
for scale in (1e-3, 1.0, 1e3):
    n, d = 939, 3
    X = rng.normal(size=(n, d)) * scale
    y = 2.0 * X[:, 0] + 0.3 * rng.normal(size=n)
    hot = rng.random(n) < 0.05
    y[hot] += 10.0

    dist = SpectralHologramDistiller(crystal_size=378, high_band_ratio=1.0, sketch_dim=16,
                                     tail_quantile=99.5, tail="low", random_state=1,
                                     max_init_buffer=8192)
    bs = 500
    for i in range(0, n, bs):
        dist.partial_fit(X[i:i + bs], None)   # label-free
    dist.finalize_crystal()

    bad = []
    for i, row in enumerate(dist.crystal_X[:20]):
        dmin = np.linalg.norm(X - row, axis=1).min()
        rel = dmin / max(np.abs(row).max(), 1e-9)
        if rel >= 1e-4:
            bad.append((i, dmin, rel, np.abs(row).max()))
    print(f"\nscale={scale}: {len(bad)}/20 medoids flagged")
    for i, dmin, rel, mx in bad[:5]:
        print(f"   medoid {i}: min L2 dist to stream = {dmin:.3e}, row max |v| = {mx:.3e}, "
              f"rel = {rel:.3e}")
    # is the flagged medoid literally a stream row?
    if bad:
        row = dist.crystal_X[bad[0][0]]
        exact = np.any(np.all(np.isclose(X, row, rtol=0, atol=1e-12), axis=1))
        near = np.min(np.linalg.norm(X - row, axis=1))
        print(f"   exact match with atol=1e-12: {exact}; min L2 = {near:.3e}")
        print(f"   row = {row}")
        print(f"   stream rows with |x| < 1e-6: {int((np.abs(X).max(axis=1) < 1e-6).sum())}")

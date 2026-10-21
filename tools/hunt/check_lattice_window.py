"""Does the Voronoi lattice see the WHOLE stream, or only the warm-up buffer?

partial_fit() buffers rows until the lattice is built:
    target_buf = min(cap, max(4500, M * 12))
    ...
    if buffered + B >= target_buf:  self._build_phase_manifold_lattice()

so the lattice is built from the FIRST ~max(4500, 12*M) rows (capped at
max_init_buffer=8192) and never rebuilt. If a regime appears LATE in the
stream, it gets no dedicated cells.

Test: a 200k-row stream where a rare regime occupies rows 100k..200k.
Does the crystal capture it?
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller

rng = np.random.default_rng(0)
N = 200_000
d = 6
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)

# rare regime occupies the LAST 10% of the stream
late = np.arange(N) >= int(0.9 * N)
X[late] += 3.0
y[late] += 12.0
crit = 12.0

for M in (300,):
    for tq in (84.0, 97.0):
        dist = SpectralHologramDistiller(crystal_size=M, high_band_ratio=0.26,
                                         sketch_dim=256, tail_quantile=tq,
                                         phase_kappa=3.2, random_state=0)
        for i in range(0, N, 2500):
            dist.partial_fit(X[i:i+2500], y[i:i+2500])
        dist.finalize_crystal()
        captured = int(np.sum(dist.crystal_Y >= crit))
        expected = M * 0.10          # 10% of the stream is critical
        print(f"M={M} tail_quantile={tq}: crystal critical rows = {captured} "
              f"(expected ~{expected:.0f} if the lattice saw the whole stream)")

# same stream, but the rare regime is EARLY
X2 = X.copy(); y2 = y.copy()
early = np.arange(N) < int(0.1 * N)
X2[early] -= 3.0; y2[early] -= 12.0   # make the early part critical instead
for tq in (84.0, 97.0):
    dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                                     sketch_dim=256, tail_quantile=tq,
                                     phase_kappa=3.2, random_state=0)
    for i in range(0, N, 2500):
        dist.partial_fit(X2[i:i+2500], y2[i:i+2500])
    dist.finalize_crystal()
    captured = int(np.sum(dist.crystal_Y <= -12.0))
    print(f"M=300 tail_quantile={tq}: EARLY-regime crystal rows = {captured}")

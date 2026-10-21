"""Two possible causes of the miss: POSITION (lattice warm-up window) or
MAGNITUDE (the regime sits outside the frozen y-scale the phase geometry uses).

Grid: position of the regime x magnitude of its y offset.
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller

rng = np.random.default_rng(3)
N, d = 60_000, 6
POSITIONS = [0.0, 0.10, 0.50, 0.90]
MAGNS = [2.0, 4.0, 8.0, 16.0]

print("captured crystal rows inside the regime (out of ~15 expected)")
print(f"{'pos':>6s} " + " ".join(f"{'y+%g' % m:>10s}" for m in MAGNS))
print("-" * 52)
for frac in POSITIONS:
    row = []
    for mag in MAGNS:
        X = rng.normal(size=(N, d))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
        lo = int(frac * N); hi = lo + int(0.05 * N)
        X[lo:hi] += 1.0
        y[lo:hi] += mag
        dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                                         sketch_dim=256, tail_quantile=84.0,
                                         phase_kappa=3.2, random_state=0)
        for i in range(0, N, 2500):
            dist.partial_fit(X[i:i+2500], y[i:i+2500])
        dist.finalize_crystal()
        row.append(int(np.sum(dist.crystal_Y >= mag - 1e-9)))
    print(f"{frac:6.0%} " + " ".join(f"{v:10d}" for v in row), flush=True)

print("\ncontrol: position 0%, magnitude +4 -> captured above")

"""The 'anytime mode' promise: training slices of the first k medoids.

finalize_crystal() sorts the medoids by cell occupancy DESCENDING:
    order = np.argsort(self.counts)[::-1]
so get_training_data(budget_points=k) returns the k MOST POPULOUS cells --
a systematic bias toward the dense background, not a representative prefix.

Test: how does the critical-regime share of a prefix compare to the full
crystal as k grows?
"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="any_")
rng = np.random.default_rng(31)
N, d = 17010, 81
X = rng.normal(size=(N, d))
y = 3.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
hot = rng.random(N) < 0.186
y[hot] += 20.0; X[hot] += 0.8
crit = 20.0

dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, N, 2500):
    dist.partial_fit(X[i:i+2500], y[i:i+2500])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz"); dist.save_hologram(p)
ld = HologramCrystalLoader(p)

full = ld.y
print("full crystal (300 medoids): %d critical (%.1f%%), stream critical share %.1f%%"
      % (int((full >= crit).sum()), 100 * (full >= crit).mean(), 100 * hot.mean()))
print()
print("%6s | %9s | %9s | %s" % ("budget", "critical", "share", "comment"))
print("-" * 62)
for k in (25, 50, 100, 150, 200, 250, 300):
    Xc, yc, wc = ld.get_training_data(budget_points=k)
    nc = int((yc >= crit).sum())
    print("%6d | %9d | %8.1f%% | %s"
          % (k, nc, 100 * (yc >= crit).mean(),
             "<= the first 25 medoids contain ZERO critical rows" if nc == 0 else ""))

# Is the ordering really by occupancy?
print()
w = ld.weights
print("gamma weight of the first 25 medoids: mean %.3f (last 25: mean %.3f)"
      % (w[:25].mean(), w[-25:].mean()))
print("=> the prefix is the dense-background end of the crystal")

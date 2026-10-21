"""The feedback loop: a late regime poisons the radar, the radar fires on NORMAL
data, an operator (or an auto-heal loop) heals on that normal data -- and the
crystal grows without ever capturing the regime it was supposed to learn."""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="fb_")
rng = np.random.default_rng(11)
N, d = 40_000, 6
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
lo = int(0.9 * N)
X_normal = X[:lo].copy(); X_shift = X[lo:].copy() + 3.0
y_normal = y[:lo].copy(); y_shift = y[lo:].copy() + 12.0
X_full = np.vstack([X_normal, X_shift]); y_full = np.concatenate([y_normal, y_shift])

dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, N, 2500):
    dist.partial_fit(X_full[i:i+2500], y_full[i:i+2500])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz"); dist.save_hologram(p)
ld = HologramCrystalLoader(p)
print("start: crystal=%d  rows in shift regime=%d" % (len(ld.X), int(np.sum(ld.y >= 12.0))))
r = ld.check_domain_shift(X_normal[:512])
print("radar on NORMAL data: BW=%.4f shifted=%s" % (r["bw_distance"], r["is_shifted"]))

for round_ in range(1, 4):
    res = ld.heal_domain_shift(X_normal[:512], y_normal[:512], n_inject=35, random_state=round_)
    r = ld.check_domain_shift(X_normal[:512])
    print("  after auto-heal round %d: injected=%d crystal=%d rows_in_shift=%d "
          "| normal BW=%.4f shifted=%s"
          % (round_, res["n_injected"], len(ld.X), int(np.sum(ld.y >= 12.0)),
             r["bw_distance"], r["is_shifted"]))

# how many DISTINCT rows are in the crystal now?
print("distinct crystal rows: %d of %d" % (len(np.unique(np.round(ld.X, 9), axis=0)), len(ld.X)))

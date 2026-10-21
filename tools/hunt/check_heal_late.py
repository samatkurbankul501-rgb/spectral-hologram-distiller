"""Can the heal path recover a late regime that the distill path missed?"""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="hl_")
rng = np.random.default_rng(13)
N, d = 60_000, 6
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
lo = int(0.9 * N)
X_normal = X[:lo].copy()
X_shift = X[lo:].copy() + 3.0
y_normal = y[:lo].copy()
y_shift = y[lo:].copy() + 12.0
X_full = np.vstack([X_normal, X_shift])
y_full = np.concatenate([y_normal, y_shift])

dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, N, 2500):
    dist.partial_fit(X_full[i:i+2500], y_full[i:i+2500])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz"); dist.save_hologram(p)
ld = HologramCrystalLoader(p)

print("BEFORE heal: crystal=%d rows_in_shift=%d" % (len(ld.X), int(np.sum(ld.y >= 12.0))))
r = ld.check_domain_shift(X_shift[:512])
print("radar on the SHIFT window: BW=%.4f shifted=%s" % (r["bw_distance"], r["is_shifted"]))
res = ld.heal_domain_shift(X_shift[:512], y_shift[:512], n_inject=35, random_state=0)
print("heal on the SHIFT window: healed=%s injected=%d bw_before=%.4f bw_after=%.4f"
      % (res["healed"], res["n_injected"], res["bw_before"], res["bw_after"]))
print("AFTER heal: crystal=%d rows_in_shift=%d" % (len(ld.X), int(np.sum(ld.y >= 12.0))))
r2 = ld.check_domain_shift(X_normal[:512])
print("radar on the NORMAL window after heal: BW=%.4f shifted=%s"
      % (r2["bw_distance"], r2["is_shifted"]))

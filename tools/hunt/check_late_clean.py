"""Clean confirmation: after a late regime, the crystal misses it AND the radar
false-alarms on the crystal's OWN in-distribution data."""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="clean_")
rng = np.random.default_rng(1)
N, d = 60_000, 6
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
late_lo = int(0.90 * N)
X_normal = X[:late_lo].copy()          # rows 0..90% : the ORIGINAL regime
X_late = X[late_lo:].copy()            # rows 90..100%: the NEW regime
X_late += 3.0
y_normal = y[:late_lo].copy()
y_late = y[late_lo:].copy() + 12.0

X_full = np.vstack([X_normal, X_late])
y_full = np.concatenate([y_normal, y_late])

dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, N, 2500):
    dist.partial_fit(X_full[i:i+2500], y_full[i:i+2500])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz")
dist.save_hologram(p)
ld = HologramCrystalLoader(p)

print(f"crystal rows inside the LATE regime : {int(np.sum(ld.y >= 12.0))}  (expected 15)")
print(f"crystal rows inside the NORMAL regime: {int(np.sum(ld.y < 12.0))}")

print("\nradar false-alarm rate on the crystal's OWN training rows (normal regime):")
fa = 0
for i in range(0, 20000, 512):
    w = X_normal[i:i+512]
    r = ld.check_domain_shift(w)
    fa += int(r["is_shifted"])
n = 0
for i in range(0, 20000, 512):
    n += 1
print(f"  windows flagged as shifted: {fa}/{n}  (BW ~ {r['bw_distance']:.4f} vs thr {r['threshold']:.4f})")

print("\nradar on the LATE regime:")
r2 = ld.check_domain_shift(X_late[:512])
print(f"  BW={r2['bw_distance']:.4f} thr={r2['threshold']:.4f} shifted={r2['is_shifted']}")

print("\n=> separation between normal and late windows: "
      f"{abs(r2['bw_distance'] - r['bw_distance']):.4f} "
      f"(the threshold is {r['threshold']:.4f})")

print("\nCONTROL: same stream WITHOUT the late regime (pure normal):")
dist2 = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                  tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, len(X_normal), 2500):
    dist2.partial_fit(X_normal[i:i+2500], y_normal[i:i+2500])
dist2.finalize_crystal()
p2 = os.path.join(tmp, "c2.npz")
dist2.save_hologram(p2)
ld2 = HologramCrystalLoader(p2)
fa2 = 0; n2 = 0; last = None
for i in range(0, 20000, 512):
    r3 = ld2.check_domain_shift(X_normal[i:i+512])
    fa2 += int(r3["is_shifted"]); n2 += 1; last = r3
print(f"  windows flagged as shifted: {fa2}/{n2} (BW ~ {last['bw_distance']:.4f})")

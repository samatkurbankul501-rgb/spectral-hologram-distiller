"""If a rare regime arrives late, can the radar see it -- and can heal fix it?"""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="late_")
rng = np.random.default_rng(0)
N, d = 60_000, 6
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
lo, hi = int(0.9 * N), N
Xlate = X[lo:hi].copy()
ylate = y[lo:hi].copy()
X[lo:hi] += 3.0
y[lo:hi] += 12.0

dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, N, 2500):
    dist.partial_fit(X[i:i+2500], y[i:i+2500])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz")
dist.save_hologram(p)
ld = HologramCrystalLoader(p)

print(f"crystal rows inside the late regime: {int(np.sum(ld.y >= 12.0))} (expected 15)")
rep_base = ld.check_domain_shift(X[:512])
rep_late = ld.check_domain_shift(Xlate[:512])
print(f"\nradar on normal rows      : BW={rep_base['bw_distance']:.5f} thr={rep_base['threshold']:.5f} shifted={rep_base['is_shifted']}")
print(f"radar on the LATE regime  : BW={rep_late['bw_distance']:.5f} thr={rep_late['threshold']:.5f} shifted={rep_late['is_shifted']}")
print(f"\n=> the radar DOES fire on the late regime: {rep_late['is_shifted']}")

res = ld.heal_domain_shift(Xlate[:512], ylate[:512], n_inject=35, random_state=0)
print(f"heal: healed={res['healed']} injected={res['n_injected']} "
      f"bw_before={res['bw_before']:.5f} bw_after={res['bw_after']:.5f}")
print(f"crystal rows inside the late regime after heal: {int(np.sum(ld.y >= 12.0))}")

# and the reverse: is the radar blind if the regime is only 1% of the stream?
print("\n--- sensitivity: how late can the regime be and still be detectable? ---")
for frac in (0.01, 0.02, 0.05, 0.10, 0.25):
    X2 = rng.normal(size=(N, d))
    y2 = 2.0 * X2[:, 0] + 0.5 * X2[:, 1] * X2[:, 2] + 0.05 * rng.normal(size=N)
    lo2 = int(frac * N)
    hi2 = lo2 + int(0.05 * N)
    Xl = X2[lo2:hi2].copy(); yl = y2[lo2:hi2].copy()
    X2[lo2:hi2] += 3.0; y2[lo2:hi2] += 12.0
    dd = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                   tail_quantile=84.0, phase_kappa=3.2, random_state=0)
    for i in range(0, N, 2500):
        dd.partial_fit(X2[i:i+2500], y2[i:i+2500])
    dd.finalize_crystal()
    pp = os.path.join(tmp, f"c{frac}.npz")
    dd.save_hologram(pp)
    l2 = HologramCrystalLoader(pp)
    in_regime = int(np.sum(l2.y >= 12.0))
    r = l2.check_domain_shift(Xl[:512])
    print(f"  regime starts at {frac:5.0%}: crystal rows in regime={in_regime:3d}  "
          f"radar BW={r['bw_distance']:.5f} shifted={r['is_shifted']}")

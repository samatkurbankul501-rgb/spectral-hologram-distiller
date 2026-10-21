"""How does a shifted regime inside the training stream poison the radar?"""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="fa_")
rng = np.random.default_rng(7)
N, d = 40_000, 6


def run(shift_frac, shift_amp=3.0):
    X = rng.normal(size=(N, d))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
    lo = int((1.0 - shift_frac) * N)
    Xn = X[:lo].copy(); Xs = X[lo:].copy()
    Xs += shift_amp
    Xf = np.vstack([Xn, Xs]); yf = np.concatenate([y[:lo], y[lo:] + 12.0])

    dist = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                     tail_quantile=84.0, phase_kappa=3.2, random_state=0)
    for i in range(0, N, 2500):
        dist.partial_fit(Xf[i:i+2500], yf[i:i+2500])
    dist.finalize_crystal()
    p = os.path.join(tmp, f"c{shift_frac}.npz")
    dist.save_hologram(p)
    ld = HologramCrystalLoader(p)
    fa = 0; n = 0; last = None
    for i in range(0, min(20000, lo - 512), 512):
        r = ld.check_domain_shift(Xn[i:i+512])
        fa += int(r["is_shifted"]); n += 1; last = r
    in_shift = int(np.sum(ld.y >= 12.0))
    return fa, n, last["bw_distance"], last["threshold"], in_shift


print(f"{'shift frac of stream':>22s} | {'FA rate on normal windows':>26s} | {'BW':>8s} | {'thr':>7s} | {'crystal rows in shift':>22s}")
print("-" * 105)
for f in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
    fa, n, bw, thr, ins = run(f)
    print(f"{f:21.0%} | {fa:12d}/{n:<3d} ({fa/n:6.0%}) | {bw:8.4f} | {thr:7.4f} | {ins:22d}")

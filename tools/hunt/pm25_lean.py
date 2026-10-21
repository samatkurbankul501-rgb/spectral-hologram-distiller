"""Lean PM2.5 check: does the delivered crystal size shrink at tq=97, and what
does that do to the rare_captured comparison? (local 5,937-row copy)"""
from __future__ import annotations
import csv, os, sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
rows = list(csv.DictReader(open(os.path.join(REPO, "data", "pm25.csv"), newline="", encoding="utf-8")))
cb = ["NW", "NE", "SE", "cv"]
data = []
for r in rows:
    try:
        y = float(r["pm2.5"])
        nums = [float(r[k]) for k in ("year","month","day","hour","DEWP","TEMP","PRES","Iws","Is","Ir")]
    except Exception:
        continue
    if r["cbwd"] not in cb: continue
    data.append(nums + [1.0 if r["cbwd"] == v else 0.0 for v in cb] + [y])
a = np.asarray(data, float); X, y = a[:, :-1], a[:, -1]
n_tr = int(round(0.8 * len(X)))
X_tr, y_tr = X[:n_tr], y[:n_tr]
crit = float(np.percentile(y_tr, 97.0))
print(f"local PM2.5 n={len(X)} train={n_tr} crit_thr={crit:.1f}", flush=True)
M = 300
for label, tq, br in (("default tq=84 br=.26", 84.0, 0.26),
                      ("published tq=97 br=.26", 97.0, 0.26),
                      ("tq=97 br=0.50", 97.0, 0.50)):
    got, rare = [], []
    for seed in range(3):
        d = SpectralHologramDistiller(crystal_size=M, high_band_ratio=br, sketch_dim=256,
                                     tail_quantile=tq, phase_kappa=3.2, random_state=seed)
        for i in range(0, n_tr, 2500):
            d.partial_fit(X_tr[i:i+2500], y_tr[i:i+2500])
        d.finalize_crystal()
        got.append(len(d.crystal_X)); rare.append(int(np.sum(d.crystal_Y >= crit)))
    print(f"  {label:26s} delivered={got}  rare_captured={rare}  "
          f"(requested {M})", flush=True)
print(f"\nstream critical rows in train = {int((y_tr >= crit).sum())} "
      f"({(y_tr >= crit).mean():.2%}); band budget M_high = {round(M*0.26)}", flush=True)

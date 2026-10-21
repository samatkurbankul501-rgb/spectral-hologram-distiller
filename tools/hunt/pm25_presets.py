"""Do the README section 6 presets deliver the budget they promise?"""
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
n_crit = int((y_tr >= crit).sum())
print(f"PM2.5 local: train={n_tr}, critical (top 3%) = {n_crit} rows\n")
print("README section 6 presets at M=300:")
PRESETS = [("global accuracy first", 0.26, 84.0),
            ("balanced", 0.50, 84.0),
            ("critical / rare-event first", 0.65, 84.0),
            ("sparse stream (top-3% regime)", 0.26, 97.0)]
for name, br, tq in PRESETS:
    got = []
    for seed in range(3):
        d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=br, sketch_dim=256,
                                     tail_quantile=tq, phase_kappa=3.2, random_state=seed)
        for i in range(0, n_tr, 2500):
            d.partial_fit(X_tr[i:i+2500], y_tr[i:i+2500])
        d.finalize_crystal()
        got.append(len(d.crystal_X))
    g = int(np.median(got))
    m_high = round(300 * br)
    flag = "" if g == 300 else f"  <<< SHORT BY {300 - g} (band budget {m_high} > {n_crit} critical rows)"
    print(f"  {name:32s} br={br:.2f} tq={tq:5.1f} -> delivered {got}{flag}")

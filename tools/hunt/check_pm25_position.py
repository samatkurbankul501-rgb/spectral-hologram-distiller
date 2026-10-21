"""Is the PM2.5 critical regime inside or outside the lattice warm-up window?

PM2.5 is the one time-ordered dataset in the suite, so this is where the
warm-up-window defect could actually bite a published number.
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")

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
print("PM2.5 local copy: train=%d, crit_thr=%.1f" % (n_tr, crit))
print("lattice window (M=300) = max(4500, 12*300) = 4500 rows = %.1f%% of the train window"
      % (100 * 4500 / n_tr))
in_win = np.zeros(n_tr, dtype=bool); in_win[:4500] = True
crit_mask = y_tr >= crit
print("critical rows inside the lattice window : %d of %d (%.1f%%)"
      % (int((crit_mask & in_win).sum()), int(crit_mask.sum()),
         100 * (crit_mask & in_win).sum() / max(crit_mask.sum(), 1)))
print("critical rows AFTER the lattice window  : %d"
      % int((crit_mask & ~in_win).sum()))
q = np.argsort(-y_tr)[:int(crit_mask.sum())]
print("median position of the critical rows in the train window: %.0f%%"
      % (100 * np.median(q) / n_tr))

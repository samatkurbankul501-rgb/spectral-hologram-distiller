"""Instrument _res_peak_replace to see which slots get replaced."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from shd import SpectralHologramDistiller
from shd.distiller import SpectralHologramDistiller as _SD

CAP = 2000
peak_cap = max(1, int(CAP * 0.4))

d = SpectralHologramDistiller(crystal_size=120, high_band_ratio=0.26,
                               lattice_rebuild_interval=1500,
                               max_init_buffer=CAP)

# Instrument the replacement: record every (j, batch, slot) hit.
orig_replace = d._res_peak_replace
hits = []
def traced_replace(j, z_row, x_row, y_row, u_row, s_row):
    # figure out which batch/slot j maps to (same logic as _res_peak_replace)
    c = 0
    for bi, arr in enumerate(d.res_peak_Z):
        n = len(arr)
        if j < c + n:
            hits.append((j, bi, j - c))
            break
        c += n
    return orig_replace(j, z_row, x_row, y_row, u_row, s_row)
d._res_peak_replace = traced_replace

rng = np.random.default_rng(11)
X_n = rng.normal(size=(3000, 4))
y_n = rng.normal(size=3000)
for i in range(0, 3000, 500):
    d.partial_fit(X_n[i:i + 500], y_n[i:i + 500])

X_d = rng.normal(size=(6000, 4))
y_d = rng.normal(size=6000) + 30.0
for i in range(0, 6000, 500):
    d.partial_fit(X_d[i:i + 500], y_d[i:i + 500])

PY = np.concatenate(d.res_peak_Y)
n_p1 = int(np.sum(PY < 29.0))
print(f"total replacement calls: {len(hits)}")
# which batches/slots were hit
from collections import Counter
batch_hits = Counter(h[1] for h in hits)
print("replacements per batch:", dict(batch_hits))
print(f"pre-drift retained: {n_p1}  (Vitter expectation ~{136*peak_cap/max(d._res_peak_seen,1):.1f})")

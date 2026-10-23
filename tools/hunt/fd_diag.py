"""Diagnostic: how many drift rows survive in the peak stratum vs true Vitter."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from shd import SpectralHologramDistiller

CAP = 2000
rng = np.random.default_rng(11)
d = SpectralHologramDistiller(crystal_size=120, high_band_ratio=0.26,
                               lattice_rebuild_interval=1500,
                               max_init_buffer=CAP)

X_n = rng.normal(size=(3000, 4))
y_n = rng.normal(size=3000)
for i in range(0, 3000, 500):
    d.partial_fit(X_n[i:i + 500], y_n[i:i + 500])

X_d = rng.normal(size=(6000, 4))
y_d = rng.normal(size=6000) + 30.0
for i in range(0, 6000, 500):
    d.partial_fit(X_d[i:i + 500], y_d[i:i + 500])

peak_cap = max(1, int(CAP * 0.4))
PY = np.concatenate(d.res_peak_Y)
n_drift = int(np.sum(PY > 29.0))
n_pre = len(PY) - n_drift
print(f"peak_cap={peak_cap} retained={d._res_peak_rows} seen={d._res_peak_seen}")
print(f"pre-drift rows in stratum: {n_pre} | drift rows: {n_drift}")
exp_pre = peak_cap * 136 / d._res_peak_seen
exp_drift = peak_cap * (d._res_peak_seen - 136) / d._res_peak_seen
print(f"true-Vitter expectation: pre-drift ~{exp_pre:.1f}, drift ~{exp_drift:.1f}")
print("per-batch structure of peak stratum:")
for bi, arr in enumerate(d.res_peak_Y):
    a = np.asarray(arr).reshape(-1)
    print(f"  batch {bi}: {len(a)} rows, {int(np.sum(a > 29.0))} drift")

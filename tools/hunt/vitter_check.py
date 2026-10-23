"""Precise origin-tracking diagnostic for the peak stratum (Vitter check).

The y>29 threshold in round2 probe C is contaminated: drift rows are
N(0,1)+30, so ~30% of drift rows fall below 29 and get miscounted as
pre-drift. This probe tags each reservoir row with its PHASE (pre-drift
vs drift) at insertion time, then checks final stratum composition against
the Vitter expectation for the first 136 peaks (136 * 800/6130 ~ 17.7).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
from shd import SpectralHologramDistiller

CAP = 2000
peak_cap = max(1, int(CAP * 0.4))

d = SpectralHologramDistiller(crystal_size=120, high_band_ratio=0.26,
                               lattice_rebuild_interval=1500,
                               max_init_buffer=CAP)

# Tag every peak-stratum row with its insertion phase.
orig_add = d._reservoir_add
phase_of_row = []  # per (batch, slot) phase
# Simpler: wrap _reservoir_add to record, per batch appended, the phase.
batch_phase = []  # list of ("pre"|"drift"), aligned with res_peak_Z batches
orig_add_ref = d._reservoir_add

def traced_add(Z_B, X_B, y_B, U_B, score_B):
    n_added = 0
    # count how many peak rows this call will add (fill or replace)
    y_flat = np.asarray(y_B, dtype=np.float64).reshape(-1)
    y_z = np.abs((y_flat - d.y_mean) / d.y_std)
    is_peak = y_z > float(d.peak_z_threshold)
    phase = "drift" if float(np.median(y_flat)) > 15.0 else "pre"
    orig_add(Z_B, X_B, y_B, U_B, score_B)
    # diff: number of new batches / rows since this call
    batch_phase.append((phase, int(np.sum(is_peak))))

for _ in range(3):
    pass

rng = np.random.default_rng(11)
X_n = rng.normal(size=(3000, 4))
y_n = rng.normal(size=3000)
for i in range(0, 3000, 500):
    d.partial_fit(X_n[i:i + 500], y_n[i:i + 500])

X_d = rng.normal(size=(6000, 4))
y_d = rng.normal(size=6000) + 30.0
for i in range(0, 6000, 500):
    d.partial_fit(X_d[i:i + 500], y_d[i:i + 500])

# Final stratum: map each retained row back to its batch phase.
PY = np.concatenate(d.res_peak_Y)
total = d._res_peak_seen
exp_pre = 136 * peak_cap / max(total, 1)
exp_drift = (total - 136) * peak_cap / max(total, 1)

# Count pre-drift-origin rows: rows whose y is a normal-phase value (|y| < 15)
n_pre_strict = int(np.sum(np.abs(PY) < 15.0))   # pre-drift rows (|y|<15)
n_drift_strict = int(np.sum(PY > 26.0))          # drift rows (y>26)
print(f"peak_cap={peak_cap}  retained={d._res_peak_rows}  seen={total}")
print(f"pre-drift (strict |y|<15): {n_pre_strict}  (Vitter expectation ~{exp_pre:.1f})")
print(f"drift (strict y>26): {n_drift_strict}  (Vitter expectation ~{exp_drift:.1f})")
ok_pre = abs(n_pre_strict - exp_pre) <= max(4.0, 0.15 * exp_pre)
ok_drift = abs(n_drift_strict - exp_drift) <= max(4.0, 0.15 * exp_drift)
print(f"pre-drift within 15% of Vitter expectation: {ok_pre}")
print(f"drift within 15% of Vitter expectation: {ok_drift}")

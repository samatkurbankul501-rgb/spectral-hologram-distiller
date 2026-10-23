"""Diagnose why the v3.13.0 combined-radar tau blows up on MetroPT-3.

Computes per-feature mean_shift_distance over the calib slice and reports which
feature dominates, and the X_std values stored in the crystal.
"""
from __future__ import annotations

import sys

import numpy as np

ROOT = r"C:\Users\User\Downloads\New folder\_release\spectral-hologram-distiller"
sys.path.insert(0, ROOT)

from shd import HologramCrystalLoader
from shd_watch import load_minute_table, slice_range

NPZ = r"C:\Users\User\Downloads\New folder\_datasets\metro_minute_cache.npz"
LEARN = ("2020-02-01", "2020-03-15")
CALIB = ("2020-03-15", "2020-04-10")

minutes, X, feats = load_minute_table(NPZ)
c_lo, c_hi = slice_range(minutes, *CALIB)

loader = HologramCrystalLoader(ROOT + r"\metro_demo_v2_crystal.hologram.npz")

print("=== crystal X_std (stored) ===")
for f, sd in zip(feats, loader.X_std):
    flag = "  <-- near-constant" if sd < 1.0 else ""
    print(f"  {f:<18s} std={sd:.4f}{flag}")

# Per-feature max mean-shift over the calib slice (window 60 / step 30)
WIN, STEP = 60, 30
sd = loader.X_std
mu = loader.X_mean
ok = sd > 1e-9
max_z = np.zeros(len(feats))
for lo in range(c_lo, c_hi - WIN, STEP):
    w = X[lo:lo + WIN]
    z = np.abs(w.mean(axis=0) - mu) / np.where(ok, sd, 1.0)
    max_z = np.maximum(max_z, z[ok] if ok.any() else 0.0)

print("\n=== per-feature max mean_shift (calib, window 60/step 30) ===")
for f, z in zip(feats, max_z):
    print(f"  {f:<18s} {z:12.4f}")

print("\n=== combined_distance on first calib window ===")
lo, hi = c_lo, min(c_lo + WIN, c_hi)
rep = loader.check_domain_shift(X[lo:hi], mean_shift=True)
print("  bw={:.4f} ms={:.4f} combined={:.4f} (ms_threshold_source={})".format(
    rep["bw_distance"], rep["mean_shift_distance"],
    rep["combined_distance"], rep["combined_threshold_source"]))

"""Round-2 defect hunt probes (written 2026-10-21, fixed in v3.12.0).

The 2026-10-21 v3.11.1 tree was probed for three suspects after the v3.11.0
code review; two reproduced as real defects and one did not:

* A (CONFIRMED -> fixed): ``tail="both"`` + ``critical_threshold`` built the
  critical band from ``y >= threshold`` (the UPPER target side), but the
  peak slice re-ranked the band by extremeness-from-median and served
  LOW-side rows the band was never built from (7 of 18 slice rows below the
  threshold). v3.12.0 persists ``critical_threshold`` in the artifact and
  the slice / budget / heal protection all read the same definition.
* B (not reproduced): ``budget_points`` + ``drop_unlabelled`` shortfall --
  the budget returned exactly the requested rows on every probe seed
  (peak-first selection includes placeholder rows; the drop happens after).
  Kept here so the non-defect stays re-checkable.
* C (CONFIRMED -> fixed): the rebuild reservoir's peak stratum kept the
  FIRST ``peak_cap`` peaks and silently dropped every later one, so a
  target drift that pushed most of the stream above the trigger filled the
  stratum with the first drifted batches and lost the late drift
  (544 of 6,000 drifted rows retained). v3.12.0 samples peaks uniformly
  over the whole stream (Vitter's algorithm R).

Run from anywhere: ``python tools/hunt/round2_probes.py``. Regression tests
for the two confirmed defects live in ``tests/test_v312_repairs.py``.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from shd.distiller import HologramCrystalLoader, SpectralHologramDistiller


def probe_a():
    """critical_threshold + tail='both': the slice must stay above y >= thr."""
    print("=" * 74)
    print("PROBE A: critical_threshold + tail='both' coherence")
    print("=" * 74)
    rng = np.random.default_rng(20261021)
    X = rng.uniform(-2, 2, size=(5000, 4))
    y = rng.normal(0, 1, size=5000)
    y[rng.random(5000) < 0.05] += 5.0
    d = SpectralHologramDistiller(crystal_size=60, high_band_ratio=0.10, tail="both",
                                  critical_threshold=1.0, max_init_buffer=2000)
    for i in range(0, 5000, 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    fd, path = tempfile.mkstemp(suffix=".hologram.npz")
    os.close(fd)
    try:
        d.save_hologram(path, include_projector=True)
        loader = HologramCrystalLoader(path)
        _, yp, _ = loader.get_phase_peak_slice(high_band_ratio=0.30)
        n_below = int(np.sum(yp < 1.0))
        print(f"peak slice rows={len(yp)}, of which BELOW the threshold: {n_below}")
        ok = len(yp) > 0 and n_below == 0
        print("coherence:", "RESTORED (defect fixed)" if ok else "STILL BROKEN")
        return ok
    finally:
        os.remove(path)


def probe_b():
    """budget_points + drop_unlabelled: the budget must not come up short."""
    print("=" * 74)
    print("PROBE B: budget_points + drop_unlabelled shortfall")
    print("=" * 74)
    rng = np.random.default_rng(20261022)
    X = rng.normal(size=(4000, 4))
    y = rng.normal(size=4000)
    d = SpectralHologramDistiller(crystal_size=100, high_band_ratio=0.26,
                                  max_init_buffer=4000)
    for i in range(0, 4000, 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    fd, path = tempfile.mkstemp(suffix=".hologram.npz")
    os.close(fd)
    try:
        d.save_hologram(path, include_projector=True)
        loader = HologramCrystalLoader(path)
        # Label-free heal injects placeholder-target rows (marked unlabelled).
        loader.heal_domain_shift(rng.normal(size=(200, 4)) + 6.0, n_inject=30,
                                 force=True)
        n_unlab = int(np.sum(~loader.y_labelled))
        budget = 60
        Xb, yb, _ = loader.get_training_data(budget_points=budget, drop_unlabelled=True)
        print(f"crystal rows={len(loader.X)}, unlabelled (median placeholders)={n_unlab}")
        print(f"requested budget={budget}, received {len(Xb)} rows")
        shortfall = "absent" if len(Xb) == budget else "PRESENT"
        print(f"shortfall: {shortfall}")
        return len(Xb) == budget
    finally:
        os.remove(path)


def probe_c():
    """Peak stratum must cover the LATE part of a drift (algorithm R)."""
    print("=" * 74)
    print("PROBE C: peak-stratum saturation drops late peaks")
    print("=" * 74)
    CAP = 2000
    peak_cap = max(1, int(CAP * 0.4))
    d = SpectralHologramDistiller(crystal_size=120, high_band_ratio=0.26,
                                  lattice_rebuild_interval=1500,
                                  max_init_buffer=CAP, sketch_dim=64)
    rng = np.random.default_rng(11)
    X_n = rng.normal(size=(3000, 4))
    y_n = rng.normal(size=3000)
    X_e = rng.normal(size=(6000, 4))
    y_e = rng.normal(size=6000) + 30.0
    X_l = rng.normal(size=(6000, 4))
    y_l = rng.normal(size=6000) + 33.0
    for X, y in ((X_n, y_n), (X_e, y_e), (X_l, y_l)):
        for i in range(0, len(X), 500):
            d.partial_fit(X[i:i + 500], y[i:i + 500])
    print(f"peak_cap = {peak_cap}, _res_peak_rows = {d._res_peak_rows}")
    peak_Y = np.concatenate(d.res_peak_Y).reshape(-1)
    n_drift = int(np.sum(peak_Y > 20.0))
    n_late = int(np.sum(peak_Y > 31.5))
    total_drift = 12000
    print(f"peak stratum drift coverage: {n_drift} of {peak_cap} slots "
          f"({total_drift} drifted rows seen)")
    print(f"late-drift rows retained: {n_late} (v3.11.1: 0)")
    ok = d._res_peak_rows == peak_cap and n_drift >= int(peak_cap * 0.85) and n_late >= 200
    print("late-drift coverage:", "RESTORED (defect fixed)" if ok else "STILL BROKEN")
    return ok


if __name__ == "__main__":
    ok_a = probe_a()
    ok_b = probe_b()
    ok_c = probe_c()
    print()
    print("=" * 74)
    print("ROUND-2 PROBE SUMMARY (v3.12.0 expectations)")
    print(f"  A both+critical_threshold coherent: {'PASS' if ok_a else 'FAIL'}")
    print(f"  B budget shortfall absent:          {'PASS' if ok_b else 'FAIL'}")
    print(f"  C late drift covered in stratum:    {'PASS' if ok_c else 'FAIL'}")
    raise SystemExit(0 if (ok_a and ok_b and ok_c) else 1)

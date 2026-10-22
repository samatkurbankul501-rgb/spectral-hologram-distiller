"""Regressions for the v3.12.0 round-2 repairs (2026-10-22).

Each block pins one defect reproduced live on the v3.11.1 tree before the
fix (see ``tools/hunt/round2_probes.py``), plus the radar-calibration
additions that close the open threshold findings from the v3.11.0 review:

1. ``tail="both"`` + ``critical_threshold``: the peak slice (and the budget
   slice, and heal's protected band) re-ranked the band by
   extremeness-from-median and served LOW-side rows, while the band was
   built from ``y >= critical_threshold`` (measured: 7 of 18 slice rows
   below the threshold). The threshold now travels with the artifact and
   every band consumer reads the same definition the lattice was built from.
2. The rebuild reservoir's peak stratum kept the FIRST ``peak_cap`` peaks
   and dropped every later one, so a target drift that pushed most of the
   stream above the trigger filled the stratum with the first drifted
   batches and lost the late drift -- the part a rebuild exists to cover
   (measured: 544 of 6,000 drifted rows retained). Peaks are now sampled
   uniformly over the whole stream via Vitter's algorithm R.
3. The radar threshold was hard-wired for passport rank 32; higher ranks
   tripped false alarms on in-distribution windows. The base now scales
   with ``sqrt(rank/32)``; rank-32 crystals are bit-identical.
4. New: ``calibrate_threshold()`` -- a per-deployment quantile of the
   radar's own distances over healthy windows, persisted with the crystal
   and invalidated by ``heal_domain_shift``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import SpectralHologramDistiller, HologramCrystalLoader
from shd.distiller import BW_ALERT_THRESHOLD


def _distill(X, y, **kw):
    d = SpectralHologramDistiller(**kw)
    for i in range(0, len(X), 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    return d


# ---------------------------------------------------------------------------
# 1. tail="both" + critical_threshold coherence
# ---------------------------------------------------------------------------

def test_both_tail_critical_slice_stays_above_threshold(tmp_path):
    rng = np.random.default_rng(20261021)
    X = rng.uniform(-2, 2, size=(5000, 4))
    y = rng.normal(0, 1, size=5000)
    y[rng.random(5000) < 0.05] += 5.0

    d = _distill(X, y, crystal_size=60, high_band_ratio=0.10, tail="both",
                 critical_threshold=1.0, max_init_buffer=2000)
    path = str(tmp_path / "crit.hologram.npz")
    d.save_hologram(path, include_projector=True)

    loader = HologramCrystalLoader(path)
    # The band definition travels with the artifact (legacy: None).
    assert loader.critical_threshold == pytest.approx(1.0)

    Xp, yp, _ = loader.get_phase_peak_slice(high_band_ratio=0.30)
    assert len(yp) > 0
    # v3.11.1 served 7 of 18 rows BELOW the threshold on this exact stream.
    assert float(yp.min()) >= 1.0, "peak slice served a LOW-side row"

    # The anytime budget admits its OWN band first -- the persisted
    # high_band_ratio quota (0.10 here), which keeps the largest-y rows
    # above the threshold; a wider slice band (0.30) is a different request.
    peak_budget = loader._peak_rank_idx(len(loader.y), float(loader.high_band_ratio))
    _, yb, _ = loader.get_training_data(budget_points=len(peak_budget) + 5)
    assert float(np.min(yb[:len(peak_budget)])) >= 1.0

    # A threshold-less crystal keeps the legacy tail behaviour (regression
    # guard for the no-critical_threshold path).
    d2 = _distill(X, y, crystal_size=60, high_band_ratio=0.10, tail="both",
                  max_init_buffer=2000)
    p2 = str(tmp_path / "both.hologram.npz")
    d2.save_hologram(p2, include_projector=True)
    l2 = HologramCrystalLoader(p2)
    assert l2.critical_threshold is None
    Xp2, yp2, _ = l2.get_phase_peak_slice(high_band_ratio=0.30)
    assert len(yp2) == 18  # k_top = 0.30 * 60, unaffected by the fix


def test_heal_protects_the_threshold_band_not_the_low_side(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(5000, 4))
    y = rng.normal(0, 1, size=5000)
    y[rng.random(5000) < 0.05] += 5.0

    d = _distill(X, y, crystal_size=60, high_band_ratio=0.30, tail="both",
                 critical_threshold=1.0, max_init_buffer=2000)
    path = str(tmp_path / "heal.hologram.npz")
    d.save_hologram(path, include_projector=True)
    loader = HologramCrystalLoader(path)

    prot = loader._peak_rank_idx(len(loader.y), loader.high_band_ratio)
    assert len(prot) > 0
    assert float(loader.y[prot].min()) >= 1.0
    X_prot = loader.X[prot].copy()

    n_old = len(loader.X)
    res = loader.heal_domain_shift(rng.uniform(6, 8, size=(200, 4)), n_inject=10,
                                   force=True, max_crystal_size=n_old)
    assert res["healed"] is True
    assert res["n_pruned"] == 10

    # Every protected band row survives the prune.
    for row in X_prot:
        assert np.isclose(loader.X, row).all(axis=1).any(), \
            "protected band row was pruned"


# ---------------------------------------------------------------------------
# 2. reservoir algorithm R (late-drift coverage)
# ---------------------------------------------------------------------------

def test_peak_stratum_covers_late_drift():
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

    assert d._res_peak_rows == peak_cap
    peak_Y = np.concatenate(d.res_peak_Y).reshape(-1)
    n_drift = int(np.sum(peak_Y > 20.0))
    n_late = int(np.sum((peak_Y > 31.5)))
    # v3.11.1: the stratum filled with the FIRST drifted batches and kept
    # n_late == 0 -- the late drift, exactly what a rebuild exists to
    # cover, was gone. Uniform sampling retains ~half the stratum for the
    # late phase (expectation ~363 of 800; the bounds are generous).
    assert n_drift >= int(peak_cap * 0.85), f"drift coverage: {n_drift}/{peak_cap}"
    assert n_late >= 200, f"late drift under-represented: {n_late}/{peak_cap}"


# ---------------------------------------------------------------------------
# 3. rank-aware radar threshold
# ---------------------------------------------------------------------------

def test_threshold_scales_with_passport_rank(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3000, 6))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=3000)
    d = _distill(X, y, crystal_size=50, high_band_ratio=0.26, sketch_dim=64,
                 max_init_buffer=4096)
    p32 = str(tmp_path / "r32.hologram.npz")
    p64 = str(tmp_path / "r64.hologram.npz")
    d.save_hologram(p32, rank_passport=32, include_projector=True)
    d.save_hologram(p64, rank_passport=64, include_projector=True)

    l32 = HologramCrystalLoader(p32)
    l64 = HologramCrystalLoader(p64)
    # Rank-32 crystals are bit-identical to the v3.11.1 gate.
    assert l32._calibrated_threshold(1024) == pytest.approx(BW_ALERT_THRESHOLD)
    assert l64._calibrated_threshold(1024) == pytest.approx(
        BW_ALERT_THRESHOLD * float(np.sqrt(2.0)))
    # The in-distribution false-alarm rate of a rank-64 passport must not
    # exceed the rank-32 baseline (v3.11.1 flagged in-dist windows at high
    # rank), and a hard shift must still trip the radar.
    windows = [rng.normal(size=(256, 6)) for _ in range(10)]
    f32 = sum(l32.check_domain_shift(w)["is_shifted"] for w in windows)
    f64 = sum(l64.check_domain_shift(w)["is_shifted"] for w in windows)
    assert f64 <= f32 + 1
    assert l64.check_domain_shift(windows[0] + 10.0)["is_shifted"]
    assert l64.check_domain_shift(windows[0])["threshold_source"] == "default"


# ---------------------------------------------------------------------------
# 4. per-deployment quantile calibration
# ---------------------------------------------------------------------------

def _cal_crystal(tmp_path, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6000, 5))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=6000)
    dist = _distill(X, y, crystal_size=50, high_band_ratio=0.26, sketch_dim=64,
                    max_init_buffer=4096)
    path = str(tmp_path / f"cal{seed}.hologram.npz")
    dist.save_hologram(path, include_projector=True)
    return rng, HologramCrystalLoader(path), path


def test_quantile_calibration_controls_false_alarms(tmp_path):
    rng, loader, _ = _cal_crystal(tmp_path, seed=21)
    X_cal = rng.normal(size=(40 * 128, 5))
    X_hold = rng.normal(size=(40 * 128, 5))
    X_shift = rng.normal(size=(40 * 128, 5)) + 3.0

    rep = loader.calibrate_threshold(X_cal, window=128, fa_rate=0.05)
    assert rep["n_windows"] == 40
    assert rep["tau"] == pytest.approx(loader.tau_calibrated)

    fa = sum(loader.check_domain_shift(X_hold[i * 128:(i + 1) * 128])["is_shifted"]
             for i in range(40))
    det = sum(loader.check_domain_shift(X_shift[i * 128:(i + 1) * 128])["is_shifted"]
              for i in range(40))
    assert fa <= 5, f"calibrated radar flags {fa}/40 healthy windows"
    assert det == 40, f"calibrated radar misses shifted windows: {40 - det}/40"
    assert loader.check_domain_shift(X_hold[:128])["threshold_source"] == "calibrated"


def test_calibration_persists_and_heal_invalidates(tmp_path):
    rng, loader, _ = _cal_crystal(tmp_path, seed=22)
    X_cal = rng.normal(size=(40 * 128, 5))
    loader.calibrate_threshold(X_cal, window=128, fa_rate=0.05)

    path2 = str(tmp_path / "cal_persist.hologram.npz")
    loader.save(path2)
    l2 = HologramCrystalLoader(path2)
    assert l2.tau_calibrated == pytest.approx(loader.tau_calibrated)
    assert l2.tau_window == 128
    assert l2.tau_n_windows == 40
    w = rng.normal(size=(128, 5))
    assert l2.check_domain_shift(w)["threshold_source"] == "calibrated"

    # A heal moves the passport; the calibration must not survive it.
    l2.heal_domain_shift(rng.normal(size=(200, 5)) + 6.0, n_inject=5, force=True)
    assert l2.tau_calibrated is None
    assert l2.check_domain_shift(w)["threshold_source"] == "default"


def test_calibrate_threshold_rejects_insufficient_reference(tmp_path):
    rng, loader, _ = _cal_crystal(tmp_path, seed=23)
    with pytest.raises(ValueError):
        loader.calibrate_threshold(rng.normal(size=(5 * 128, 5)), window=128)
    with pytest.raises(ValueError):
        loader.calibrate_threshold(rng.normal(size=(40 * 128, 5)), window=4)
    with pytest.raises(ValueError):
        loader.calibrate_threshold(rng.normal(size=(40 * 128, 5)), window=128,
                                   fa_rate=0.0)

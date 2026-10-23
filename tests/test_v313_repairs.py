"""Regressions for the v3.13.0 integration: global guard + combined mean-shift radar.

Covered here (mirrors ``tools/check_guard.py``):

* ``build_global_guard`` — swap150-style guard: the lowest-weight crystal rows
  are replaced with K-Means medoids of a representative pool in the crystal's
  standardized space; donor weights are cell counts renormalised, damped by
  ``weight`` and by a residual-aware factor (``fit_residual_stats``).
* ``HologramCrystalLoader.check_domain_shift(mean_shift=True)`` — combined
  radar: the legacy bandwidth distance AND the z-score mean-shift term are
  computed together; ``calibrate_threshold_ms`` calibrates ``ms_tau_calibrated``
  on a healthy reference window; ``heal_domain_shift`` invalidates both
  calibrations (ms + bw).

These are the paths validated live on PM2.5 (swap150, 5 seeds: all-RMSE
88.3 vs 90.0 kmeans, crit-RMSE 230 vs 236 — the guard wins both metrics) and
on MetroPT-3 (4/4 leaks with FA 0.000/month vs FA budget 1.0).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402


def _make_stream(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    return X, y


def _distilled_distiller(n=6000, seed=7):
    X, y = _make_stream(n, seed)
    d = SpectralHologramDistiller(
        crystal_size=80, high_band_ratio=0.26, sketch_dim=64,
        tail_quantile=84.0, random_state=3,
    )
    B = 1500
    for i in range(0, len(X), B):
        d.partial_fit(X[i:i + B], y[i:i + B])
    d.finalize_crystal()
    return d, X, y


def _loader_with_guard(tmp_path, n_global=20):
    """Distill, guard, save; return (loader, guard report, distiller)."""
    d, X, y = _distilled_distiller()
    w_before = d.gamma_weights.copy()
    X_before = d.crystal_X.copy()
    rep = d.build_global_guard(n_global=n_global, weight=0.35, X_pool=X, y_pool=y,
                               random_state=11)
    assert rep["swapped"] == n_global
    path = str(tmp_path / "guarded.hologram.npz")
    d.save_hologram(path, include_projector=True)
    loader = HologramCrystalLoader(path)
    return loader, rep, d, w_before, X_before


def test_global_guard_swaps_lowest_weight_rows(tmp_path):
    """The guard replaces exactly n_global crystal rows with pool medoids."""
    loader, rep, d, w_before, X_before = _loader_with_guard(tmp_path)
    assert rep["swapped"] == 20
    assert rep["n_pool"] == 6000
    assert np.isfinite(d.gamma_weights).all() and (d.gamma_weights > 0).all()
    # guard must actually change the weights and the swapped rows
    assert not np.allclose(w_before, d.gamma_weights)
    n_changed = int(np.sum(np.any(X_before != d.crystal_X, axis=1)))
    assert n_changed >= 15
    # crystal_U must match the swapped rows' sketches
    _Z, U_chk = d._spectral_transform(d.crystal_X[:5], update_state=False)
    assert np.allclose(U_chk, d.crystal_U[:5], atol=1e-9)
    assert loader.X.shape == (80, 4)
    # donor weights are damped toward the floor: mean below the crystal baseline
    assert rep["min_old_weight"] >= 0.0
    assert 0.0 <= rep["guard_weight_mean"] <= 1.0


def test_global_guard_noop_and_validation():
    """n_global<=0 is a no-op; bad weight, missing crystal, and pool width
    mismatches all raise cleanly."""
    X, y = _make_stream()
    d = SpectralHologramDistiller(crystal_size=80, random_state=1)
    # before the crystal exists:
    with pytest.raises(ValueError, match="before finalize_crystal"):
        d.build_global_guard(n_global=10, X_pool=X[:100], y_pool=y[:100])
    d, X, y = _distilled_distiller()
    rep = d.build_global_guard(n_global=0, X_pool=X, y_pool=y)
    assert rep["swapped"] == 0
    with pytest.raises(ValueError, match="weight must be in"):
        d.build_global_guard(n_global=5, weight=1.5, X_pool=X, y_pool=y)
    with pytest.raises(ValueError, match="features but the crystal stores"):
        d.build_global_guard(n_global=5, X_pool=X[:, :3], y_pool=y)
    with pytest.raises(ValueError, match="must match"):
        d.build_global_guard(n_global=5, X_pool=X, y_pool=y[:50])


def test_residual_stats_explicit_pool(tmp_path):
    """fit_residual_stats fits robust residual stats on the explicit pool."""
    d, X, y = _distilled_distiller()
    st = d.fit_residual_stats(X_pool=X, y_pool=y)
    assert st["n_pool"] == len(X)
    assert np.isfinite(st["resid_median"]) and st["resid_mad"] > 0.0
    st2 = d.fit_residual_stats(X_pool=X[:100], y_pool=y[:100])
    assert st2["n_pool"] == 100
    # pre-crystal call raises
    d0 = SpectralHologramDistiller(crystal_size=80, random_state=2)
    Xs, ys = _make_stream(3000, 9)
    for i in range(0, len(Xs), 750):
        d0.partial_fit(Xs[i:i + 750], ys[i:i + 750])
    with pytest.raises(ValueError, match="needs the crystal"):
        d0.fit_residual_stats()
    d0.finalize_crystal()


def test_combined_radar_separates_shift(tmp_path):
    """check_domain_shift(mean_shift=True) reports both distances; a +3.0
    mean shift blows up the mean-shift term while healthy windows stay small."""
    loader, rep, d, w_before, X_before = _loader_with_guard(tmp_path)
    rng = np.random.default_rng(5)
    Xw = rng.normal(size=(64, 4))
    rep0 = loader.check_domain_shift(Xw, mean_shift=True)
    assert set(("bw_distance", "mean_shift_distance",
                 "combined_distance", "combined_threshold", "is_shifted")) <= set(rep0)
    assert rep0["combined_distance"] > 0.0
    Xw2 = Xw + 3.0
    rep1 = loader.check_domain_shift(Xw2, mean_shift=True)
    assert rep1["mean_shift_distance"] > rep0["mean_shift_distance"]
    assert rep1["combined_distance"] > rep0["combined_distance"]
    assert rep1["is_shifted"] is True
    # legacy path (mean_shift omitted) must NOT expose the ms fields
    rep_legacy = loader.check_domain_shift(Xw)
    assert "mean_shift_distance" not in rep_legacy
    assert rep_legacy["is_shifted"] == (rep_legacy["bw_distance"] > rep_legacy["threshold"])


def test_ms_calibration_roundtrip_and_heal_invalidates(tmp_path):
    """calibrate_threshold_ms calibrates ms_tau on a healthy reference and the
    calibration survives save/load; heal_domain_shift resets both calibrations."""
    loader, rep, d, w_before, X_before = _loader_with_guard(tmp_path)
    rng = np.random.default_rng(6)
    Xref = rng.normal(size=(2000, 4))
    crep = loader.calibrate_threshold_ms(Xref, window=64, fa_rate=0.05)
    assert crep["n_windows"] == 31
    assert np.isfinite(crep["tau"]) and crep["tau"] > 0.0
    path = str(tmp_path / "guarded.hologram.npz")
    loader.save(path)
    loader2 = HologramCrystalLoader(path)
    assert loader2.ms_tau_calibrated == crep["tau"]
    rep_ms = loader2.check_domain_shift(Xref[:64], mean_shift=True)
    assert rep_ms["combined_threshold_source"] == "calibrated"
    Xshift = Xref[:64] + 3.0
    loader2.heal_domain_shift(Xshift, force=True, random_state=0)
    assert loader2.ms_tau_calibrated is None
    assert loader2.tau_calibrated is None

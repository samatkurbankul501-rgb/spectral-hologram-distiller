"""Regressions for the 2026-11-13 Banner hunt (findings U, Y, LL, EE).

U: auto-band crystals must persist the EFFECTIVE band ratio; the loader's
   slice/heal quota follows it, not the constructor's raw ratio (measured:
   raw 0.26 gave a 78-medoid quota for a band the lattice built at 30).
Y: delta crystals persist true temporal dx statistics; the mean-shift term
   uses them (count-ordered medoid differences inflated dx std 4-5x on
   autocorrelated telemetry, desensitizing the trend alarm 3.35x).
LL: the e-process alarm is documented as a heuristic, NOT anytime-valid.
EE: CoresetPortfolio.predict shares one pair of KRR predictions with the
    gate (bit-identical output, half the predict work).
"""
import os
import tempfile

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd.portfolio import CoresetPortfolio


def _load(dist, td, name="c.hologram.npz"):
    p = os.path.join(td, name)
    dist.save_hologram(p)
    return HologramCrystalLoader(p), p


def _strip_keys(path, keys, out):
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files if k not in keys}
    np.savez_compressed(out, **payload)


def _ar1(n=6000, rho=0.95, seed=11):
    rng = np.random.default_rng(seed)
    xs = np.zeros((n, 4))
    eps = rng.normal(size=(n, 4))
    for t in range(1, n):
        xs[t] = rho * xs[t - 1] + eps[t]
    return xs


# --- U: effective band ratio -------------------------------------------------

def test_U_effective_band_ratio_persisted():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6000, 4))
    y = X[:, 0] + 0.3 * rng.normal(size=6000)
    crit = float(np.percentile(y, 97))
    d = SpectralHologramDistiller(crystal_size=300, critical_threshold=crit,
                                  auto_band_ratio=True, random_state=42)
    for i in range(0, 6000, 2500):
        d.partial_fit(X[i:i + 2500], y[i:i + 2500])
    d.finalize_crystal()
    eff = d._effective_band_ratio
    assert eff == pytest.approx(0.10, abs=0.02)  # clip(2*0.03, 0.10, 0.30)
    with tempfile.TemporaryDirectory() as td:
        ld, _ = _load(d, td)
        assert ld.auto_band_ratio is True
        assert ld.high_band_ratio == pytest.approx(eff, abs=1e-6)
        _, yp, _ = ld.get_phase_peak_slice()
        quota = max(1, int(round(300 * eff)))
        assert len(yp) <= quota


def test_U_plain_crystal_quota_unchanged():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5000, 4))
    y = X[:, 0]
    d = SpectralHologramDistiller(crystal_size=200, random_state=42)
    for i in range(0, 5000, 2500):
        d.partial_fit(X[i:i + 2500], y[i:i + 2500])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        ld, _ = _load(d, td)
        assert ld.auto_band_ratio is False
        assert ld.high_band_ratio == pytest.approx(0.26)


def test_U_legacy_artifact_without_keys_falls_back():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5000, 4))
    y = X[:, 0]
    d = SpectralHologramDistiller(crystal_size=200, critical_threshold=1.0,
                                  auto_band_ratio=True, random_state=42)
    for i in range(0, 5000, 2500):
        d.partial_fit(X[i:i + 2500], y[i:i + 2500])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        _, p = _load(d, td)
        legacy = os.path.join(td, "legacy.hologram.npz")
        _strip_keys(p, {"auto_band_ratio", "effective_band_ratio"}, legacy)
        ld = HologramCrystalLoader(legacy)
        assert ld.auto_band_ratio is False
        assert ld.high_band_ratio == pytest.approx(0.26)  # raw fallback


# --- Y: true temporal dx statistics ------------------------------------------

def test_Y_dx_stats_persisted_and_trend_alarm_restored():
    xs = _ar1()
    d = SpectralHologramDistiller(crystal_size=300, include_deltas=True,
                                  random_state=42)
    for i in range(0, 6000, 2500):
        d.partial_fit(xs[i:i + 2500])
    d.finalize_crystal()
    true_sd = (xs[1:] - xs[:-1]).std(axis=0)
    with tempfile.TemporaryDirectory() as td:
        ld, _ = _load(d, td)
        assert ld.dx_mean_ is not None and ld.dx_std_ is not None
        assert np.allclose(ld.dx_std_, true_sd, rtol=0.05)
        # Steep centered trend: raw mean ~ 0, dx mean ~ 1/step.
        win = xs[-256:].copy()
        win[:, 0] = np.linspace(-128, 127, 256)
        ms, _ = ld._mean_shift_distance(win)
        ref = ld._mean_shift_reference()
        X_eff = ld._effective_stream(win)
        mu = np.concatenate([ref[:, :4].mean(axis=0), ld.dx_mean_])
        sd = np.concatenate([ref[:, :4].std(axis=0), true_sd])
        ok = sd > 1e-9
        ms_true = float((np.abs(X_eff.mean(axis=0) - mu) / np.where(ok, sd, 1.0))[ok].max())
        # Pre-fix this ratio was ~3.35x (desensitized); now it must be ~1.
        assert ms >= 0.8 * ms_true


def test_Y_legacy_delta_artifact_falls_back():
    xs = _ar1(n=5000)
    d = SpectralHologramDistiller(crystal_size=200, include_deltas=True,
                                  random_state=42)
    for i in range(0, 5000, 2500):
        d.partial_fit(xs[i:i + 2500])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        _, p = _load(d, td)
        legacy = os.path.join(td, "legacy.hologram.npz")
        _strip_keys(p, {"dx_mean", "dx_std"}, legacy)
        ld = HologramCrystalLoader(legacy)
        assert ld.dx_mean_ is None and ld.dx_std_ is None
        ms, n = ld._mean_shift_distance(xs[-128:])
        assert np.isfinite(ms) and n == 128


def test_Y_non_delta_crystal_has_no_dx_keys():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5000, 4))
    d = SpectralHologramDistiller(crystal_size=200, random_state=42)
    for i in range(0, 5000, 2500):
        d.partial_fit(X[i:i + 2500], X[i:i + 2500, 0])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        _, p = _load(d, td)
        with np.load(p, allow_pickle=False) as data:
            assert "dx_mean" not in data.files
            assert "dx_std" not in data.files
        ld = HologramCrystalLoader(p)
        assert ld.dx_mean_ is None


# --- LL: e-process honesty ----------------------------------------------------

def test_LL_e_process_documented_as_heuristic():
    doc = HologramCrystalLoader._e_process_alarm.__doc__
    assert "HEURISTIC" in doc
    assert "NOT a certified anytime-valid" in doc


def test_LL_e_process_behaviour_unchanged():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(2000, 4))
    d = SpectralHologramDistiller(crystal_size=100, random_state=42)
    for i in range(0, 2000, 500):
        d.partial_fit(X[i:i + 500], X[i:i + 500, 0])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        ld, _ = _load(d, td)
        ld.calibrate_threshold_panel(X[:1200], window=64, fa_rate=0.05,
                                     e_process=True)
        rep = ld.check_domain_shift(X[1200:1264], panel=True, e_process=True)
        for key in ("e_value", "e_t", "e_average", "e_threshold", "e_alarm"):
            assert key in rep
        assert rep["e_t"] == 1


# --- EE: portfolio predict shares predictions --------------------------------

def test_EE_predict_bit_identical_to_manual_blend():
    rng = np.random.default_rng(5)
    Xc = rng.normal(size=(120, 4))
    yc = Xc[:, 0] + 0.1 * rng.normal(size=120)
    Xg = rng.normal(size=(120, 4))
    yg = Xg[:, 0] + 0.1 * rng.normal(size=120)
    scaler = StandardScaler().fit(np.vstack([Xc, Xg]))
    pf = CoresetPortfolio(critical_threshold=float(np.percentile(yc, 90))).fit(
        Xc, yc, None, Xg, yg, None, scaler, rbf_gamma=1.0 / (4 * 1.1))
    Xt = rng.normal(size=(500, 4))
    p = pf.predict(Xt)
    pc, pg = pf.expert_predictions(Xt)
    w = pf.gate_weights(Xt)
    assert np.array_equal(p, w * pc + (1.0 - w) * pg)
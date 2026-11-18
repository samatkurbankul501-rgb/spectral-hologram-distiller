"""Regression for the 2026-11-18 hunt finding Y2.

Y2: HologramCrystalLoader.save() dropped the finding-Y temporal dx statistics
    (dx_mean/dx_std) and the finding-U band keys (auto_band_ratio /
    effective_band_ratio). heal->save->reload (CLI ``shd heal``) and plain
    load->save (CLI ``shd migrate``) therefore reverted a delta crystal to the
    count-ordered medoid dx scale -- measured 4.32x trend-alarm desensitization
    on an identical window (mean-shift 3.2918 -> 0.7623). save() now persists
    both, so a re-saved crystal keeps the radar sensitivity and band quota it
    was distilled with. The distiller's save_hologram() path is untouched.
"""
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from shd import HologramCrystalLoader, SpectralHologramDistiller

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shd_watch as watch


def _ar1(n=6000, rho=0.95, seed=11):
    rng = np.random.default_rng(seed)
    xs = np.zeros((n, 4))
    eps = rng.normal(size=(n, 4))
    for t in range(1, n):
        xs[t] = rho * xs[t - 1] + eps[t]
    return xs


def _delta_crystal(td, name="c.hologram.npz"):
    xs = _ar1()
    d = SpectralHologramDistiller(crystal_size=200, include_deltas=True,
                                  random_state=42)
    for i in range(0, len(xs), 2500):
        d.partial_fit(xs[i:i + 2500])
    d.finalize_crystal()
    p = os.path.join(td, name)
    d.save_hologram(p)
    return xs, p


def _trend_win(xs):
    win = xs[-256:].copy()
    win[:, 0] = np.linspace(-128, 127, 256)
    return win


def test_Y2_heal_save_reload_preserves_dx_and_sensitivity():
    with tempfile.TemporaryDirectory() as td:
        xs, p = _delta_crystal(td)
        ld = HologramCrystalLoader(p)
        assert ld.dx_mean_ is not None and ld.dx_std_ is not None
        info = ld.heal_domain_shift(xs[-512:], n_inject=10, force=True)
        assert info["healed"]
        win = _trend_win(xs)
        ms_mem, _ = ld._mean_shift_distance(win)
        out = os.path.join(td, "healed.hologram.npz")
        ld.save(out)
        with np.load(out, allow_pickle=False) as z:
            assert "dx_mean" in z.files and "dx_std" in z.files
        ld2 = HologramCrystalLoader(out)
        assert ld2.dx_mean_ is not None and ld2.dx_std_ is not None
        ms_reload, _ = ld2._mean_shift_distance(win)
        # Pre-fix the reloaded crystal fell back to the count-ordered medoid dx
        # scale and scored far below the in-memory healed loader. Now the
        # re-saved crystal keeps the same sensitivity (data-independent check:
        # compare against a dx-stripped copy, which reproduces the broken path).
        assert ms_reload == pytest.approx(ms_mem, rel=0.05)
        stripped = os.path.join(td, "stripped.hologram.npz")
        with np.load(out, allow_pickle=False) as z:
            payload = {k: z[k] for k in z.files if k not in ("dx_mean", "dx_std")}
        np.savez_compressed(stripped, **payload)
        ms_stripped, _ = HologramCrystalLoader(stripped)._mean_shift_distance(win)
        assert ms_reload > ms_stripped


def test_Y2_migrate_roundtrip_preserves_dx():
    with tempfile.TemporaryDirectory() as td:
        _, p = _delta_crystal(td)
        ld = HologramCrystalLoader(p)
        out = os.path.join(td, "migrated.hologram.npz")
        ld.save(out)  # plain load->save, no heal (the CLI migrate path)
        ld2 = HologramCrystalLoader(out)
        assert ld2.dx_mean_ is not None and ld2.dx_std_ is not None
        assert np.allclose(ld2.dx_mean_, ld.dx_mean_, rtol=1e-5)
        assert np.allclose(ld2.dx_std_, ld.dx_std_, rtol=1e-5)


def test_Y2_band_keys_survive_roundtrip():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6000, 4))
    y = X[:, 0] + 0.3 * rng.normal(size=6000)
    crit = float(np.percentile(y, 97))
    d = SpectralHologramDistiller(crystal_size=300, critical_threshold=crit,
                                  auto_band_ratio=True, random_state=42)
    for i in range(0, 6000, 2500):
        d.partial_fit(X[i:i + 2500], y[i:i + 2500])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "auto.hologram.npz")
        d.save_hologram(p)
        ld = HologramCrystalLoader(p)
        assert ld.auto_band_ratio is True
        eff = ld.high_band_ratio
        out = os.path.join(td, "auto_resaved.hologram.npz")
        ld.save(out)
        ld2 = HologramCrystalLoader(out)
        assert ld2.auto_band_ratio is True
        assert ld2.high_band_ratio == pytest.approx(eff)


def test_Y2_non_delta_save_has_no_dx_keys():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5000, 4))
    d = SpectralHologramDistiller(crystal_size=200, random_state=42)
    for i in range(0, 5000, 2500):
        d.partial_fit(X[i:i + 2500], X[i:i + 2500, 0])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "plain.hologram.npz")
        d.save_hologram(p)
        out = os.path.join(td, "plain_resaved.hologram.npz")
        HologramCrystalLoader(p).save(out)
        with np.load(out, allow_pickle=False) as z:
            assert "dx_mean" not in z.files and "dx_std" not in z.files
        assert HologramCrystalLoader(out).dx_mean_ is None


# --- D3: uncalibrated combined radar must not report a bogus verdict ----------

def _stationary_loader(td):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6000, 4))
    d = SpectralHologramDistiller(crystal_size=150, random_state=42)
    for i in range(0, 6000, 2500):
        d.partial_fit(X[i:i + 2500], X[i:i + 2500, 0])
    d.finalize_crystal()
    p = os.path.join(td, "c.hologram.npz")
    d.save_hologram(p)
    return HologramCrystalLoader(p), rng


def test_D3_uncalibrated_combined_verdict_guarded():
    with tempfile.TemporaryDirectory() as td:
        ld, rng = _stationary_loader(td)
        win = rng.normal(size=(64, 4))  # healthy, in-distribution
        bw = ld.check_domain_shift(win)
        with pytest.warns(UserWarning, match="calibrate_threshold_ms"):
            rep = ld.check_domain_shift(win, mean_shift=True)
        # Pre-fix: is_shifted = combined > BW-constant false-alarmed on healthy
        # windows (probe: 26/30). Now the combined verdict is unknown and
        # is_shifted falls back to the BW verdict, exactly like the panel guard.
        assert rep["combined_is_shifted"] is None
        assert rep["combined_threshold_source"] == "uncalibrated"
        assert rep["is_shifted"] == bw["is_shifted"]
        # Calibration restores a live combined verdict.
        ld.calibrate_threshold_ms(rng.normal(size=(2000, 4)), window=64, fa_rate=0.05)
        rep2 = ld.check_domain_shift(win, mean_shift=True)
        assert isinstance(rep2["combined_is_shifted"], bool)
        assert rep2["combined_threshold_source"] == "calibrated"
        assert rep2["is_shifted"] == rep2["combined_is_shifted"]


def test_D3_adaptive_off_keeps_default_source_without_warning():
    with tempfile.TemporaryDirectory() as td:
        ld, rng = _stationary_loader(td)
        ld.calibrate_threshold_ms(rng.normal(size=(2000, 4)), window=64, fa_rate=0.05)
        win = rng.normal(size=(64, 4))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            rep = ld.check_domain_shift(win, mean_shift=True, adaptive_threshold=False)
        assert rep["combined_threshold_source"] == "default"
        assert rep["combined_is_shifted"] is None
        assert not any("calibrate_threshold_ms" in str(w.message) for w in ws)


# --- W5: shd_watch can deploy the combined/panel radar, not just BW -----------

def test_W5_window_scores_radar_modes_match_loader():
    with tempfile.TemporaryDirectory() as td:
        ld, rng = _stationary_loader(td)
        minutes = (np.datetime64("2020-03-01", "m")
                   + np.arange(240) * np.timedelta64(1, "m"))
        Xw = rng.normal(size=(240, 4))
        ib, sb = watch.window_scores(ld, Xw, 60, 30, minutes=minutes, radar="bw")
        ic, sc = watch.window_scores(ld, Xw, 60, 30, minutes=minutes, radar="combined")
        ip, sp = watch.window_scores(ld, Xw, 60, 30, minutes=minutes, radar="panel")
        np.testing.assert_array_equal(ib, ic)
        np.testing.assert_array_equal(ib, ip)
        for (lo, hi), s in zip(watch.window_bounds(minutes, 60, 30), sc):
            rep = ld.check_domain_shift(Xw[lo:hi], adaptive_threshold=False,
                                        mean_shift=True)
            assert s == pytest.approx(rep["combined_distance"])
        assert np.all(sp >= sc - 1e-9)  # panel adds non-negative terms to combined
        with pytest.raises(ValueError, match="radar"):
            watch.window_scores(ld, Xw, 60, 30, minutes=minutes, radar="bogus")


class _MultiRadar:
    def check_domain_shift(self, X, **kw):
        m = float(np.mean(X))
        return {"bw_distance": m, "combined_distance": m + 1.0,
                "panel_distance": m + 2.0, "threshold": 1.0, "is_shifted": False}


def test_W5_calibrate_records_radar_and_watch_enforces_contract(tmp_path, monkeypatch, capsys):
    ts = (np.datetime64("2020-03-01T00:00", "m")
          + np.arange(4 * 24 * 60) * np.timedelta64(1, "m"))
    p = tmp_path / "minutes.npz"
    np.savez(p, minutes=ts.astype("int64"), X=np.ones((len(ts), 2)), features=["a", "b"])
    monkeypatch.setattr(watch, "HologramCrystalLoader", lambda path: _MultiRadar())
    cfg = tmp_path / "threshold.json"
    common = ["--npz", str(p), "--crystal", "unused.npz"]
    watch.main(["calibrate", *common, "--from", "2020-03-01", "--to", "2020-03-03",
                "--radar", "combined", "--out", str(cfg)])
    c = json.loads(capsys.readouterr().out)
    assert c["radar"] == "combined"
    watch.main(["watch", *common, "--threshold", str(cfg),
                "--from", "2020-03-03", "--to", "2020-03-05"])
    r = json.loads(capsys.readouterr().out)
    assert r["radar"] == "combined"
    with pytest.raises(SystemExit, match="re-running calibrate"):
        watch.main(["watch", *common, "--threshold", str(cfg), "--from", "2020-03-03",
                    "--to", "2020-03-05", "--radar", "bw"])
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
import os
import tempfile

import numpy as np
import pytest

from shd import HologramCrystalLoader, SpectralHologramDistiller


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
"""Regressions for the v3.11.1 repairs (2026-10-21).

Each block pins one defect that the v3.11.0 code review found. The tests
assert the *contract* that was broken, not a tuned number, so they survive
later re-tuning. Every defect below was reproduced on the v3.11.0 tree
before the repair.

1. anytime budget starved the phase-peak band
2. the peak slice ignored the crystal's persisted band ratio
3. a label-free heal wrote unlabelled rows without saying so
4. ``critical_threshold`` + ``tail="low"`` built a self-contradictory crystal
5. the passport reference default changed the published radar semantics
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import SpectralHologramDistiller, HologramCrystalLoader


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _rare_stream(n: int = 4000, d: int = 4, rare_frac: float = 0.06, seed: int = 5):
    """A stream with a clear high-y rare regime (the phase-peak band)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=n)
    rare = rng.random(n) < rare_frac
    y[rare] += 8.0
    return X, y


def _crystal(tmp_path, *, crystal_size=100, band=0.30, seed=5):
    X, y = _rare_stream(seed=seed)
    d = SpectralHologramDistiller(
        crystal_size=crystal_size, high_band_ratio=band, max_init_buffer=4096,
    )
    for i in range(0, len(X), 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    path = str(tmp_path / "c.hologram.npz")
    d.save_hologram(path, include_projector=True)
    return path, HologramCrystalLoader(path)


# --------------------------------------------------------------------------
# 1. anytime budget must not starve the phase-peak band
# --------------------------------------------------------------------------

def test_budget_slice_keeps_the_phase_peak_band(tmp_path):
    """A budget slice must admit the peak band FIRST, not the background.

    v3.11.0 bug: crystal rows are stored sorted by cell count, so
    ``get_training_data(budget_points=k)`` took the first k rows -- i.e. the
    background. Measured: a 200-medoid crystal with a 60-row peak band kept
    only 4 peak rows in a budget of 100.
    """
    path, loader = _crystal(tmp_path, crystal_size=100, band=0.30)
    n_peak = max(1, int(round(len(loader.y) * loader.high_band_ratio)))
    peak_y = np.sort(loader.y)[-n_peak:]

    budget = n_peak + 10
    _, y_sub, w_sub = loader.get_training_data(budget_points=budget)

    assert len(y_sub) == budget
    # every peak-band row survived the budget
    for v in peak_y:
        assert np.isclose(y_sub, v).any(), "phase-peak row dropped by the budget slice"
    # and the slice really is the rare regime, not just any rows
    assert y_sub.max() == pytest.approx(loader.y.max())


def test_budget_slice_does_not_mutate_crystal_weights(tmp_path):
    """Renormalising the returned weights must not touch the crystal."""
    path, loader = _crystal(tmp_path, crystal_size=60, band=0.30)
    before = loader.weights.copy()
    loader.get_training_data(budget_points=20)
    assert np.allclose(loader.weights, before)


# --------------------------------------------------------------------------
# 2. the peak slice must follow the crystal's persisted band ratio
# --------------------------------------------------------------------------

def test_phase_peak_slice_uses_persisted_band_ratio(tmp_path):
    """``get_phase_peak_slice()`` must honour the ratio stored in the artifact.

    v3.11.0 bug: the default was hard-coded to 0.26 while the crystal was
    built with (and heal protects) its own persisted ratio -- a critical-first
    crystal (br=0.65) served a band 2.5x too narrow.
    """
    path, loader = _crystal(tmp_path, crystal_size=100, band=0.30)
    assert loader.high_band_ratio == pytest.approx(0.30)

    X_peak, y_peak, _ = loader.get_phase_peak_slice()
    n_expected = max(1, int(round(len(loader.y) * 0.30)))
    assert len(X_peak) == n_expected
    # the old hard-coded default would have returned a different band
    assert len(X_peak) != max(1, int(round(len(loader.y) * 0.26)))

    # an explicit ratio still wins
    X_explicit, _, _ = loader.get_phase_peak_slice(high_band_ratio=0.26)
    assert len(X_explicit) == max(1, int(round(len(loader.y) * 0.26)))


# --------------------------------------------------------------------------
# 3. a label-free heal must not silently poison the training set
# --------------------------------------------------------------------------

def test_label_free_heal_marks_rows_and_reports_them(tmp_path):
    """Unlabelled injected rows must be marked, droppable and reported."""
    path, loader = _crystal(tmp_path, crystal_size=60, band=0.30)
    m_before = len(loader.y)
    real_median = float(np.median(loader.y))

    rng = np.random.default_rng(99)
    X_ood = rng.uniform(5.0, 7.0, size=(200, 4))
    res = loader.heal_domain_shift(X_ood, n_inject=6, force=True)

    assert res["healed"] is True
    assert res["n_injected"] == 6
    # transparency: the caller is told what happened
    assert res["n_unlabelled_injected"] == 6
    assert res["warning"] and "placeholder" in res["warning"]

    # the injected rows are marked unlabelled and carry the neutral placeholder
    assert len(loader.y_labelled) == len(loader.y)
    assert not loader.y_labelled[-6:].any()
    injected = loader.y[-6:]
    assert np.allclose(injected, real_median)

    # the default training path drops them
    _, y_train, _ = loader.get_training_data()
    assert len(y_train) == m_before
    # and the opt-out still returns them
    _, y_all, _ = loader.get_training_data(drop_unlabelled=False)
    assert len(y_all) == m_before + 6


def test_label_free_heal_does_not_mix_scales(tmp_path):
    """The placeholder must live on the crystal's own target scale.

    A novelty score in [0, 1] written next to Kelvin-scale targets silently
    creates a mixture of two incompatible quantities, which breaks any
    downstream model trained on the crystal.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 4))
    y = 85.0 + 12.0 * X[:, 0] + rng.normal(scale=0.4, size=3000)
    d = SpectralHologramDistiller(crystal_size=60, high_band_ratio=0.30,
                                  max_init_buffer=4096)
    for i in range(0, len(X), 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    path = str(tmp_path / "kelvin.hologram.npz")
    d.save_hologram(path, include_projector=True)
    loader = HologramCrystalLoader(path)

    X_ood = rng.uniform(6.0, 8.0, size=(150, 4))
    loader.heal_domain_shift(X_ood, n_inject=5, force=True)

    injected = loader.y[-5:]
    assert injected.min() > 1.0, "novelty-scale target leaked into the crystal"
    assert np.allclose(injected, float(np.median(loader.y[:-5])))


# --------------------------------------------------------------------------
# 4. critical_threshold + tail="low" is a contradiction
# --------------------------------------------------------------------------
def test_critical_threshold_rejects_low_tail():
    """The critical band is the UPPER target side; tail='low' serves the lowest.

    v3.11.0 accepted the combination and produced a crystal whose band was
    built from y >= threshold but read back from the low tail.
    """
    with pytest.raises(ValueError, match="tail='low'"):
        SpectralHologramDistiller(tail="low", critical_threshold=1.0)

    # the sane combinations are still accepted
    SpectralHologramDistiller(tail="high", critical_threshold=1.0)
    SpectralHologramDistiller(tail="both", critical_threshold=1.0)
    SpectralHologramDistiller(tail="low")


# --------------------------------------------------------------------------
# 5. the passport reference default must not change published semantics
# --------------------------------------------------------------------------

def test_passport_reference_default_is_stream(tmp_path):
    """The default must reproduce the published artifacts' radar semantics.

    The experimental 'background' reference is opt-in: it changes the radar's
    sensitivity, not only its reference population.
    """
    d = SpectralHologramDistiller(crystal_size=20)
    assert d.passport_reference == "stream"

    path, loader = _crystal(tmp_path, crystal_size=40, band=0.30)
    assert loader.passport_reference == "stream"

    d_bg = SpectralHologramDistiller(crystal_size=20, passport_reference="background")
    assert d_bg.passport_reference == "background"
    with pytest.raises(ValueError):
        SpectralHologramDistiller(crystal_size=20, passport_reference="nonsense")

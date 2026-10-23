"""Regressions for the v3.12.1 repair (2026-10-22, round-2 probe B).

Round-2 probe B (``tools/hunt/round2_probes.py``) reproduced on the v3.11.1
tree: ``get_training_data(budget_points=k)`` admitted the peak band FIRST
and dropped unlabelled heal rows AFTER, so a 60-row budget returned 30 rows
(30 unlabelled median placeholders were admitted by the budget, then
dropped by ``drop_unlabelled=True``). v3.12.1 drops unlabelled rows FIRST
and spends the budget on the remaining labelled rows, so the requested
budget is honoured unless fewer labelled rows exist than the budget.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import SpectralHologramDistiller, HologramCrystalLoader


def _crystal_with_heal(tmp_path, seed=7):
    """A crystal plus 30 forced (label-free) heal rows, saved to disk."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6000, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=6000)
    d = SpectralHologramDistiller(crystal_size=100, high_band_ratio=0.26,
                                  max_init_buffer=4096)
    for i in range(0, len(X), 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    path = str(tmp_path / "healed.hologram.npz")
    d.save_hologram(path, include_projector=True)
    loader = HologramCrystalLoader(path)

    X_ood = rng.uniform(6, 8, size=(400, 4))
    loader.heal_domain_shift(X_ood, n_inject=30, force=True)
    assert int(np.sum(~loader.y_labelled)) == 30
    return loader


def test_budget_slice_honours_the_budget_after_heal(tmp_path):
    """A budget slice must return exactly the requested number of rows.

    v3.11.1 bug: budget admission ran on the FULL crystal (labelled +
    unlabelled), and ``drop_unlabelled`` then removed the admitted
    placeholders -- reproduced in round-2 probe B: 60 requested, 30 returned.
    """
    loader = _crystal_with_heal(tmp_path)
    budget = 60
    Xb, yb, wb = loader.get_training_data(budget_points=budget)
    assert len(yb) == budget
    assert len(Xb) == budget and len(wb) == budget
    # weights are renormalised over the returned rows
    assert abs(float(np.mean(wb)) - 1.0) < 1e-9
    # and the budget is still peak-first on the LABELLED rows
    n_lab = int(np.sum(loader.y_labelled))
    assert budget <= n_lab
    y_lab = loader.y[loader.y_labelled]
    peak = loader._peak_rank_idx(n_lab, float(loader.high_band_ratio), y=y_lab)
    n_peak = len(peak)
    assert float(np.min(yb[:n_peak])) >= float(np.min(y_lab[peak]))


def test_budget_shortfall_only_when_labelled_rows_are_fewer(tmp_path):
    """If the crystal holds fewer LABELLED rows than the budget, the slice
    returns exactly the labelled rows -- never more, never fewer."""
    loader = _crystal_with_heal(tmp_path)
    n_lab = int(np.sum(loader.y_labelled))
    Xb, yb, wb = loader.get_training_data(budget_points=n_lab + 50)
    assert len(yb) == n_lab
    Xb2, yb2, _ = loader.get_training_data(budget_points=n_lab + 50,
                                            drop_unlabelled=False)
    # no drop: a budget larger than the crystal (130 > 110 rows) returns
    # the WHOLE crystal, so the shortfall is exactly the unlabelled rows.
    assert len(yb2) == len(loader.y)
    assert len(yb2) - (n_lab + 50) == -20  # 30 unlabelled rows short of the budget


def test_budget_without_unlabelled_rows_unchanged(tmp_path):
    """Crystals without placeholder rows keep the v3.11.1 peak-first budget
    behaviour (regression guard for the unhealed path)."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(3000, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=3000)
    d = SpectralHologramDistiller(crystal_size=80, high_band_ratio=0.26,
                                   max_init_buffer=4096)
    for i in range(0, len(X), 500):
        d.partial_fit(X[i:i + 500], y[i:i + 500])
    d.finalize_crystal()
    path = str(tmp_path / "plain.hologram.npz")
    d.save_hologram(path, include_projector=True)
    loader = HologramCrystalLoader(path)

    budget = 30
    Xb, yb, wb = loader.get_training_data(budget_points=budget)
    assert len(yb) == budget
    assert abs(float(np.mean(wb)) - 1.0) < 1e-9

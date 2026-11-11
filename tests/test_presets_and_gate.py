"""Regression tests for the v3.14.1 iteration: named deployment presets
(shd.presets), the deterministic gate solver policy (lbfgs primary with a
liblinear fallback), and preset drift validation.

The presets are MEASURED operating points documented in README section 6;
they must stay subset-of-signature, non-redundant and constructible.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shd  # noqa: E402
from shd import SpectralHologramDistiller, apply_preset  # noqa: E402
from shd.presets import PRESETS  # noqa: E402


def make_stream(n=6000, seed=11):
    """4D stream with a rare phase-peak regime and heterogeneous noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3] + 0.8 * rng.normal(size=int(hot.sum()))
    return X, y


def make_arms(X, y, seed=7, n_coreset=150, threshold=6.0):
    """Two coresets (critical oversampled, global uniform) + scaler."""
    rng = np.random.default_rng(seed)
    hi = y >= threshold
    idx_c = np.flatnonzero(hi)
    idx_l = np.flatnonzero(~hi)
    pick_c = rng.choice(idx_c, size=min(60, len(idx_c)), replace=False)
    pick_l = rng.choice(idx_l, size=n_coreset - len(pick_c), replace=False)
    Xc = X[np.r_[pick_c, pick_l]]
    yc = y[np.r_[pick_c, pick_l]]
    wc = np.where(yc >= threshold, 1.0, 0.5)
    Xg = X[:n_coreset]
    yg = y[:n_coreset]
    scaler = StandardScaler().fit(X)
    return Xc, yc, wc, Xg, yg, scaler


# -- presets ---------------------------------------------------------------

def test_validate_presets_passes():
    """The shipped preset table is consistent with the constructor."""
    shd.validate_presets()


def test_preset_params_are_subset_of_signature():
    import inspect
    sig = inspect.signature(SpectralHologramDistiller.__init__)
    accepted = set(sig.parameters) - {"self"}
    for name, preset in PRESETS.items():
        assert set(preset["params"]) <= accepted, name


def test_presets_do_not_own_budget_knobs():
    reserved = {"crystal_size", "sketch_dim", "random_state"}
    for name, preset in PRESETS.items():
        assert not (set(preset["params"]) & reserved), name


def test_apply_preset_unknown_name():
    with pytest.raises(KeyError):
        apply_preset("no_such_preset")


def test_apply_preset_lists_available_names_in_error():
    with pytest.raises(KeyError) as e:
        apply_preset("nope")
    for name in PRESETS:
        assert name in str(e.value)


def test_apply_preset_rejects_reserved_overrides():
    with pytest.raises(ValueError):
        apply_preset("balanced", crystal_size=600)


def test_apply_preset_overrides_win():
    params = apply_preset("critical_first", high_band_ratio=0.40)
    assert params["high_band_ratio"] == 0.40


def test_apply_preset_returns_fresh_dict():
    a = apply_preset("balanced")
    a["high_band_ratio"] = 0.99
    assert apply_preset("balanced")["high_band_ratio"] == 0.50


def test_every_preset_builds_a_distiller():
    for name in PRESETS:
        d = SpectralHologramDistiller(crystal_size=32, **apply_preset(name))
        assert d.M == 32


def test_preset_values_match_readme_section6():
    """The preset numbers are the measured README section 6 operating points."""
    assert PRESETS["balanced"]["params"]["high_band_ratio"] == 0.50
    assert PRESETS["critical_first"]["params"]["high_band_ratio"] == 0.65
    assert PRESETS["sparse_stream_top3pct"]["params"]["tail_quantile"] == 97.0
    assert PRESETS["default"]["params"] == {}
    assert PRESETS["global_first"]["params"] == {
        "high_band_ratio": 0.26, "tail_quantile": 84.0,
    }


def test_default_preset_changes_do_not_silently_alter_protocol():
    """If the constructor defaults ever change, the explicit-default preset
    and the test above must be updated on purpose, not by drift."""
    import inspect
    sig = inspect.signature(SpectralHologramDistiller.__init__)
    defaults = {k: p.default for k, p in sig.parameters.items()
                if p.default is not inspect.Parameter.empty}
    assert defaults["high_band_ratio"] == 0.26
    assert defaults["tail_quantile"] == 84.0


# -- gate solver policy ----------------------------------------------------

def _fit_gate(make_arms_kwargs=None):
    X, y = make_stream()
    args = make_arms_kwargs or {}
    Xc, yc, wc, Xg, yg, scaler = make_arms(X, y, **args)
    pf = shd.CoresetPortfolio(critical_threshold=6.0).fit(
        Xc, yc, wc, Xg, yg, np.ones(len(yg)), scaler,
        rbf_gamma=1.0 / (X.shape[1] * 1.1), model_seed=42)
    return pf


def test_gate_solver_recorded():
    pf = _fit_gate()
    assert pf.gate_solver_ in ("lbfgs", "liblinear")


def test_gate_no_convergence_warning_and_deterministic():
    import warnings
    X, y = make_stream(seed=3)
    Xc, yc, wc, Xg, yg, scaler = make_arms(X, y, seed=9)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        p1 = shd.CoresetPortfolio(critical_threshold=6.0).fit(
            Xc, yc, wc, Xg, yg, np.ones(len(yg)), scaler,
            rbf_gamma=1.0 / (X.shape[1] * 1.1), model_seed=42)
        p2 = shd.CoresetPortfolio(critical_threshold=6.0).fit(
            Xc, yc, wc, Xg, yg, np.ones(len(yg)), scaler,
            rbf_gamma=1.0 / (X.shape[1] * 1.1), model_seed=42)
    assert not [w for w in wlist if "onvergen" in str(w.category)]
    w1 = p1.gate_weights(X)
    w2 = p2.gate_weights(X)
    assert np.array_equal(w1, w2)


def test_gate_max_iter_raised():
    pf = _fit_gate()
    assert pf.gate_.max_iter == 5000


def test_gate_weights_contract_unchanged():
    """The solver policy must not change the public weight contract."""
    pf = _fit_gate()
    X, _ = make_stream()
    w = pf.gate_weights(X)
    assert w.shape == (len(X),)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)

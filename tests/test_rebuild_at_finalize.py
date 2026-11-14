"""Regressions for the rebuild_at_finalize suppressor fix (2026-11-14).

The default interval=0 online-only lattice cannot create critical cells for
post-warm-up rows; a single finalize-time rebuild recovers them. The fix is
opt-in: rebuild_at_finalize=False (the default) must stay bit-for-bit.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from bench_seeds import load_reactor
from shd import SpectralHologramDistiller, apply_preset
from shd.presets import validate_presets


def _build(X_tr, y_tr, seed, **kw):
    d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                                  sketch_dim=256, tail_quantile=84.0,
                                  phase_kappa=3.2, random_state=seed, **kw)
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i:i + 2500], y_tr[i:i + 2500])
    d.finalize_crystal()
    return d


def test_default_bit_for_bit_with_flag_false():
    X, y, crit, _ = load_reactor()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    a = _build(X_tr, y_tr, 0)
    b = _build(X_tr, y_tr, 0, rebuild_at_finalize=False)
    assert np.array_equal(a.crystal_X, b.crystal_X)
    assert np.array_equal(a.crystal_Y, b.crystal_Y)
    assert np.array_equal(a.gamma_weights, b.gamma_weights)


def test_rebuild_at_finalize_recovers_critical_capture():
    X, y, crit, _ = load_reactor()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    default = _build(X_tr, y_tr, 0)
    rebuild = _build(X_tr, y_tr, 0, rebuild_at_finalize=True)
    rare_default = int(np.sum(default.crystal_Y >= crit))
    rare_rebuild = int(np.sum(rebuild.crystal_Y >= crit))
    # the suppressor fix must capture more critical medoids (measured 18 -> 35+)
    assert rare_rebuild > rare_default
    assert rare_rebuild >= 30


def test_critical_rebuild_preset_valid():
    validate_presets()
    params = apply_preset("critical_rebuild")
    assert params == {"rebuild_at_finalize": True}
    SpectralHologramDistiller(crystal_size=32, **params)


def test_rebuild_at_finalize_does_not_periodic_rebuild():
    # with rebuild_at_finalize=True and interval=0, the reservoir is collected
    # but no periodic rebuild fires mid-stream (rows_since_rebuild keeps growing)
    X, y, crit, _ = load_reactor()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.20, random_state=42)
    d = SpectralHologramDistiller(crystal_size=300, random_state=0,
                                  rebuild_at_finalize=True)
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i:i + 2500], y_tr[i:i + 2500])
    assert d._reservoir_rows > 0           # reservoir collected
    assert d._rows_since_rebuild > 0       # no periodic rebuild reset it
    d.finalize_crystal()
    assert d._rows_since_rebuild == 0      # finalize rebuild reset it
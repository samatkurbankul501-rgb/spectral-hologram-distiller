"""Audit 2026-11-11: the lattice_rebuild_interval contract.

The v3.12.0 docstring promised that the default interval=0 rebuilds the
lattice once after warm-up so the crystal "covers the whole stream"; the
code has gated BOTH the reservoir and the rebuild on > 0 since the
mechanism was introduced. These tests pin the ACTUAL contract -- 0 is the
prefix-only, bit-parity published default; >0 gives late regimes dedicated
cells -- plus the finalize-time warning, so the docstring cannot drift
again without a red test.
"""

import warnings

import numpy as np
import pytest

from shd import SpectralHologramDistiller


def _stream(dist, n_normal=18000, n_late=2000, seed=3):
    """Normal regime, then a shifted regime in the last 10% of the stream."""
    rng = np.random.default_rng(seed)
    X_n = rng.normal(size=(n_normal, 4))
    y_n = rng.normal(size=n_normal)
    X_l = rng.normal(size=(n_late, 4)) + 6.0
    y_l = rng.normal(size=n_late) + 8.0
    for X, y in ((X_n, y_n), (X_l, y_l)):
        for i in range(0, len(X), 1000):
            dist.partial_fit(X[i:i + 1000], y[i:i + 1000])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dist.finalize_crystal()
    return dist


def test_interval_zero_is_prefix_only():
    # The published default: no reservoir, no rebuild. The late regime may
    # snap a couple of medoids through the online nudge, but it gets no
    # dedicated lattice cells (60k-row probe: 1/300 critical medoids).
    d = SpectralHologramDistiller(crystal_size=60, sketch_dim=64,
                                  random_state=3)
    _stream(d)
    assert d.lattice_rebuild_interval == 0
    assert d._reservoir_rows == 0
    n_late = int(np.sum(d.med_Y > 4.0))
    assert n_late <= 4, f"prefix-only contract broken: {n_late}/60 late medoids"


def test_interval_positive_covers_the_late_regime():
    d = SpectralHologramDistiller(crystal_size=60, sketch_dim=64,
                                  random_state=3,
                                  lattice_rebuild_interval=1500)
    _stream(d)
    assert d._reservoir_rows > 0
    n_late = int(np.sum(d.med_Y > 4.0))
    assert n_late >= 10, f"late regime under-covered: {n_late}/60 medoids"


def test_finalize_warns_when_prefix_only_stream_outgrew_warmup():
    d = SpectralHologramDistiller(crystal_size=60, sketch_dim=32,
                                  random_state=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 4))
    y = rng.normal(size=6000)
    d.partial_fit(X, y)
    with pytest.warns(UserWarning, match="lattice_rebuild_interval=0"):
        d.finalize_crystal()


def test_finalize_silent_on_short_stream_and_with_rebuild_enabled():
    for interval, n in ((0, 2000), (1500, 6000)):
        d = SpectralHologramDistiller(crystal_size=60, sketch_dim=32,
                                      random_state=0,
                                      lattice_rebuild_interval=interval)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 4))
        y = rng.normal(size=n)
        d.partial_fit(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            d.finalize_crystal()
        assert not any("lattice_rebuild_interval=0" in str(w.message)
                       for w in caught), f"interval={interval}, n={n}"


def test_docstring_states_the_true_default():
    doc = SpectralHologramDistiller.__doc__
    assert "``0`` (the published default) NEVER rebuilds" in doc
    assert "covers the whole stream" not in doc
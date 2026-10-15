"""Smoke tests for the SHD package: distill -> save -> load -> OOD radar.

Runs the full v3.7 pipeline on a small synthetic dataset with a rare
phase-peak regime, then round-trips the artifact through HologramCrystalLoader.
"""

import os
import sys

import numpy as np

try:
    import pytest
except ImportError:  # dependency-free mode: run via tests/run_smoke.py
    pytest = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import HologramCrystalLoader, SpectralHologramDistiller, __version__  # noqa: E402


def make_stream(n=6000, seed=7):
    """4D stream with a rare nonlinear phase-peak regime in y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    # Rare regime: |x0| > 1.8 triggers a phase transition (hot band).
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    return X, y


def make_distilled(tmp_path_factory):
    """Build a small crystal, save it, and return (distiller, path, rel_err, kb)."""
    tmp = tmp_path_factory.mktemp("shd")
    X, y = make_stream()
    d = SpectralHologramDistiller(
        crystal_size=80, high_band_ratio=0.26, sketch_dim=64, tail_quantile=84.0
    )
    B = 1500
    for i in range(0, len(X), B):
        d.partial_fit(X[i : i + B], y[i : i + B])
    rel_err = d.finalize_crystal()
    path = str(tmp / "test.hologram.npz")
    kb = d.save_hologram(path)
    return d, path, rel_err, kb


if pytest is not None:

    @pytest.fixture(scope="module")
    def distilled(tmp_path_factory):
        return make_distilled(tmp_path_factory)


def test_version():
    assert __version__ == "3.7.0"


def test_distiller_shapes(distilled):
    d, _, rel_err, _ = distilled
    assert d.crystal_X.shape == (80, 4)
    assert d.crystal_Y.shape == (80,)
    assert d.gamma_weights.shape == (80,)
    assert d.gamma_weights.size == 80
    # Spectral error should be modest on this toy landscape.
    assert rel_err < 0.60
    # Rare regime must be captured by the crystal (hot rows are ~2% of the stream;
    # 26% of cells sit in the hot band, so >= 5 hot medoids is a weak-but-real check).
    X, y = make_stream()
    hot_share = np.mean(np.abs(X[:, 0]) > 1.8)
    hot_medoids = int(np.sum(np.abs(d.crystal_X[:, 0]) > 1.8))
    assert hot_medoids > max(3, int(80 * hot_share))


def test_save_load_roundtrip(distilled):
    d, path, _, kb = distilled
    assert kb > 0 and os.path.exists(path)
    loader = HologramCrystalLoader(path)
    assert loader.X.shape == d.crystal_X.shape
    assert loader.n_seen == d.n_seen
    Xs, ys, ws = loader.get_training_data(budget_points=30)
    assert Xs.shape[0] == 30 and ws.shape[0] == 30
    assert abs(np.mean(ws) - 1.0) < 1e-9
    Xp, yp, wp = loader.get_phase_peak_slice(high_band_ratio=0.25)
    assert Xp.shape == (20, 4) and yp.shape == (20,)
    assert abs(np.mean(wp) - 1.0) < 1e-9
    assert float(np.median(yp)) > float(np.median(loader.y))


def test_ood_radar(distilled):
    _, path, _, _ = distilled
    loader = HologramCrystalLoader(path)  # projector stored -> self-contained
    X, _ = make_stream(n=2000, seed=99)
    ok = loader.check_domain_shift(X)
    assert ok["is_shifted"] is False
    drift = X.copy()
    drift[:, 0:2] = drift[:, 0:2] * 1.65 + 1.25
    bad = loader.check_domain_shift(drift)
    assert bad["is_shifted"] is True
    assert bad["bw_distance"] > ok["bw_distance"]


def test_legacy_artifact_without_projector():
    """The published superconduct_v37 artifact has no Omega/phase: loader must
    load it and only require a distiller_ref for the radar."""
    art = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts",
        "superconduct_v37.hologram.npz",
    )
    if not os.path.exists(art):
        if pytest is not None:
            pytest.skip("artifact not present")
        print("SKIP  test_legacy_artifact_without_projector (artifact not present)")
        return
    loader = HologramCrystalLoader(art)
    assert loader.X.shape[1] == 81
    assert loader.n_seen == 17010
    if pytest is not None:
        with pytest.raises(ValueError):
            loader.check_domain_shift(np.zeros((10, 81)))
    else:
        try:
            loader.check_domain_shift(np.zeros((10, 81)))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for legacy artifact without distiller_ref")


def test_unsupervised_label_free_mode():
    """partial_fit(X, y_batch=None) must capture rare sensor anomalies via Bochner
    spectral leverage without requiring external labels."""
    rng = np.random.default_rng(123)
    X_norm = rng.normal(0.0, 1.0, size=(3800, 4))
    # Inject two distinct rare anomalous sensor regimes (2.5% each = 5% total)
    X_anom_pos = rng.normal(4.5, 0.80, size=(100, 4))
    X_anom_neg = rng.normal(-4.5, 0.80, size=(100, 4))
    X = np.vstack([X_norm, X_anom_pos, X_anom_neg])
    perm = rng.permutation(len(X))
    X = X[perm]

    d = SpectralHologramDistiller(
        crystal_size=80,
        high_band_ratio=0.25,
        sketch_dim=64,
        tail_quantile=92.0,
        max_init_buffer=2500,
    )
    for i in range(0, len(X), 1000):
        d.partial_fit(X[i : i + 1000], y_batch=None)
    rel_err = d.finalize_crystal()
    assert d.unsupervised_mode is True
    assert d.crystal_X.shape == (80, 4)
    assert d.crystal_Y.shape == (80,)
    assert rel_err < 0.65
    # Both rare regimes (> 2.8 and < -2.8 on axis 0) must be captured, and their
    # synthetic leverage scores in crystal_Y must exceed the background median.
    pos_mask = d.crystal_X[:, 0] > 2.8
    neg_mask = d.crystal_X[:, 0] < -2.8
    anom_mask = pos_mask | neg_mask
    anom_medoids = int(np.sum(anom_mask))
    assert int(np.sum(pos_mask)) >= 3 and int(np.sum(neg_mask)) >= 3
    assert anom_medoids >= 8, f"Expected >= 8 anomaly medoids, got {anom_medoids}"
    assert float(np.median(d.crystal_Y[anom_mask])) > float(np.median(d.crystal_Y[~anom_mask]))


def test_max_init_buffer_cap():
    """Warm-up buffer must build lattice once max_init_buffer is reached instead
    of accumulating M * 12 rows unboundedly."""
    rng = np.random.default_rng(77)
    d = SpectralHologramDistiller(
        crystal_size=200, sketch_dim=32, max_init_buffer=600
    )
    assert d.max_init_buffer == 600  # M * 12 would be 2400 without the cap
    X = rng.normal(size=(700, 3))
    y = rng.normal(size=700)
    d.partial_fit(X[:350], y[:350])
    assert d.lattice_ready is False
    d.partial_fit(X[350:700], y[350:700])
    assert d.lattice_ready is True
    assert d.init_X == []


def test_heal_domain_shift(distilled):
    """heal_domain_shift must leaves in-distribution streams untouched and
    surgically inject medoids + lower BW distance on shifted streams."""
    _, path, _, _ = distilled
    loader = HologramCrystalLoader(path)
    X_in, y_in = make_stream(n=1200, seed=55)
    res_in = loader.heal_domain_shift(X_in, y_in, n_inject=20)
    assert res_in["healed"] is False
    assert res_in["n_injected"] == 0
    assert len(loader.X) == 80

    X_old_copy = loader.X.copy()
    X_shift = X_in.copy()
    X_shift[:, 0:2] = X_shift[:, 0:2] * 1.75 + 1.50
    y_shift = y_in + 8.0
    res_shift = loader.heal_domain_shift(X_shift, y_shift, n_inject=20)
    assert res_shift["healed"] is True
    assert res_shift["n_injected"] == 20
    assert len(loader.X) == 100
    # Zero catastrophic forgetting: first 80 medoids are bitwise identical
    assert np.array_equal(loader.X[:80], X_old_copy)
    assert abs(np.mean(loader.weights) - 1.0) < 1e-9
    assert res_shift["bw_after"] < res_shift["bw_before"]


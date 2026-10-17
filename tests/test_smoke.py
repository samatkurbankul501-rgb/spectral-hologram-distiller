"""Smoke tests for the SHD package: distill -> save -> load -> OOD radar.

Runs the full v3.8 pipeline (two-tailed trigger, optional adaptive y-stats,
opt-in streaming delta continuity) on small synthetic datasets with rare
phase-peak regimes, then round-trips the artifacts through HologramCrystalLoader.
"""

import os
import sys
import tempfile

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


def make_stream_low(n=6000, seed=11):
    """4D stream with a rare critical regime in the LOW tail of y
    (e.g. a pressure sag): |x1| > 1.8 drives y down by ~6."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    sag = np.abs(X[:, 1]) > 1.8
    y[sag] -= 6.0 + 1.5 * X[sag, 3]
    return X, y


def make_stream_both(n=6000, seed=13):
    """4D stream with rare regimes in BOTH tails: hot (+6 on |x0| > 1.8)
    and sag (-6 on |x1| > 1.8)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    sag = np.abs(X[:, 1]) > 1.8
    y[sag] -= 6.0 + 1.5 * X[sag, 3]
    return X, y


def _fit_crystal(X, y, tail, crystal_size=80, seed=3):
    d = SpectralHologramDistiller(
        crystal_size=crystal_size, high_band_ratio=0.26, sketch_dim=64,
        tail_quantile=84.0, tail=tail, random_state=seed,
    )
    B = 1500
    for i in range(0, len(X), B):
        d.partial_fit(X[i : i + B], y[i : i + B])
    d.finalize_crystal()
    return d


def test_version():
    assert __version__ == "3.9.2"


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


def test_loader_save_and_cli_commands(distilled):
    """loader.save() and the `shd` CLI subcommands (inspect, distill, radar, heal)
    must round-trip crystals cleanly without pandas."""
    import json
    import tempfile
    from shd.cli import main as cli_main

    _, path, _, _ = distilled
    loader = HologramCrystalLoader(path)
    info = loader.summary()
    assert info["n_medoids"] == 80
    assert info["has_projector"] is True

    tmp_dir = tempfile.mkdtemp(prefix="shd_cli_")
    csv_in = os.path.join(tmp_dir, "stream.csv")
    csv_drift = os.path.join(tmp_dir, "drift.csv")
    cryst_out = os.path.join(tmp_dir, "cli.hologram.npz")
    healed_out = os.path.join(tmp_dir, "healed.hologram.npz")

    X, y = make_stream(n=1600, seed=21)
    np.savetxt(csv_in, np.column_stack([X, y]), delimiter=",", header="x0,x1,x2,x3,target", comments="")

    X_dr = X.copy()
    X_dr[:, 0:2] = X_dr[:, 0:2] * 1.8 + 1.6
    np.savetxt(csv_drift, np.column_stack([X_dr, y + 5.0]), delimiter=",")

    assert cli_main([
        "distill",
        "--input", csv_in,
        "--output", cryst_out,
        "--target-col", "-1",
        "--crystal-size", "40",
        "--sketch-dim", "32",
        "--batch-size", "800",
    ]) == 0
    assert os.path.exists(cryst_out)

    assert cli_main(["inspect", cryst_out]) == 0
    assert cli_main(["radar", "--crystal", cryst_out, "--input", csv_drift, "--target-col", "-1"]) == 0
    assert cli_main([
        "heal",
        "--crystal", cryst_out,
        "--input", csv_drift,
        "--output", healed_out,
        "--target-col", "-1",
        "--n-inject", "12",
    ]) == 0

    healed_loader = HologramCrystalLoader(healed_out)
    assert len(healed_loader.X) == 52


def test_adaptive_threshold_and_small_batch(distilled):
    """check_domain_shift must expose a batch-size-calibrated threshold and not
    raise false alarms on small in-distribution windows (adaptive_threshold=True)."""
    _, path, _, _ = distilled
    loader = HologramCrystalLoader(path)
    X_in, _ = make_stream(n=2000, seed=31)

    rep_small = loader.check_domain_shift(X_in[:64])
    rep_big = loader.check_domain_shift(X_in)
    assert rep_small["threshold"] > rep_big["threshold"] >= 0.085
    assert rep_small["is_shifted"] is False
    assert rep_big["is_shifted"] is False

    drift = X_in.copy()
    drift[:, 0:2] = drift[:, 0:2] * 1.75 + 1.45
    rep_drift = loader.check_domain_shift(drift[:64])
    assert rep_drift["is_shifted"] is True
    assert rep_drift["bw_distance"] > rep_small["bw_distance"]


def test_heal_max_crystal_size_prunes_background(distilled):
    """heal_domain_shift with max_crystal_size must prune only low-weight
    background medoids while preserving the top-26% phase-peak medoids."""
    _, path, _, _ = distilled
    loader = HologramCrystalLoader(path)
    X_in, y_in = make_stream(n=1200, seed=66)
    old_peak_idx = np.argsort(loader.y)[-int(round(0.26 * len(loader.y))):]
    old_peak_X = loader.X[old_peak_idx].copy()

    X_shift = X_in.copy()
    X_shift[:, 0:2] = X_shift[:, 0:2] * 1.8 + 1.6
    res = loader.heal_domain_shift(
        X_shift, y_in + 8.0, n_inject=25, max_crystal_size=90, random_state=3
    )
    assert res["healed"] is True
    assert res["n_injected"] == 25
    # 80 old + 25 injected = 105 medoids; cap 90 -> exactly 15 pruned.
    assert res["n_pruned"] == 15
    assert len(loader.X) == 90
    # All old phase-peak medoids must survive the pruning
    kept = set(map(tuple, np.round(loader.X, 6)))
    for row in old_peak_X:
        assert tuple(np.round(row, 6)) in kept
    assert abs(np.mean(loader.weights) - 1.0) < 1e-9
    assert res["bw_after"] < res["bw_before"]


def test_heal_max_crystal_size_is_hard_cap(distilled):
    """The cap is a hard contract: final size <= cap, and an unsatisfiable cap
    raises before mutating the loader."""
    _, path, _, _ = distilled
    X_in, y_in = make_stream(n=1200, seed=66)
    X_shift = X_in.copy()
    X_shift[:, 0:2] = X_shift[:, 0:2] * 1.8 + 1.6

    loader = HologramCrystalLoader(path)
    res = loader.heal_domain_shift(
        X_shift, y_in + 8.0, n_inject=25, max_crystal_size=60, random_state=4
    )
    assert res["healed"] is True
    assert res["n_injected"] == 25
    # 80 old + 25 new = 105; protected = 25 new + 21 old peak = 46.
    assert res["n_pruned"] == 45
    assert len(loader.X) == 60
    assert len(loader.X) <= 60

    # 25 new + 21 protected old = 46 > 30: must fail loudly, before mutation.
    loader2 = HologramCrystalLoader(path)
    try:
        loader2.heal_domain_shift(
            X_shift, y_in + 8.0, n_inject=25, max_crystal_size=30, random_state=5
        )
    except ValueError as exc:
        assert "max_crystal_size=30" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unsatisfiable max_crystal_size")
    assert len(loader2.X) == 80


def test_cli_npz_target_col():
    """NPZ files without an explicit y must honour --target-col."""
    from shd.cli import _load_tabular

    tmp = tempfile.mkdtemp(prefix="shd_cli_npz_")
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)

    p1 = os.path.join(tmp, "features_and_target.npz")
    np.savez(p1, X=np.column_stack([X, y]))
    Xa, ya = _load_tabular(p1, target_col=-1)
    assert ya is not None and ya.shape == (10,)
    assert Xa.shape == (10, 2)
    assert np.allclose(ya, y)
    assert np.allclose(Xa, X)

    # An explicit y wins over --target-col; the feature matrix is unchanged.
    p2 = os.path.join(tmp, "features_and_y.npz")
    np.savez(p2, X=X, y=y)
    Xb, yb = _load_tabular(p2, target_col=0)
    assert yb is not None and Xb.shape == (10, 2)
    assert np.allclose(yb, y)
    assert np.allclose(Xb, X)

    # A 1-D .npy must be reshaped to a single feature column.
    p3 = os.path.join(tmp, "vector.npy")
    np.save(p3, y)
    Xc, yc = _load_tabular(p3)
    assert Xc.shape == (10, 1) and yc is None


def test_golden_crystal_numbers():
    """Regression lock for a fixed synthetic stream.

    Any refactor that silently changes medoid selection, FISTA weights, the
    passport rank, or the radar geometry will fail here. Values were recorded on
    the v3.9.1 default path; tolerances only absorb BLAS-level noise.
    """
    X, y = make_stream(n=8000, seed=123)
    d = SpectralHologramDistiller(
        crystal_size=120,
        high_band_ratio=0.26,
        sketch_dim=64,
        tail_quantile=84.0,
        phase_kappa=3.2,
        random_state=7,
    )
    for i in range(0, len(X), 1000):
        d.partial_fit(X[i : i + 1000], y[i : i + 1000])
    rel_err = d.finalize_crystal()

    assert len(d.crystal_X) == 120
    assert d.crystal_X.shape == (120, 4)
    assert abs(float(np.median(d.crystal_Y)) - 0.274023952995281) < 1e-7
    assert abs(float(np.median(d.gamma_weights)) - 0.944936234133797) < 1e-7
    assert abs(float(rel_err) - 0.08607388641467272) < 1e-7

    path = os.path.join(tempfile.mkdtemp(prefix="shd_golden_"), "golden.npz")
    d.save_hologram(path)
    loader = HologramCrystalLoader(path)
    summary = loader.summary()
    assert summary["n_medoids"] == 120
    assert summary["n_features"] == 4
    assert summary["n_seen"] == 8000
    assert summary["passport_rank"] == 32
    assert summary["has_projector"] is True

    X_shift = X[:200].copy()
    X_shift[:, 0] = X_shift[:, 0] * 2.0 + 4.0
    radar = loader.check_domain_shift(X_shift)
    assert radar["is_shifted"] is True
    assert abs(radar["bw_distance"] - 0.504485376709556) < 1e-3


def test_two_tailed_trigger():
    """tail='low'/'both' must be accepted, stored in the artifact, direct the
    loader's peak slice toward the matching extreme, and be ignored (forced
    'high') in label-free mode.

    Honest note (measured on these synthetic streams): capture of the matching
    extreme regime at crystal level is dominated by the residual trigger and
    the kappa*y phase geometry, so capture counts are NOT asserted to improve
    with the matching tail; the deterministic effects are asserted instead.
    """
    import tempfile

    X, y = make_stream_low()
    d = _fit_crystal(X, y, tail="low")
    assert d.crystal_X.shape == (80, 4)
    path = os.path.join(tempfile.mkdtemp(prefix="shd_tail_"), "low.hologram.npz")
    d.save_hologram(path)

    loader = HologramCrystalLoader(path)
    assert loader.tail == "low"
    assert loader.summary()["tail"] == "low"
    # the protected band follows the tail: peak slice = LOW-y medoids
    Xp, yp, wp = loader.get_phase_peak_slice(high_band_ratio=0.25)
    assert Xp.shape == (20, 4) and yp.shape == (20,)
    assert float(np.median(yp)) < float(np.median(loader.y))

    # tail='both' slice must contain BOTH extremes (hot and sag)
    Xb, yb = make_stream_both()
    db = _fit_crystal(Xb, yb, tail="both")
    path_b = os.path.join(tempfile.mkdtemp(prefix="shd_tail_"), "both.hologram.npz")
    db.save_hologram(path_b)
    lb = HologramCrystalLoader(path_b)
    assert lb.tail == "both"
    _, ypb, _ = lb.get_phase_peak_slice(high_band_ratio=0.25)
    assert int(np.sum(ypb > 3.0)) >= 2 and int(np.sum(ypb < -3.0)) >= 2

    # label-free mode forces tail='high' regardless of the option:
    # identical seeds -> bitwise-equal crystals for tail='low' and tail='high'
    dl = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, tail="low", random_state=5, max_init_buffer=2500
    )
    dh = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, tail="high", random_state=5, max_init_buffer=2500
    )
    for i in range(0, len(X), 1500):
        dl.partial_fit(X[i : i + 1500], y_batch=None)
        dh.partial_fit(X[i : i + 1500], y_batch=None)
    dl.finalize_crystal()
    dh.finalize_crystal()
    assert dl.unsupervised_mode and dh.unsupervised_mode
    assert np.allclose(dl.crystal_X, dh.crystal_X)

    # invalid tail is rejected
    try:
        SpectralHologramDistiller(tail="sideways")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for tail='sideways'")


def test_adaptive_y_stats():
    """adaptive_y_stats=True must track a target mean-shift after the lattice
    is built; the default must keep the frozen first-batch statistics."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(4000, 3))
    y = np.concatenate([rng.normal(0, 1, 2000), rng.normal(10, 1, 2000)])

    d_frozen = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, max_init_buffer=1500, random_state=1
    )
    d_adapt = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, max_init_buffer=1500,
        adaptive_y_stats=True, random_state=1,
    )
    B = 1000
    for i in range(0, 4000, B):
        d_frozen.partial_fit(X[i : i + B], y[i : i + B])
        d_adapt.partial_fit(X[i : i + B], y[i : i + B])
    assert d_frozen.lattice_ready and d_adapt.lattice_ready
    # frozen: first-batch statistics (measured y_mean ~ -0.05)
    assert abs(d_frozen.y_mean) < 0.30
    # adaptive: EMA tracked the +10 drift in the post-warm-up batches
    # (measured y_mean ~ 1.86 with the min(0.1, B/n_seen) rate)
    assert d_adapt.y_mean > 1.0
    assert d_adapt.y_std > 0.5
    rel_f = d_frozen.finalize_crystal()
    rel_a = d_adapt.finalize_crystal()
    assert rel_f > 0 and rel_a > 0
    assert d_frozen.crystal_X.shape == d_adapt.crystal_X.shape == (40, 3)


def test_loader_streaming_delta_continuity():
    """streaming=True must carry the last raw row across radar calls for a
    delta crystal (cross-call dX[0] continuity); default calls stay stateless
    and non-delta crystals ignore the flag."""
    import tempfile

    rng = np.random.default_rng(5)
    n = 2500
    t = np.arange(n)
    X = np.column_stack([
        np.sin(0.05 * t) + 0.05 * rng.normal(size=n),
        np.cos(0.03 * t) + 0.05 * rng.normal(size=n),
        rng.normal(0.0, 0.2, size=n),
    ])
    X[1500:, 0] += 3.0

    d = SpectralHologramDistiller(
        crystal_size=60, high_band_ratio=0.30, sketch_dim=48, tail_quantile=88.0,
        include_deltas=True, max_init_buffer=1200, random_state=5,
    )
    for i in range(0, n, 500):
        d.partial_fit(X[i : i + 500], y_batch=None)
    d.finalize_crystal()
    path = os.path.join(tempfile.mkdtemp(prefix="shd_delta_"), "delta.hologram.npz")
    d.save_hologram(path)

    loader = HologramCrystalLoader(path)
    loader.check_domain_shift(X[:1000], streaming=True)
    assert loader._prev_row is not None
    assert np.allclose(loader._prev_row, X[999])
    loader.check_domain_shift(X[1000:2000], streaming=True)
    assert np.allclose(loader._prev_row, X[1999])

    # default calls are stateless windows
    loader2 = HologramCrystalLoader(path)
    loader2.check_domain_shift(X[:1000])
    assert loader2._prev_row is None

    # non-delta crystals ignore the streaming flag entirely
    Xn, yn = make_stream(n=2000, seed=4)
    dn = SpectralHologramDistiller(crystal_size=30, sketch_dim=32, random_state=7)
    dn.partial_fit(Xn[:1000], yn[:1000])
    dn.partial_fit(Xn[1000:2000], yn[1000:2000])
    dn.finalize_crystal()
    path_n = os.path.join(tempfile.mkdtemp(prefix="shd_delta_"), "plain.hologram.npz")
    dn.save_hologram(path_n)
    loader3 = HologramCrystalLoader(path_n)
    loader3.check_domain_shift(Xn[:1000], streaming=True)
    assert loader3._prev_row is None


def test_heal_protects_low_tail_band():
    """With a tail='low' crystal, the heal size-cap must prune only background
    medoids and protect the LOW-y phase-peak band (bottom 26% of old medoids)."""
    import tempfile

    X, y = make_stream_low()
    d = _fit_crystal(X, y, tail="low")
    path = os.path.join(tempfile.mkdtemp(prefix="shd_heal_"), "low.hologram.npz")
    d.save_hologram(path)

    loader = HologramCrystalLoader(path)
    n_old = len(loader.X)
    low_band_idx = np.argsort(loader.y)[: int(round(0.26 * n_old))]
    low_band_X = loader.X[low_band_idx].copy()

    X_dr = X[:1200].copy()
    X_dr[:, 0:2] = X_dr[:, 0:2] * 1.8 + 1.6
    y_dr = y[:1200] + 8.0
    res = loader.heal_domain_shift(
        X_dr, y_dr, n_inject=25, max_crystal_size=90, random_state=3
    )
    assert res["healed"] is True
    assert res["n_injected"] == 25
    # 80 old + 25 injected = 105; cap 90 -> exactly 15 background pruned
    assert res["n_pruned"] == 15
    assert len(loader.X) == 90
    # every low-band (protected) medoid survived the pruning
    kept = set(map(tuple, np.round(loader.X, 6)))
    for row in low_band_X:
        assert tuple(np.round(row, 6)) in kept
    assert abs(np.mean(loader.weights) - 1.0) < 1e-9
    assert res["bw_after"] < res["bw_before"]


def test_input_validation_and_degenerate_data():
    """Non-finite inputs, empty batches, shape mismatches, finalize-without-fit
    and fully constant data must produce clear behaviour, not NumPy internals."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = rng.normal(size=200)

    # NaN / Inf in features and targets -> ValueError naming the problem
    for bad, label in ((np.nan, "NaN"), (np.inf, "Inf")):
        Xb = X.copy()
        Xb[3, 1] = bad
        d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
        try:
            d.partial_fit(Xb, y)
        except ValueError as e:
            assert "NaN/Inf" in str(e) and "rows" in str(e), f"{label}: {e}"
        else:
            raise AssertionError(f"{label} in features was silently accepted")

    yb = y.copy()
    yb[5] = np.nan
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    try:
        d.partial_fit(X, yb)
    except ValueError as e:
        assert "NaN/Inf" in str(e)
    else:
        raise AssertionError("NaN target was silently accepted")

    # on_nonfinite='drop' skips bad rows instead
    Xd = X.copy()
    Xd[3, 1] = np.nan
    Xd[9, 0] = np.inf
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  on_nonfinite="drop")
    d.partial_fit(Xd, y)
    d.finalize_crystal()
    assert d.n_seen == len(X) - 2

    # empty batch is a valid no-op
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    d.partial_fit(np.empty((0, 4)), np.empty(0))
    assert d.n_seen == 0

    # length mismatch and width changes are rejected with explicit messages
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    try:
        d.partial_fit(X, y[:50])
    except ValueError as e:
        assert "200" in str(e) and "50" in str(e)
    else:
        raise AssertionError("X/y length mismatch accepted")
    d.partial_fit(X, y)
    try:
        d.partial_fit(rng.normal(size=(10, 7)), rng.normal(size=10))
    except ValueError as e:
        assert "features" in str(e)
    else:
        raise AssertionError("feature-width change accepted mid-stream")

    # finalize without data
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    try:
        d.finalize_crystal()
    except ValueError as e:
        assert "partial_fit" in str(e)
    else:
        raise AssertionError("finalize_crystal() without data did not raise")

    # fully constant data builds a valid crystal (used to crash in k-means++)
    d = SpectralHologramDistiller(crystal_size=10, sketch_dim=16, random_state=0)
    d.partial_fit(np.ones((100, 4)), np.ones(100))
    d.finalize_crystal()
    assert d.crystal_X.shape == (10, 4)
    assert np.all(np.isfinite(d.gamma_weights))

    # radar window validation
    import tempfile
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    d.partial_fit(X, y)
    d.finalize_crystal()
    path = os.path.join(tempfile.mkdtemp(prefix="shd_val_"), "c.hologram.npz")
    d.save_hologram(path)
    ld = HologramCrystalLoader(path)
    for window, needle in (
        (np.full((64, 4), np.nan), "NaN/Inf"),
        (rng.normal(size=(64, 7)), "features"),
        (np.empty((0, 4)), "empty"),
    ):
        try:
            ld.check_domain_shift(window)
        except ValueError as e:
            assert needle in str(e), f"expected {needle!r} in {e}"
        else:
            raise AssertionError(f"radar accepted bad window ({needle})")


def test_architecture_constants_are_wired():
    """Sweepable constants must actually reach the computation, while the
    default constructor keeps the published behaviour."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1200, 4))
    y = np.sin(2 * X[:, 0]) + 0.2 * rng.normal(size=1200)

    def fit(**kw):
        d = SpectralHologramDistiller(crystal_size=40, sketch_dim=32, random_state=3, **kw)
        for i in range(0, len(X), 400):
            d.partial_fit(X[i : i + 400], y[i : i + 400])
        d.finalize_crystal()
        return d

    base = fit()
    explicit = fit(phase_kappa=3.2, high_band_ratio=0.26, voronoi_temper=0.55,
                   herding_mix=0.75, center_lr=0.25, fista_l2_reg=0.20,
                   omega_low_scale=0.75, omega_high_scale=2.10)
    assert np.allclose(base.crystal_X, explicit.crystal_X), "explicit defaults must match implicit defaults"

    no_phase = fit(phase_kappa=0.0)
    assert not np.allclose(base.crystal_X, no_phase.crystal_X), "phase_kappa did not reach the lattice"

    diff_weights = fit(herding_mix=0.0)
    assert not np.allclose(base.gamma_weights, diff_weights.gamma_weights), "herding_mix did not reach the weights"

    try:
        SpectralHologramDistiller(on_nonfinite="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid on_nonfinite accepted")


def test_include_deltas_streaming_augmentation():
    """include_deltas=True must keep crystal_X in original d dims while the
    internal projector consumes the [X || dX] augmented space (2d features)."""
    rng = np.random.default_rng(5)
    n = 2500
    t = np.arange(n)
    X = np.column_stack([
        np.sin(0.05 * t) + 0.05 * rng.normal(size=n),
        np.cos(0.03 * t) + 0.05 * rng.normal(size=n),
        rng.normal(0.0, 0.2, size=n),
    ])
    # A sharp shock transient at t=1500: feature 0 jumps by 3.0 within one step
    X[1500:, 0] += 3.0

    d = SpectralHologramDistiller(
        crystal_size=60,
        high_band_ratio=0.30,
        sketch_dim=48,
        tail_quantile=88.0,
        include_deltas=True,
        max_init_buffer=1200,
    )
    for i in range(0, n, 500):
        d.partial_fit(X[i : i + 500], y_batch=None)
    d.finalize_crystal()
    assert d.crystal_X.shape == (60, 3)  # original feature space preserved
    assert d.Omega.shape[0] == 6         # projector consumed [X || dX]
    # The shock region (post-jump rows) must be captured by the crystal
    assert int(np.sum(d.crystal_X[:, 0] > 1.5)) >= 3


def test_partial_fit_copies_input_buffers():
    """The warm-up buffer must copy caller-owned X/y so a reused streaming buffer
    cannot mutate a crystal that has not been finalized yet."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = X[:, 0].copy()
    original_X = X.copy()

    d = SpectralHologramDistiller(
        crystal_size=12, sketch_dim=16, random_state=0, max_init_buffer=4500
    )
    d.partial_fit(X, y)
    assert not d.lattice_ready

    # Typical streaming pattern: the caller reuses its batch buffer in place.
    X[:] = 999.0
    y[:] = -999.0
    d.finalize_crystal()

    assert not np.allclose(d.crystal_X, 999.0)
    assert not np.allclose(d.crystal_Y, -999.0)
    dists = np.linalg.norm(original_X[:, None, :] - d.crystal_X[None, :, :], axis=2)
    assert np.all(np.min(dists, axis=0) < 1e-9)


def test_mode_switch_is_rejected():
    """A stream cannot switch between supervised and label-free mode midway."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 4))
    y = X[:, 0]

    d = SpectralHologramDistiller(crystal_size=10, sketch_dim=16, random_state=0)
    d.partial_fit(X, y)
    try:
        d.partial_fit(X, None)
    except ValueError as e:
        assert "cannot switch" in str(e)
    else:
        raise AssertionError("supervised -> label-free switch was silently accepted")

    dl = SpectralHologramDistiller(crystal_size=10, sketch_dim=16, random_state=0)
    dl.partial_fit(X)
    assert dl.unsupervised_mode is True
    try:
        dl.partial_fit(X, y)
    except ValueError as e:
        assert "cannot switch" in str(e)
    else:
        raise AssertionError("label-free -> supervised switch was silently accepted")


def test_delta_width_ambiguity_is_rejected():
    """A plain 2-feature crystal must reject a 1-feature window instead of
    silently reinterpreting it as a delta stream; a real delta crystal must
    accept both raw d and pre-augmented 2d windows."""
    import tempfile

    rng = np.random.default_rng(11)
    X = rng.normal(size=(600, 2))
    y = X[:, 0]

    d = SpectralHologramDistiller(
        crystal_size=20, sketch_dim=16, random_state=0, max_init_buffer=200
    )
    for i in range(0, len(X), 200):
        d.partial_fit(X[i : i + 200], y[i : i + 200])
    d.finalize_crystal()
    path = os.path.join(tempfile.mkdtemp(prefix="shd_width_"), "plain.hologram.npz")
    d.save_hologram(path)

    loader = HologramCrystalLoader(path)
    assert loader.include_deltas is False
    try:
        loader.check_domain_shift(rng.normal(size=(32, 1)))
    except ValueError as e:
        assert "features" in str(e)
    else:
        raise AssertionError("1-feature window was silently treated as a delta stream")

    dd = SpectralHologramDistiller(
        crystal_size=20,
        sketch_dim=16,
        random_state=0,
        include_deltas=True,
        max_init_buffer=400,
    )
    for i in range(0, len(X), 200):
        dd.partial_fit(X[i : i + 200], y[i : i + 200])
    dd.finalize_crystal()
    path_d = os.path.join(tempfile.mkdtemp(prefix="shd_width_"), "delta.hologram.npz")
    dd.save_hologram(path_d)

    loader_d = HologramCrystalLoader(path_d)
    assert loader_d.include_deltas is True
    assert np.isfinite(loader_d.check_domain_shift(rng.normal(size=(32, 2)))["bw_distance"])
    assert np.isfinite(loader_d.check_domain_shift(rng.normal(size=(32, 4)))["bw_distance"])

    # loader.save() must persist the flag across a second round-trip
    path_d2 = os.path.join(tempfile.mkdtemp(prefix="shd_width_"), "delta_roundtrip.hologram.npz")
    loader_d.save(path_d2)
    assert HologramCrystalLoader(path_d2).include_deltas is True



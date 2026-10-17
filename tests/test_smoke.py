"""Smoke tests for the SHD package: distill -> save -> load -> OOD radar.

Runs the full v3.8 pipeline (two-tailed trigger, optional adaptive y-stats,
opt-in streaming delta continuity) on small synthetic datasets with rare
phase-peak regimes, then round-trips the artifacts through HologramCrystalLoader.
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
    assert __version__ == "3.8.0"


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



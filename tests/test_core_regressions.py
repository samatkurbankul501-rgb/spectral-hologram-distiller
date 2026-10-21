"""Targeted regressions for the eight core/CLI defects repaired in v3.9.1.

Each block below is labelled with the defect it pins down. The tests assert
the *contract* that was broken (e.g. "the warm-up buffer never holds more than
max_init_buffer rows"), not a specific number, so they keep working when the
algorithm is tuned later. For the two defects where a number had to move
(defect 5, the exported-blend spectral error) the golden value lives in
``test_smoke.py::test_golden_crystal_numbers``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import SpectralHologramDistiller, HologramCrystalLoader


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _drive(d: SpectralHologramDistiller, X: np.ndarray, y: np.ndarray, batch: int = 500) -> None:
    for i in range(0, len(X), batch):
        d.partial_fit(X[i:i + batch], y[i:i + batch])


def _make_stream(n: int = 800, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.05, size=n)
    return X, y


class _BufferSpy:
    """Records the warm-up buffer size at every lattice build."""

    def __init__(self, d: SpectralHologramDistiller):
        self.d = d
        self.sizes: list[int] = []
        self.original = d._build_phase_manifold_lattice

    def __enter__(self):
        spy = self

        def wrapped():
            spy.sizes.append(int(sum(len(a) for a in spy.d.init_Z)))
            return spy.original()

        self.d._build_phase_manifold_lattice = wrapped
        return self

    def __exit__(self, *exc):
        self.d._build_phase_manifold_lattice = self.original
        return False


# --------------------------------------------------------------------------
# Defect 1: _seed_kmeans_pp with n_select <= 0 returned one stray centre,
#           so the lattice grew past crystal_size (M=20 -> 21 medoids for
#           high_band_ratio 0/1/.01, and M=1 -> 2 medoids).
# --------------------------------------------------------------------------

def test_seed_kmeans_pp_zero_budget_returns_empty():
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0)
    cand = np.arange(24, dtype=np.float64).reshape(8, 3)
    for n in (0, -1, -7):
        out = d._seed_kmeans_pp(cand, n)
        assert len(out) == 0, f"n_select={n} must select nothing, got {len(out)}"
    # a real budget still selects that many
    assert len(d._seed_kmeans_pp(cand, 5)) == 5
    assert len(d._seed_kmeans_pp(cand, 99)) == 8


@pytest.mark.parametrize("band", [0.0, 1.0, 0.01])
def test_crystal_size_is_a_hard_cap_at_extreme_band_ratios(band):
    X, y = _make_stream(n=900, seed=3)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  high_band_ratio=band)
    _drive(d, X, y)
    d.finalize_crystal()
    assert len(d.crystal_X) <= 20, f"crystal_size=20 with band={band} grew to {len(d.crystal_X)}"
    assert len(d.crystal_X) >= 19


def test_crystal_size_one_yields_exactly_one_medoid():
    X, y = _make_stream(n=600, seed=5)
    d = SpectralHologramDistiller(crystal_size=1, sketch_dim=16, random_state=0)
    _drive(d, X, y)
    d.finalize_crystal()
    assert len(d.crystal_X) <= 1, f"crystal_size=1 grew to {len(d.crystal_X)}"
    assert len(d.crystal_X) == 1


def test_invalid_lattice_parameters_are_rejected_up_front():
    with pytest.raises(ValueError, match="crystal_size"):
        SpectralHologramDistiller(crystal_size=0, sketch_dim=16)
    with pytest.raises(ValueError, match="sketch_dim"):
        SpectralHologramDistiller(crystal_size=10, sketch_dim=1)
    with pytest.raises(ValueError, match="high_band_ratio"):
        SpectralHologramDistiller(crystal_size=10, sketch_dim=16, high_band_ratio=1.5)
    with pytest.raises(ValueError, match="high_band_ratio"):
        SpectralHologramDistiller(crystal_size=10, sketch_dim=16, high_band_ratio=-0.2)
    # degenerate but legal: a single-medoid lattice
    SpectralHologramDistiller(crystal_size=1, sketch_dim=2)


# --------------------------------------------------------------------------
# Defect 2: the warm-up buffer ignored max_init_buffer for oversized batches.
#           A 2000-row batch with max_init_buffer=40 used to buffer all 2000
#           rows before building the lattice.
# --------------------------------------------------------------------------

def test_max_init_buffer_bounds_oversized_first_batch():
    X, y = _make_stream(n=2000, seed=13)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  max_init_buffer=40)
    with _BufferSpy(d) as spy:
        d.partial_fit(X, y)
        assert spy.sizes, "lattice was never built"
        assert max(spy.sizes) <= 40, (
            f"warm-up buffer held {max(spy.sizes)} rows with max_init_buffer=40"
        )
    # the split rows still stream through the online update, not get dropped
    assert d.n_seen == 2000, "splitting the batch lost rows"


def test_max_init_buffer_bounds_oversized_later_batch():
    X, y = _make_stream(n=3000, seed=17)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  max_init_buffer=40)
    with _BufferSpy(d) as spy:
        for i in range(0, len(X), 500):
            d.partial_fit(X[i:i + 500], y[i:i + 500])
            assert max(spy.sizes) <= 40
    assert d.n_seen == 3000


def test_small_batches_are_not_split_when_under_the_cap():
    """Ordinary batches that fit the cap keep the unchanged all-at-once path."""
    X, y = _make_stream(n=1200, seed=19)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  max_init_buffer=4000)
    with _BufferSpy(d) as spy:
        for i in range(0, len(X), 500):
            d.partial_fit(X[i:i + 500], y[i:i + 500])
        # never split: each observed buffer size is a whole batch multiple
        assert all(s % 500 == 0 for s in spy.sizes), spy.sizes
    assert d.n_seen == 1200


def test_split_does_not_change_golden_small_batch_numbering():
    """With a big cap the split path is never taken, so historic behaviour holds."""
    def _run():
        X, y = _make_stream(n=900, seed=23)
        d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=4,
                                      max_init_buffer=100000)
        _drive(d, X, y)
        return d.finalize_crystal()

    first = _run()
    for _ in range(3):
        assert _run() == first


# --------------------------------------------------------------------------
# Defect 3: adaptive y-statistics updated mean/std but centres_ZY kept the
#           last column in the OLD statistics, so new and old rows were
#           assigned in two different y-coordinate systems.
# --------------------------------------------------------------------------

def test_adaptive_y_stats_rebases_centers():
    """A drifted batch must move mean/std AND the stored centre coordinates.

    The old code updated y_mean/y_std but left the last column of centers_ZY in
    the previous statistics, so subsequent rows were assigned into a lattice
    built on a different y-scale than the incoming rows.
    """
    X, y = _make_stream(n=1200, seed=2)
    d = SpectralHologramDistiller(crystal_size=30, sketch_dim=16, random_state=2,
                                  adaptive_y_stats=True, high_band_ratio=0.25,
                                  center_lr=0.0, max_init_buffer=1000)
    _drive(d, X, y, batch=300)
    assert d.lattice_ready, "lattice must exist before the drift batch"
    assert hasattr(d, "centers_ZY") and d.centers_ZY is not None

    before = np.asarray(d.centers_ZY).copy()
    before_mean, before_std = d.y_mean, d.y_std
    assert np.isfinite(before).all()

    d.partial_fit(X[:200], y[:200] + 25.0)
    after = np.asarray(d.centers_ZY)
    assert np.isfinite(after).all()
    # the statistics really moved, so this is a rebase test and not a tautology
    assert d.y_mean != before_mean and d.y_std != before_std

    # Independent reference: back-transform each stored centre to its raw target
    # with the OLD statistics and re-project it with the NEW ones.
    n = min(len(before), len(after))
    for j in range(n):
        raw_y = before[j, -1] * before_std / d.kappa + before_mean
        expected_col = d.kappa * (raw_y - d.y_mean) / d.y_std
        assert after[j, -1] == pytest.approx(expected_col, rel=1e-9, abs=1e-9), (
            f"centre {j} was not rebased onto the new y-statistics: "
            f"{after[j, -1]} vs {expected_col}"
        )


def test_adaptive_y_stats_keeps_center_and_row_coordinates_consistent():
    """Same stream: adaptive stats must not misassign rows into stale cells."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(400, 4)), rng.normal(size=(400, 4)) + 5.0])
    y = np.concatenate([rng.normal(size=400), rng.normal(size=400) + 4.0])

    results = []
    for adaptive in (False, True):
        d = SpectralHologramDistiller(crystal_size=30, sketch_dim=16, random_state=7,
                                      adaptive_y_stats=adaptive, high_band_ratio=0.25)
        _drive(d, X, y, batch=200)
        results.append(d.finalize_crystal())
    assert all(np.isfinite(r) for r in results)


def test_adaptive_y_stats_with_default_center_lr():
    """Default center_lr must combine with the rebase without breaking."""
    X, y = _make_stream(n=1200, seed=29)
    d = SpectralHologramDistiller(crystal_size=24, sketch_dim=16, random_state=29,
                                  adaptive_y_stats=True, high_band_ratio=0.3,
                                  max_init_buffer=1000)
    _drive(d, X, y, batch=200)
    assert d.lattice_ready
    before_mean = d.y_mean
    d.partial_fit(X[:200], y[:200] + 25.0)
    assert d.y_mean != before_mean
    assert np.isfinite(np.asarray(d.centers_ZY)).all()
    assert np.isfinite(np.asarray(d.med_Y)).all()
    # the crystal still round-trips after the drift
    err = d.finalize_crystal()
    assert np.isfinite(err)
    assert d.crystal_Y.ndim == 1


def test_center_lr_default_path_still_updates_centers():
    """The online centre update (center_lr) must still work after the rebase."""
    X, y = _make_stream(n=1200, seed=31)
    d = SpectralHologramDistiller(crystal_size=24, sketch_dim=16, random_state=31,
                                  center_lr=0.25, high_band_ratio=0.25,
                                  max_init_buffer=1000)
    _drive(d, X, y, batch=200)
    before = np.asarray(d.centers_ZY).copy()
    d.partial_fit(X[:200] + 3.0, y[:200] + 3.0)
    after = np.asarray(d.centers_ZY)
    assert np.isfinite(after).all()
    assert np.isfinite(np.asarray(d.med_X)).all()
    # centres actually moved (this is the online centre-update path)
    assert not np.array_equal(before, after)


# --------------------------------------------------------------------------
# Defect 4: crystal_Y was squeeze()d, so a single-medoid lattice produced a
#           0-d array and broke save/load, the peak slice and heal.
# --------------------------------------------------------------------------

def test_single_medoid_crystal_roundtrip(tmp_path):
    X, y = _make_stream(n=700, seed=37)
    d = SpectralHologramDistiller(crystal_size=1, sketch_dim=16, random_state=37)
    _drive(d, X, y)
    d.finalize_crystal()

    assert d.crystal_Y.ndim == 1, f"crystal_Y must be (N,), got shape {d.crystal_Y.shape}"
    assert d.crystal_Y.shape == (len(d.crystal_X),)

    out = tmp_path / "single.npz"
    d.save_hologram(str(out))
    loader = HologramCrystalLoader(str(out))
    assert loader.y.shape == (1,), f"roundtrip changed crystal_Y shape: {loader.y.shape}"

    # peak slice and the loader's explain helpers must work
    peak = loader.get_phase_peak_slice()
    assert peak is not None
    # heal must work on the 1-medoid crystal
    res = loader.heal_domain_shift(X[:64], n_inject=1, random_state=0)
    assert res is not None


def test_multi_medoid_crystal_shape_unchanged():
    X, y = _make_stream(n=700, seed=41)
    d = SpectralHologramDistiller(crystal_size=12, sketch_dim=16, random_state=41)
    _drive(d, X, y)
    d.finalize_crystal()
    assert d.crystal_Y.ndim == 1
    assert d.crystal_Y.shape == (12,)


# --------------------------------------------------------------------------
# Defect 5: finalize_crystal returned the error of the intermediate pure
#           herding solution instead of the exported blended weights.
# --------------------------------------------------------------------------

def _expected_exported_error(d):
    psi_hat = d.Psi_sum / d.n_seen
    p = d.gamma_weights / float(np.sum(d.gamma_weights))
    psi = (d.crystal_U.T * p) @ d.crystal_U
    return float(np.linalg.norm(psi - psi_hat, ord="fro")
                 / np.linalg.norm(psi_hat, ord="fro"))


def test_finalize_returns_exported_blend_error():
    X, y = _make_stream(n=900, seed=43)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=43,
                                  high_band_ratio=0.26)
    _drive(d, X, y)
    returned = d.finalize_crystal()

    expected = _expected_exported_error(d)
    assert abs(returned - expected) < 1e-12, (
        f"finalize returned {returned} but the exported blend scores {expected}"
    )
    # and it is not the pure-herding intermediate
    assert abs(returned - d.herding_rel_error) > 1e-6, (
        "returned error still equals the pure-herding intermediate"
    )


def test_cli_reports_the_exported_error(tmp_path):
    from shd.cli import main as cli_main
    X, y = _make_stream(n=900, seed=47)
    csv = tmp_path / "d.csv"
    np.savetxt(csv, np.column_stack([X, y]), delimiter=",",
               header="f0,f1,f2,f3,target", comments="")
    out = tmp_path / "c.npz"
    rc = cli_main(["distill", "--input", str(csv), "--target-col", "-1",
                   "--crystal-size", "16", "--sketch-dim", "16",
                   "--seed", "43", "--output", str(out)])
    assert rc == 0
    loader = HologramCrystalLoader(str(out))
    assert np.isfinite(loader.weights).all()
    assert len(loader.X) <= 16
    # A single-row CSV must not be silently readable as a column dataset: the
    # CLI keeps the 4 feature columns (target already stripped by --target-col).
    assert loader.X.shape[1] == 4


def test_herding_error_exposed_under_explicit_name():
    X, y = _make_stream(n=600, seed=53)
    d = SpectralHologramDistiller(crystal_size=16, sketch_dim=16, random_state=53)
    _drive(d, X, y)
    returned = d.finalize_crystal()
    assert hasattr(d, "herding_rel_error")
    assert np.isfinite(d.herding_rel_error)
    assert returned != d.herding_rel_error


def _loader_from_distiller(d):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.npz")
        d.save_hologram(p)
        return HologramCrystalLoader(p)


def _shift_train():
    """Standard 400-row training window used for the radar tests."""
    return np.vstack([np.zeros((300, 81)), np.ones((100, 81))]).astype(np.float64)


def _radar_loader():
    train = _shift_train()
    yy = np.arange(400, dtype=np.float64)
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=32, random_state=0,
                                  max_init_buffer=500)
    _drive(d, train, yy, batch=100)
    d.finalize_crystal()
    return _loader_from_distiller(d)


def test_same_point_vector_and_matrix_decide_identically():
    """An in-domain training point must not be flagged by one form only.

    Previously the (d,) form calibrated its threshold on 81 "rows" instead of
    1, which flagged this exact training row as a domain shift while the
    identical (1, d) row correctly stayed silent.
    """
    loader = _radar_loader()
    pt = np.zeros(81)  # a literal training row
    as_vector = loader.check_domain_shift(pt)
    as_matrix = loader.check_domain_shift(pt.reshape(1, 81))
    assert as_vector["is_shifted"] == as_matrix["is_shifted"], (
        "the same point decided differently as (81,) vs (1, 81)"
    )
    assert not as_vector["is_shifted"], (
        "an in-domain training row was reported as a domain shift"
    )
    assert as_vector["bw_distance"] == pytest.approx(as_matrix["bw_distance"], abs=1e-12)


def test_domain_shift_threshold_uses_normalized_row_count():
    """An in-domain (1,d) batch and the equivalent (d,) point agree."""
    loader = _radar_loader()
    in_domain = np.zeros(81)
    v = loader.check_domain_shift(in_domain)
    m = loader.check_domain_shift(in_domain.reshape(1, 81))
    assert v["bw_distance"] == pytest.approx(m["bw_distance"], abs=1e-12)
    assert v["is_shifted"] == m["is_shifted"]


def test_domain_shift_multi_row_batches_agree():
    """A batch and its row-wise equivalent must give the same verdict."""
    loader = _radar_loader()
    batch = np.vstack([np.zeros((5, 81)), np.full((1, 81), 4.0)])
    for pt in batch:
        v = loader.check_domain_shift(pt)
        m = loader.check_domain_shift(pt.reshape(1, 81))
        assert v["bw_distance"] == pytest.approx(m["bw_distance"], abs=1e-12)
        assert v["is_shifted"] == m["is_shifted"]


# --------------------------------------------------------------------------
# Defect 7: the heal precheck summed the protected band and the protected
#           weights instead of their union, so overlapping protections
#           (peak 5 + top-weight 5 = the same 5 medoids) wrongly required 10
#           extra slots and rejected a cap of 7.
# --------------------------------------------------------------------------

def test_heal_cap_counts_protected_union_not_sum():
    """peak5 + top-weight5 that are the SAME 5 medoids need 5 slots, not 10."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  high_band_ratio=0.25, max_init_buffer=500)
    _drive(d, X, np.arange(200, dtype=np.float64), batch=100)
    d.finalize_crystal()
    loader = _loader_from_distiller(d)
    assert len(loader.X) == 20

    # Force the overlap explicitly: the protected band (top-5 by target) must
    # also be the top-5 by gamma weight, so the union is 5, not 10.
    n_old = len(loader.X)
    band = max(1, int(round(loader.high_band_ratio * n_old)))
    peak_idx = np.argsort(loader.y)[-band:]
    weights = np.zeros(n_old)
    weights[peak_idx] = 1.0 + np.arange(band) * 0.1   # peak cells carry the mass
    loader.weights = weights

    # union == 5 medoids, so injected + union == 7 fits exactly
    res = loader.heal_domain_shift(
        rng.normal(size=(64, 4)), n_inject=2, max_crystal_size=7,
        protect_old_weight_frac=0.25, force=True, random_state=1,
    )
    assert res is not None
    assert len(loader.X) <= 7, f"cap violated: {len(loader.X)} > 7"


def test_heal_cap_one_slot_short_of_the_union_raises():
    """With the union counted, cap = injected + union - 1 must be rejected."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                                  high_band_ratio=0.25, max_init_buffer=500)
    _drive(d, X, np.arange(200, dtype=np.float64), batch=100)
    d.finalize_crystal()
    loader = _loader_from_distiller(d)

    n_old = len(loader.X)
    band = max(1, int(round(loader.high_band_ratio * n_old)))
    peak_idx = np.argsort(loader.y)[-band:]
    loader.weights = np.where(np.isin(np.arange(n_old), peak_idx), 1.0, 0.01)

    with pytest.raises(ValueError, match="max_crystal_size"):
        loader.heal_domain_shift(rng.normal(size=(64, 4)), n_inject=2,
                                 max_crystal_size=6, protect_old_weight_frac=0.25,
                                 force=True, random_state=1)


def test_heal_rejected_by_cap_does_not_mutate_loader():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=2)
    _drive(d, X, np.arange(200, dtype=np.float64), batch=100)
    d.finalize_crystal()
    loader = _loader_from_distiller(d)

    before = (loader.X.copy(), loader.y.copy(), loader.weights.copy())
    with pytest.raises(ValueError, match="max_crystal_size"):
        loader.heal_domain_shift(rng.normal(size=(64, 4)), n_inject=25,
                                 max_crystal_size=3, force=True, random_state=3)
    after = (loader.X, loader.y, loader.weights)
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])


def test_heal_rejected_by_cap_preserves_streaming_continuity_state():
    """A rejected heal must not advance the loader's streaming state either."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    d = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=3,
                                  include_deltas=True, max_init_buffer=500)
    _drive(d, X, np.arange(200, dtype=np.float64), batch=100)
    d.finalize_crystal()
    loader = _loader_from_distiller(d)
    # a delta crystal legitimately tracks streaming continuity
    loader.include_deltas = True
    prev_ptr = loader._prev_row

    before = (loader.X.copy(), loader.y.copy(), loader.weights.copy())
    with pytest.raises(ValueError, match="max_crystal_size"):
        loader.heal_domain_shift(rng.normal(size=(64, 4)), n_inject=25,
                                 max_crystal_size=3, force=True, random_state=4,
                                 streaming=True)
    assert np.array_equal(before[0], loader.X)
    assert np.array_equal(before[1], loader.y)
    assert np.array_equal(before[2], loader.weights)
    # streaming continuity state untouched by the rejected call
    assert loader._prev_row is prev_ptr


# --------------------------------------------------------------------------
# Defect 8: cli._load_tabular loaded a single-row .csv via np.loadtxt without
#           ndmin=2, so it became a 1-D feature array (a column dataset).
# --------------------------------------------------------------------------

def _load_tabular(path, target_col=None):
    from shd.cli import _load_tabular as _impl
    return _impl(path, target_col=target_col)


def test_load_tabular_single_row_csv_is_one_row(tmp_path):
    p = tmp_path / "one_row.csv"
    p.write_text("f0,f1,f2\n1.0,2.0,3.0\n")
    X, y = _load_tabular(str(p))
    assert X.shape == (1, 3), f"single-row CSV must be (1,3), got {X.shape}"
    assert y is None


def test_load_tabular_single_row_csv_with_target(tmp_path):
    p = tmp_path / "one_row_t.csv"
    p.write_text("f0,f1,target\n1.0,2.0,3.0\n")
    X, y = _load_tabular(str(p), target_col=-1)
    assert X.shape == (1, 2), f"single-row CSV must stay (1,2), got {X.shape}"
    assert y is not None and y.shape == (1,)
    assert y[0] == 3.0


def test_load_tabular_single_column_single_value_csv(tmp_path):
    p = tmp_path / "one_val.csv"
    p.write_text("5.0\n")
    X, y = _load_tabular(str(p))
    assert X.shape == (1, 1), f"single-value CSV must be (1,1), got {X.shape}"


def test_load_tabular_multi_row_csv_still_reshapes_by_column(tmp_path):
    p = tmp_path / "three.csv"
    p.write_text("1,2\n3,4\n5,6\n")
    X, y = _load_tabular(str(p))
    assert X.shape == (3, 2), f"3x2 CSV must stay (3,2), got {X.shape}"


def test_load_tabular_single_row_npy_still_one_column(tmp_path):
    p = tmp_path / "one_row.npy"
    np.save(p, np.array([1.0, 2.0, 3.0]))
    X, y = _load_tabular(str(p))
    assert X.shape == (3, 1), f"1-D npy must be a column (3,1), got {X.shape}"


def test_cli_end_to_end_single_row_csv(tmp_path):
    from shd.cli import main as cli_main
    p = tmp_path / "one.csv"
    p.write_text("f0,f1,f2,f3,target\n1.0,2.0,3.0,4.0,5.0\n")
    out = tmp_path / "c.npz"
    rc = cli_main(["distill", "--input", str(p), "--target-col", "-1",
                   "--crystal-size", "8", "--sketch-dim", "16",
                   "--seed", "0", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    loader = HologramCrystalLoader(str(out))
    assert len(loader.X) >= 1
    assert loader.X.shape[1] == 4


# --------------------------------------------------------------------------
# cross-cutting: nothing above may regress ordinary behaviour
# --------------------------------------------------------------------------

def test_heal_1d_stream_injects_and_increments_n_seen_by_one():
    """A 1-D observation vector must not crash heal with shape[1] IndexError,
    and n_seen must increment by 1 (the single row), not by the feature count."""
    d = SpectralHologramDistiller(crystal_size=5, sketch_dim=8, random_state=0)
    d.partial_fit(np.ones((20, 3)), np.ones(20))
    d.finalize_crystal()
    loader = _loader_from_distiller(d)
    n_before = loader.n_seen
    rep = loader.heal_domain_shift(np.array([2.0, 2.0, 2.0]), force=True, n_inject=1)
    assert rep["healed"] is True
    assert len(loader.X) == 6
    assert loader.n_seen == n_before + 1
    assert loader.X.shape == (6, 3)
    assert loader.y.shape == (6,)
    assert loader.weights.shape == (6,)


def test_ordinary_stream_still_finalizes():
    X, y = _make_stream(n=800, seed=59)
    d = SpectralHologramDistiller(crystal_size=24, sketch_dim=16, random_state=59)
    _drive(d, X, y)
    err = d.finalize_crystal()
    assert err > 0.0
    assert err < 0.60
    assert d.crystal_X.shape[1] == 4

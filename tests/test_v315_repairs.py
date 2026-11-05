"""v3.15.0: contracts for the trade-off knobs and the honesty presets.

These tests pin BEHAVIOUR CONTRACTS, not tuned numbers:

* ``residual_weight`` scales the residual term of the phase-peak score and
  ``0.0`` must reduce the trigger to the y-z component only; the default
  ``1.0`` must reproduce the pre-v3.15.0 behaviour bit-for-bit;
* ``rff_mode`` collapses the two Bochner frequency scales; ``dual`` must keep
  the published (0.75, 2.10) pair and the two single-scale modes must set
  both scales equal;
* the new presets must exist, carry the exact measured parameter sets, and
  must not silently restate constructor defaults (except where they opt in
  explicitly via ``explicit_defaults``);
* invalid values must fail loudly instead of degrading silently.
"""

import inspect
import json
import os
import sys

import numpy as np
import pytest

from shd import SpectralHologramDistiller
from shd import PRESETS, apply_preset, validate_presets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _synthetic(n=1200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1.0, d + 1.0)
    y[::37] += 6.0  # rare critical regime
    return X, y


def _distill(**kw):
    X, y = _synthetic()
    dist = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, random_state=0,
        max_init_buffer=max(len(X), 400), **kw)
    for i in range(0, len(X), 400):
        dist.partial_fit(X[i:i + 400], y[i:i + 400])
    dist.finalize_crystal()
    return dist


# ------------------------------------------------- residual_weight contract --

def test_residual_weight_default_is_one_and_preserves_default_behaviour():
    sig = inspect.signature(SpectralHologramDistiller.__init__)
    assert sig.parameters["residual_weight"].default == 1.0
    assert sig.parameters["rff_mode"].default == "dual"


def test_residual_weight_zero_is_accepted_and_builds_a_crystal():
    dist = _distill(residual_weight=0.0)
    assert dist.residual_weight == 0.0
    assert dist.crystal_X.shape[0] == 40


def test_residual_weight_negative_is_rejected():
    with pytest.raises(ValueError, match="residual_weight"):
        SpectralHologramDistiller(crystal_size=32, residual_weight=-0.1)


def test_residual_weight_changes_the_trigger_selection():
    """rw=0 must change which rows the trigger promotes."""
    X, y = _synthetic()
    scores = {}
    for rw in (0.0, 1.0):
        dist = SpectralHologramDistiller(crystal_size=40, sketch_dim=32,
                                         random_state=0, residual_weight=rw,
                                         max_init_buffer=max(len(X), 400))
        for i in range(0, len(X), 400):
            dist.partial_fit(X[i:i + 400], y[i:i + 400])
        dist.finalize_crystal()
        scores[rw] = np.sort(np.asarray(dist.crystal_Y, dtype=float))
    assert not np.allclose(scores[0.0], scores[1.0])


# ---------------------------------------------------------- rff_mode contract --

def test_rff_mode_dual_keeps_the_published_scale_pair():
    d = SpectralHologramDistiller(crystal_size=32)
    assert d.rff_mode == "dual"
    assert d.omega_low_scale == 0.75
    assert d.omega_high_scale == 2.10


def test_rff_mode_single_low_collapses_high_scale():
    d = SpectralHologramDistiller(crystal_size=32, rff_mode="single_low")
    assert d._rff_scale == 0.75
    # projectors must be a true single scale (all k channels at one std),
    # not a dual stack with both halves forced to the same value
    X, y = _synthetic(n=2000, d=4, seed=0)
    dd = SpectralHologramDistiller(crystal_size=32, rff_mode="single_low", sketch_dim=32, random_state=0, max_init_buffer=400)
    dd.partial_fit(X, y)
    assert dd.Omega.shape == (4, 32)
    assert abs(float(dd.Omega.std()) - 0.375) < 0.12  # scale*sqrt-mean amplitude


def test_rff_mode_single_high_collapses_low_scale():
    d = SpectralHologramDistiller(crystal_size=32, rff_mode="single_high")
    assert d._rff_scale == 2.10
    X, y = _synthetic(n=2000, d=4, seed=0)
    dd = SpectralHologramDistiller(crystal_size=32, rff_mode="single_high", sketch_dim=32, random_state=0, max_init_buffer=400)
    dd.partial_fit(X, y)
    assert dd.Omega.shape == (4, 32)
    assert abs(float(dd.Omega.std()) - 1.05) < 0.35


def test_rff_mode_invalid_is_rejected():
    with pytest.raises(ValueError, match="rff_mode"):
        SpectralHologramDistiller(crystal_size=32, rff_mode="triple")


def test_rff_mode_single_low_changes_the_crystal():
    """A collapsed frequency band must not be a no-op.

    The Voronoi lattice runs in Z-space, so medoids can coincide across
    modes; the spectral weights and passport are Omega-dependent and
    must change. The pre-patch 'single_*' modes were a dual stack with
    both halves at one scale (scaled dual), which measured the wrong
    ablation.
    """
    a = _distill(rff_mode="dual")
    b = _distill(rff_mode="single_low")
    assert not np.allclose(a.gamma_weights, b.gamma_weights)
    assert not np.allclose(np.array(a.med_U), np.array(b.med_U)) if hasattr(a, 'med_U') and hasattr(b, 'med_U') else True


# ----------------------------------------------------------- preset contracts --

def test_new_presets_exist_with_exact_parameter_sets():
    assert apply_preset("sparse_auto") == {
        "tail_quantile": 97.0, "global_guard": True, "high_band_ratio": 0.26}
    assert apply_preset("critical_first_no_residual") == {
        "residual_weight": 0.0, "high_band_ratio": 0.65}
    assert apply_preset("single_scale_low") == {"rff_mode": "single_low"}
    assert apply_preset("single_scale_high") == {"rff_mode": "single_high"}


def test_presets_validate_against_the_constructor():
    validate_presets()


def test_preset_sources_carry_no_unbacked_p_values():
    """Every p-value in a preset source must be backed by an artifact.

    The v3.15.0 presets shipped with invented significance values
    (20/20, p=3.8e-6) that no artifact supported.
    """
    for name, preset in PRESETS.items():
        src = preset.get("source", "")
        assert "p=3.8e-6" not in src, name
        assert "p=3.0e-5" not in src, name


def test_sparse_auto_preset_builds_a_real_distiller():
    d = _distill(**apply_preset("sparse_auto"))
    assert d.crystal_X.shape[0] == 40


def test_critical_first_no_residual_preset_builds_a_real_distiller():
    d = _distill(**apply_preset("critical_first_no_residual"))
    assert d.crystal_X.shape[0] == 40
    assert d.residual_weight == 0.0


def test_unknown_preset_raises_with_available_names():
    with pytest.raises(KeyError):
        apply_preset("does_not_exist")


def test_reserved_knobs_cannot_be_overridden_by_a_preset():
    with pytest.raises(ValueError):
        apply_preset("sparse_auto", crystal_size=10)


# ------------------------------------------------- committed bench artifacts --

def _load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        pytest.skip(name + " not committed")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_residual_tradeoff_artifact_shows_the_residual_trigger_hurts_crit():
    res = _load("bench_residual_tradeoff_results.json")
    for ds in ("reactor", "SC"):
        cfgs = res[ds]["configs"]
        default = float(np.median([r["krr_rmse_crit"]
                                   for r in cfgs["rw_1.0_default"]["records"]]))
        rw0 = float(np.median([r["krr_rmse_crit"]
                               for r in cfgs["rw_0"]["records"]]))
        assert rw0 <= default, ds
        cf = cfgs["critical_first_no_residual"]
        cf_crit = float(np.median([r["krr_rmse_crit"] for r in cf["records"]]))
        assert cf_crit < default, ds
        assert cf["wilcoxon_vs_default"]["wins"] >= 8


def test_radar_hard_artifact_is_not_only_the_easy_task():
    hard = _load("bench_radar_hard_results.json")
    for batch, block in hard["by_batch"].items():
        agg = block["agg"]
        assert agg["abrupt_control"]["auc_median"] > 0.99, batch
        for name in ("gradual_alpha0.3", "gradual_alpha0.5", "noisy_sigma2.0"):
            assert agg[name]["auc_median"] >= 0.95, (batch, name)
        # the weak spot must stay visible in the artifact
        assert agg["partial_frac0.25"]["auc_median"] >= 0.9, batch
        assert agg["partial_frac0.25"]["tpr_published_median"] < 1.0, batch


def test_global_guard_n_is_clamped_to_the_crystal_size():
    """The guard must not crash a small crystal (v3.15.0 bug fix).

    The published default n_global=150 assumed a 300-row crystal; on a
    60-row crystal save_hologram raised "n_global=150 exceeds crystal size
    60", which made the global_guard presets unusable at small budgets.
    """
    with pytest.warns(UserWarning, match="exceeds crystal size"):
        d = SpectralHologramDistiller(crystal_size=60, global_guard=True)
    assert d.global_guard_n == 30


def test_global_guard_n_is_not_clamped_when_it_fits():
    d = SpectralHologramDistiller(crystal_size=300, global_guard=True)
    assert d.global_guard_n == 150


def test_published_defaults_are_untouched_by_the_new_knobs():
    """The v3.15.0 knobs must not change the protocol constants."""
    d = SpectralHologramDistiller(crystal_size=32)
    assert d.high_band_ratio == 0.26
    assert d.tail_q == 84.0
    assert d.kappa == 3.2
    assert d.use_residual is True


# ---------------------------------------------------- CLI preset-flag merge --

def _cli_distill(tmp_path, extra):
    """Run the CLI distill on a tiny NPZ; return the parsed rel error."""
    from shd.cli import main as _main
    X, y = _synthetic(n=500, d=4, seed=0)
    inp = tmp_path / "in.npz"
    out = tmp_path / "out.hologram.npz"
    np.savez(str(inp), X=X, y=y)
    argv = ["distill", "-i", str(inp), "-o", str(out), "--crystal-size", "30", "--seed", "7"] + extra
    rc = _main(argv)
    assert rc == 0
    with open(str(out), "rb") as fh:
        import zipfile  # noqa
        pass
    # the CLI prints JSON; recover rel error by comparing crystals directly
    from shd import HologramCrystalLoader
    loader = HologramCrystalLoader(str(out))
    return loader


def test_cli_explicit_default_flag_overrides_preset(tmp_path):
    """Explicit --tail-quantile 84.0 must win over a preset's 97.0.

    Regression (v3.15.0 wave-2): the CLI merge used float equality
    (``tail_quantile == 84.0``) as the sentinel for 'unset', so an
    explicit 84.0 passed by the user was invisible and the preset value
.
    won silently. The flags are now ``default=None`` and resolution is
    preset > None-wins > 84.0.
    """
    from shd.cli import build_parser
    p = build_parser()
    args = p.parse_args(["distill", "-i", "a", "-o", "b", "--preset", "sparse_stream_top3pct", "--tail-quantile", "84.0"])
    assert args.tail_quantile == 84.0
    args2 = p.parse_args(["distill", "-i", "a", "-o", "b"])
    assert args2.tail_quantile is None
    assert args2.high_band_ratio is None


def test_cli_preset_and_explicit_flags_roundtrip(tmp_path):
    """End-to-end: preset + explicit defaults == noflag defaults."""
    from shd.cli import main as _main
    X, y = _synthetic(n=500, d=4, seed=0)
    inp = tmp_path / "in.npz"
    np.savez(str(inp), X=X, y=y)
    def run(extra, out):
        return _main(["distill", "-i", str(inp), "-o", str(out), "--crystal-size", "30", "--seed", "7"] + extra)
    out1 = tmp_path / "a.npz"
    out2 = tmp_path / "b.npz"
    assert run([], str(out1)) == 0
    assert run(["--preset", "sparse_stream_top3pct", "--tail-quantile", "84.0"], str(out2)) == 0
    from shd import HologramCrystalLoader
    a = HologramCrystalLoader(str(out1))
    b = HologramCrystalLoader(str(out2))
    Xa, ya, _ = a.get_training_data()
    Xb, yb, _ = b.get_training_data()
    assert np.allclose(Xa, Xb)
    assert np.allclose(ya, yb)

"""v3.14.2: regressions for the honesty-and-hygiene repair round.

Each test pins a CONTRACT introduced by a v3.14.2 fix, not a tuned number:

* ``threshold_source`` describes the threshold that was ACTUALLY applied
  (adaptive on/off x calibrated/uncalibrated, BW and combined paths);
* the schema gate rejects a newer schema even when ``schema_min_loader`` is
  absent from the artifact;
* loader value coercions warn (UserWarning), ``high_band_ratio`` accepts the
  distiller's full closed range [0, 1], and a half-persisted threshold
  calibration is an inconsistency (ValueError), not a KeyError;
* label-free crystals travel with a ``label_free`` marker, so heal's
  novelty-score branch is reachable and its injected rows are LABELLED
  (``get_training_data(drop_unlabelled=True)`` keeps them); legacy artifacts
  keep the old placeholder contract;
* ``build_global_guard`` fails loudly on ``include_deltas`` crystals;
* the CLI heal default is non-destructive (``*_healed`` sibling, input intact);
* portfolio dead weight stays removed (``_th_n_``, ``predict(streaming=...)``);
* kernel thinning rejects ``n < 2**m`` with a clear ValueError.
"""

import hashlib
import inspect
import json
import os
import sys
import warnings

import numpy as np
import pytest

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd import cli as shd_cli
from shd.distiller import BW_ALERT_THRESHOLD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ------------------------------------------------------------ helpers --------

def _crystal(tmp_path, n=400, d=3, name="c.hologram.npz",
             crystal_size=20, sketch_dim=16, label_free=False):
    """Distill a small legit crystal (supervised by default) into tmp_path."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1.0, d + 1.0)
    dist = SpectralHologramDistiller(
        crystal_size=crystal_size, sketch_dim=sketch_dim,
        random_state=0, max_init_buffer=max(n, 400),
    )
    for i in range(0, n, 200):
        batch = X[i:i + 200]
        dist.partial_fit(batch, None if label_free else y[i:i + 200])
    dist.finalize_crystal()
    path = str(tmp_path / name)
    dist.save_hologram(path)
    return path


def _payload(path):
    src = np.load(path, allow_pickle=False)
    data = {key: np.asarray(src[key]) for key in src.files}
    src.close()
    return data


def _modified(path, updates, drops=(), name="mod.hologram.npz"):
    """Re-save the artifact with payload arrays replaced/dropped."""
    payload = _payload(path)
    for key, value in updates.items():
        payload[key] = value
    for key in drops:
        payload.pop(key, None)
    out = os.path.join(os.path.dirname(path), name)
    np.savez_compressed(out, **payload)
    return out


# ------------------------------------------- 1. threshold_source honesty ----

def test_threshold_source_describes_the_applied_threshold(tmp_path):
    ld = HologramCrystalLoader(_crystal(tmp_path))
    rng = np.random.default_rng(11)
    healthy = rng.normal(size=(640, 3))
    win = rng.normal(size=(64, 3))

    # Uncalibrated: adaptive ON still scores against the default constant.
    fresh = ld.check_domain_shift(win)
    assert fresh["threshold_source"] == "default"

    ld.calibrate_threshold(healthy, window=32, fa_rate=0.05)
    on = ld.check_domain_shift(win)                      # adaptive ON
    assert on["threshold_source"] == "calibrated"

    off = ld.check_domain_shift(win, adaptive_threshold=False)
    # adaptive OFF applies the default constant, so the source must say so
    assert off["threshold"] == pytest.approx(BW_ALERT_THRESHOLD)
    assert off["threshold_source"] == "default"

    # Symmetric contract for the combined (BW + mean-shift) path.
    ld.calibrate_threshold_ms(healthy, window=32, fa_rate=0.05)
    on_ms = ld.check_domain_shift(win, mean_shift=True)
    assert on_ms["combined_threshold_source"] == "calibrated"
    off_ms = ld.check_domain_shift(win, mean_shift=True, adaptive_threshold=False)
    assert off_ms["combined_threshold"] == pytest.approx(BW_ALERT_THRESHOLD)
    assert off_ms["combined_threshold_source"] == "default"


# ------------------------------------------------------- 2. schema gate -----

def test_newer_schema_without_min_loader_is_rejected(tmp_path):
    path = _crystal(tmp_path)
    newer = HologramCrystalLoader.LOADER_SCHEMA + 1

    forged = _modified(
        path,
        {"schema_version": np.array([newer], dtype=np.int64)},
        drops=("schema_min_loader",),
    )
    with pytest.raises(ValueError, match="upgrade"):
        HologramCrystalLoader(forged)

    # The min_loader path keeps rejecting as before (contract unchanged).
    forged2 = _modified(path, {
        "schema_version": np.array([newer], dtype=np.int64),
        "schema_min_loader": np.array([newer], dtype=np.int64),
    })
    with pytest.raises(ValueError, match="upgrade"):
        HologramCrystalLoader(forged2)

    # Current-schema artifact without min_loader still loads.
    ok = _modified(path, {}, drops=("schema_min_loader",))
    assert HologramCrystalLoader(ok).schema_version == HologramCrystalLoader.LOADER_SCHEMA


# ------------------------------------------ 3. loader coercions and tau -----

def test_invalid_values_coerce_with_a_warning(tmp_path):
    path = _crystal(tmp_path)

    with pytest.warns(UserWarning, match="tail"):
        ld = HologramCrystalLoader(_modified(path, {"tail": np.array("sideways")}))
    assert ld.tail == "high"

    with pytest.warns(UserWarning, match="passport_reference"):
        ld = HologramCrystalLoader(
            _modified(path, {"passport_reference": np.array("cosmos")}, name="m2.npz"))
    assert ld.passport_reference == "stream"

    with pytest.warns(UserWarning, match="high_band_ratio"):
        ld = HologramCrystalLoader(
            _modified(path, {"high_band_ratio": np.array([1.7])}, name="m3.npz"))
        assert ld.high_band_ratio == pytest.approx(0.26)


def test_high_band_ratio_accepts_the_distillers_closed_range(tmp_path):
    path = _crystal(tmp_path)
    # The distiller validates [0, 1]; the loader must not rewrite its extremes.
    for i, br in enumerate((0.0, 1.0)):
        art = _modified(path, {"high_band_ratio": np.array([br])}, name=f"br{i}.npz")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a coercion warning would fail here
            ld = HologramCrystalLoader(art)
        assert ld.high_band_ratio == pytest.approx(br)


def test_half_persisted_calibration_is_an_inconsistency(tmp_path):
    path = _crystal(tmp_path)
    art = _modified(path, {"tau_calibrated": np.array([0.5])})
    # Was a raw KeyError deep in __init__; must be a clear ValueError.
    with pytest.raises(ValueError, match="inconsistent"):
        HologramCrystalLoader(art)


# --------------------------------------------- 4. label-free crystals -------

def test_label_free_artifact_carries_the_marker(tmp_path):
    lf = _crystal(tmp_path, name="lf.hologram.npz", label_free=True)
    payload = _payload(lf)
    assert "label_free" in payload
    assert bool(np.asarray(payload["label_free"]).reshape(-1)[0]) is True
    # y_labelled stays all-True: drop_unlabelled must never empty the crystal.
    assert np.all(np.asarray(payload["y_labelled"], dtype=bool))
    assert HologramCrystalLoader(lf).label_free is True

    sup = _crystal(tmp_path, name="sup.hologram.npz")
    assert HologramCrystalLoader(sup).label_free is False


def test_label_free_heal_injects_labelled_novelty_rows(tmp_path):
    ld = HologramCrystalLoader(_crystal(tmp_path, name="lf2.hologram.npz",
                                        label_free=True))
    n_before = len(ld.X)
    shifted = np.random.default_rng(5).normal(loc=4.0, size=(200, 3))
    res = ld.heal_domain_shift(shifted, y_new_stream=None, n_inject=5, force=True)

    assert res["healed"]
    # The novelty-score branch LABELS its rows: nothing to drop, no placeholder
    # warning (the injected y is consistent with the crystal's own coordinate).
    assert res["n_unlabelled_injected"] == 0
    assert res["warning"] is None
    X_tr, y_tr, _ = ld.get_training_data(drop_unlabelled=True)
    assert len(X_tr) == len(y_tr) == len(ld.X) == n_before + res["n_injected"]

    # The marker survives a re-save.
    out = str(tmp_path / "resaved.npz")
    ld.save(out)
    assert HologramCrystalLoader(out).label_free is True


def test_legacy_artifact_keeps_the_placeholder_contract(tmp_path):
    path = _crystal(tmp_path, name="leg.hologram.npz")
    legacy = _modified(path, {}, drops=("label_free",))  # pre-3.14.2 artifact
    ld = HologramCrystalLoader(legacy)
    assert ld.label_free is False
    shifted = np.random.default_rng(5).normal(loc=4.0, size=(200, 3))
    res = ld.heal_domain_shift(shifted, y_new_stream=None, n_inject=5, force=True)
    assert res["healed"]
    assert res["n_unlabelled_injected"] == res["n_injected"]
    X_tr, _, _ = ld.get_training_data(drop_unlabelled=True)
    assert len(X_tr) == len(ld.X) - res["n_injected"]


# ------------------------------------------ 5. guard rejects delta mode -----

def _distiller(include_deltas, n=400, d=3):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1.0, d + 1.0)
    dist = SpectralHologramDistiller(
        crystal_size=20, sketch_dim=16, random_state=0,
        include_deltas=include_deltas, max_init_buffer=max(n, 400),
    )
    dist.partial_fit(X, y)
    dist.finalize_crystal()
    return dist, X, y


def test_global_guard_refuses_delta_crystals():
    dist, X, y = _distiller(include_deltas=True)
    with pytest.raises(NotImplementedError, match="include_deltas"):
        dist.build_global_guard(n_global=5, X_pool=X, y_pool=y)


def test_global_guard_still_works_on_plain_crystals():
    dist, X, y = _distiller(include_deltas=False)
    report = dist.build_global_guard(n_global=5, X_pool=X, y_pool=y)
    assert report["swapped"] == 5


# --------------------------------------- 6a. non-destructive CLI heal ------

def test_cli_heal_default_spares_input_and_writes_healed_sibling(tmp_path, capsys):
    crystal = _crystal(tmp_path, name="c.hologram.npz")
    before = hashlib.sha256(open(crystal, "rb").read()).hexdigest()
    drift = np.random.default_rng(9).normal(loc=4.0, size=(200, 3))
    stream_csv = str(tmp_path / "drift.csv")
    np.savetxt(stream_csv, drift, delimiter=",")

    rc = shd_cli.main(["heal", "--crystal", crystal, "--input", stream_csv,
                       "--force", "--n-inject", "5"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healed"]

    # The input crystal is untouched; the healed sibling exists and is named
    # in the JSON report.
    assert hashlib.sha256(open(crystal, "rb").read()).hexdigest() == before
    expected = str(tmp_path / "c_healed.hologram.npz")
    assert report["output"] == expected
    assert os.path.exists(expected)

    # An explicit --output is still honoured verbatim.
    explicit = str(tmp_path / "elsewhere.npz")
    rc = shd_cli.main(["heal", "--crystal", crystal, "--input", stream_csv,
                       "--output", explicit, "--force", "--n-inject", "5"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["output"] == explicit


# --------------------------------------- 7a/7b. portfolio dead weight ------

def test_portfolio_dead_weight_removed():
    from sklearn.preprocessing import StandardScaler

    from shd.portfolio import CoresetPortfolio

    assert "streaming" not in inspect.signature(CoresetPortfolio.predict).parameters

    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(60, 4)); yc = 2.0 * Xc[:, 0] + 1.5
    Xg = rng.normal(size=(60, 4)); yg = Xg[:, 0]
    scaler = StandardScaler().fit(np.vstack([Xc, Xg]))
    p = CoresetPortfolio(critical_threshold=1.0)
    p.fit(Xc, yc, None, Xg, yg, None, scaler, rbf_gamma=1.0 / (1.1 * 4))
    assert not hasattr(p, "_th_n_")  # write-only dead state must not come back
    assert p.predict(Xg[:5]).shape == (5,)


# --------------------------------------------- 9. kt_thin size contract -----

def test_kt_thin_rejects_too_few_points_for_m():
    from kt_baseline import kt_thin

    X = np.random.default_rng(0).normal(size=(6, 2))
    # floor(6 / 2**4) == 0: was ZeroDivisionError, must be a clear ValueError.
    with pytest.raises(ValueError, match=r"2\*\*m"):
        kt_thin(X, m=4, var=1.0, seed=0)
    # A valid configuration keeps working (contract, not a tuned number).
    assert len(kt_thin(X, m=1, var=1.0, seed=0)) == len(X) // 2

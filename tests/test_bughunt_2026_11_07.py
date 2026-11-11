# -*- coding: utf-8 -*-
"""Regression tests for the 2026-11-07 bug hunt (7 confirmed defects).

Each test below fails against the code as it was BEFORE the fix; the
comment above the test names the defect it pins.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd.portfolio import ConformalCrystal, CoresetPortfolio
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


def _synthetic(n=3000, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=n)
    return X, y


# --- 1) a second finalize_crystal() silently discarded the global guard ---
def test_second_finalize_does_not_discard_the_global_guard(tmp_path):
    X, y = _synthetic(4000)

    def build(guard: bool):
        d = SpectralHologramDistiller(
            crystal_size=60, sketch_dim=32, random_state=0,
            global_guard=guard, global_guard_n=20,
        )
        for i in range(0, len(X), 1000):
            d.partial_fit(X[i:i + 1000], y[i:i + 1000])
        d.finalize_crystal()
        return d

    guarded = build(True)
    p1 = str(tmp_path / "g1.npz")
    guarded.save_hologram(p1)
    assert guarded._guard_report is not None
    assert guarded._guard_report["swapped"] == 20

    # the second finalize rebuilds the crystal from med_* -> the guard is gone
    guarded.finalize_crystal()
    p2 = str(tmp_path / "g2.npz")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        guarded.save_hologram(p2)

    # the guard must be re-applied, not skipped as "already applied"
    assert not any("already applied" in str(w.message) for w in rec), (
        "the second save skipped the guard on a crystal that was rebuilt"
    )
    plain = build(False)
    plain_crystal = plain.crystal_X.astype(np.float32)
    saved = np.load(p2)["crystal_X"]
    assert not np.allclose(saved, plain_crystal, atol=1e-5), (
        "the re-saved crystal is identical to a NON-guarded crystal: the guard"
        " was silently dropped"
    )


# --- 2) partial_fit after finalize_crystal() silently lost the rows -----
def test_save_after_post_finalize_data_fails_loudly(tmp_path):
    X, y = _synthetic(4000)
    d = SpectralHologramDistiller(crystal_size=40, sketch_dim=32, random_state=0)
    for i in range(0, 2000, 1000):
        d.partial_fit(X[i:i + 1000], y[i:i + 1000])
    d.finalize_crystal()
    d.partial_fit(X[2000:], y[2000:])          # silently accepted before
    with pytest.raises(ValueError, match="stream grew after finalize_crystal"):
        d.save_hologram(str(tmp_path / "stale.npz"))
    # re-finalizing rebuilds the crystal and the save succeeds
    d.finalize_crystal()
    d.save_hologram(str(tmp_path / "fresh.npz"))
    assert os.path.exists(str(tmp_path / "fresh.npz"))


# --- 3) the CLI crashed on an output path without a .npz suffix ----------
def test_cli_output_path_without_npz_is_normalised(tmp_path):
    from shd.cli import main

    X, y = _synthetic(1200, d=4)
    csv = str(tmp_path / "data.csv")
    np.savetxt(csv, np.c_[X, y], delimiter=",",
               header=",".join([f"f{i}" for i in range(5)]), comments="")
    out = str(tmp_path / "out1.hologram")          # no .npz suffix
    assert main(["distill", "-i", csv, "--target-col", "4", "-o", out,
                 "--crystal-size", "20"]) == 0
    assert os.path.exists(out + ".npz"), "numpy appends .npz; the CLI must report that path"
    assert not os.path.exists(out), "no stray file without the suffix"


# --- 4) the portfolio gate trained with two opposite column conventions --
def test_portfolio_gate_uses_one_column_convention():
    rng = np.random.default_rng(0)
    Xc = rng.uniform(0.45, 0.55, (80, 1))
    yc = np.full(80, 100.0)
    Xg = rng.uniform(0, 1, (80, 1))
    yg = 100.0 * (Xg[:, 0] > 0.5)
    sc = StandardScaler().fit(np.vstack([Xc, Xg]))
    port = CoresetPortfolio(critical_threshold=50.0).fit(
        Xc, yc, None, Xg, yg, None, sc, 1 / 1.1)
    Zc, Zg = sc.transform(Xc), sc.transform(Xg)
    # the two training blocks must use the SAME slot convention, and it must
    # be the one gate_weights() uses at inference: slot0 = critical expert
    block_crit = port._gate_design(port.oof_crit_, port.krr_glob_.predict(Zc), Zc)
    block_glob = port._gate_design(port.krr_crit_.predict(Zg), port.oof_glob_, Zg)
    assert np.allclose(block_crit[:, 0], port.oof_crit_)
    assert np.allclose(block_glob[:, 0], port.krr_crit_.predict(Zg)), (
        "the global block still puts its own expert in slot0"
    )
    # and the routing must not send most global-domain rows to the critical one
    w = port.gate_weights(np.vstack([Xc[:20], Xg[:20]]))
    misrouted = float(np.mean(w[20:] > 0.5))
    assert misrouted < 0.35, f"{misrouted:.0%} of global rows routed to the critical expert"


# --- 5) the conformal calibration used in-sample residuals ----------------
def test_conformal_warns_when_the_model_saw_the_calibration_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=200)
    sc = StandardScaler().fit(X)
    model = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1).fit(sc.transform(X), y)
    with pytest.warns(UserWarning, match="in-sample"):
        ConformalCrystal(k=7).fit(X, y, scaler=sc, model=model)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ConformalCrystal(k=7).fit(X, y, scaler=sc, model=model, model_independent=True)
    assert not [w for w in rec if "in-sample" in str(w.message)]


# --- 6) emergency_band=0.0 disabled the documented alarm-priority floor --
def test_emergency_band_zero_still_floors_the_alarm():
    rng = np.random.default_rng(0)
    Xc = rng.uniform(0.45, 0.55, (60, 1))
    yc = np.full(60, 100.0)
    Xg = rng.uniform(0, 1, (60, 1))
    yg = 100.0 * (Xg[:, 0] > 0.5)
    sc = StandardScaler().fit(np.vstack([Xc, Xg]))
    port = CoresetPortfolio(critical_threshold=50.0, emergency_band=0.0).fit(
        Xc, yc, None, Xg, yg, None, sc, 1 / 1.1)
    Zc = sc.transform(Xc)
    p_crit = port.krr_crit_.predict(Zc)
    w = port.gate_weights(Xc)
    above = p_crit >= 50.0
    assert above.any(), "precondition: some critical rows are above the threshold"
    assert np.allclose(w[above], 1.0), (
        "the alarm-priority floor must apply at/above the threshold even when"
        " emergency_band=0.0"
    )


# --- 7) the artifact did not record the operating point ------------------
def test_artifact_records_tail_quantile_and_phase_kappa(tmp_path):
    X, y = _synthetic(2000)
    d = SpectralHologramDistiller(crystal_size=30, sketch_dim=32, random_state=0,
                                 tail_quantile=91.5, phase_kappa=2.7)
    for i in range(0, len(X), 1000):
        d.partial_fit(X[i:i + 1000], y[i:i + 1000])
    d.finalize_crystal()
    path = str(tmp_path / "op.npz")
    d.save_hologram(path)
    raw = np.load(path)
    assert "tail_quantile" in raw.files and "phase_kappa" in raw.files
    assert float(raw["tail_quantile"][0]) == pytest.approx(91.5)
    assert float(raw["phase_kappa"][0]) == pytest.approx(2.7)
    s = HologramCrystalLoader(path).summary()
    assert s["tail_quantile"] == pytest.approx(91.5)
    assert s["phase_kappa"] == pytest.approx(2.7)
    # a legacy artifact without the fields must load and report them as unknown
    legacy = str(tmp_path / "legacy.npz")
    payload = {k: raw[k] for k in raw.files if k not in ("tail_quantile", "phase_kappa")}
    np.savez_compressed(legacy, **payload)
    s_legacy = HologramCrystalLoader(legacy).summary()
    assert s_legacy["tail_quantile"] is None
    assert s_legacy["phase_kappa"] is None

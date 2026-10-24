"""Tests for shd.portfolio: CoresetPortfolio (soft OOF stacker gate) and
ConformalCrystal (split / Mondrian conformal intervals, incl. alarm_zone).

Statistical checks use fixed seeds and conservative tolerances: they verify
the *guarantees* the classes document (marginal split-conformal coverage,
group-conditional alarm-zone coverage, alarm-priority floor), not
tuned-to-pass numbers.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import CoresetPortfolio, ConformalCrystal  # noqa: E402


def make_stream(n=6000, seed=11):
    """4D stream with a rare phase-peak regime and heterogeneous noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3] + 0.8 * rng.normal(size=int(hot.sum()))
    return X, y


def make_arms(X, y, seed=7, n_coreset=150, threshold=6.0):
    """Two coresets (critical SHD-like y-stratified, global uniform) + scaler.

    The critical arm oversamples high-y rows exactly like the distiller's
    critical crystal; the global arm is a uniform random sketch. Both are
    small, so the whole suite stays fast.
    """
    rng = np.random.default_rng(seed)
    hi = np.where(y >= threshold)[0]
    lo = np.where(y < threshold)[0]
    take_hi = rng.choice(hi, size=min(len(hi), n_coreset // 2), replace=False)
    take_lo = rng.choice(lo, size=n_coreset - len(take_hi), replace=False)
    idx_c = np.concatenate([take_hi, take_lo])
    Xc, yc = X[idx_c], y[idx_c]
    wc = np.where(yc >= threshold, 1.0, 0.5)

    idx_g = rng.choice(len(X), size=n_coreset, replace=False)
    Xg, yg = X[idx_g], y[idx_g]
    wg = np.ones(n_coreset)

    scaler = StandardScaler().fit(X)
    gamma = 1.0 / (X.shape[1] * 1.1)
    return Xc, yc, wc, Xg, yg, wg, scaler, gamma


# ---------------------------------------------------------------------------
# CoresetPortfolio
# ---------------------------------------------------------------------------


def test_portfolio_fit_predict_contract():
    X, y = make_stream()
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(X, y)
    pf = CoresetPortfolio(critical_threshold=6.0).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma, model_seed=3)
    rng = np.random.default_rng(5)
    Xq = rng.normal(size=(50, 4))
    p = pf.predict(Xq)
    assert p.shape == (50,)
    assert np.all(np.isfinite(p))
    w = pf.gate_weights(Xq)
    assert w.shape == (50,)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)
    p_c, p_g = pf.expert_predictions(Xq)
    assert p_c.shape == p_g.shape == (50,)
    # The blend must lie between the two experts' predictions per row.
    lo = np.minimum(p_c, p_g) - 1e-9
    hi = np.maximum(p_c, p_g) + 1e-9
    assert np.all(p >= lo) and np.all(p <= hi)


def test_portfolio_streaming_flag_is_stateless():
    X, y = make_stream()
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(X, y)
    pf = CoresetPortfolio(critical_threshold=6.0).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma)
    X1 = X[:64]
    X2 = X[64:128]
    both = pf.predict(np.vstack([X1, X2]), streaming=True)
    sep = np.concatenate([pf.predict(X1, streaming=True),
                          pf.predict(X2, streaming=True)])
    assert np.allclose(both, sep)


def test_portfolio_emergency_floor_forces_critical_expert():
    """A huge emergency band floors every row's weight at 1.0 (alarm priority
    saturates); band=0 keeps the pure logistic gate."""
    X, y = make_stream()
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(X, y)
    Xq = X[:100]
    pf_off = CoresetPortfolio(critical_threshold=6.0, emergency_band=0.0).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma)
    pf_on = CoresetPortfolio(critical_threshold=6.0, emergency_band=1e9).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma)
    w_off = pf_off.gate_weights(Xq)
    w_on = pf_on.gate_weights(Xq)
    assert np.all(w_on == 1.0)
    assert not np.all(w_off == 1.0)


def test_portfolio_is_deterministic():
    X, y = make_stream()
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(X, y)
    preds = []
    for _ in range(2):
        pf = CoresetPortfolio(critical_threshold=6.0).fit(
            Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma, model_seed=3)
        preds.append(pf.predict(X[:80]))
    assert np.array_equal(preds[0], preds[1])


def test_portfolio_validation_errors():
    X, y = make_stream()
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(X, y)
    with pytest.raises(ValueError):
        CoresetPortfolio(critical_threshold=6.0, n_folds=2)
    with pytest.raises(ValueError):
        CoresetPortfolio(critical_threshold=float("nan"))
    pf = CoresetPortfolio(critical_threshold=6.0).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma)
    with pytest.raises(RuntimeError):
        CoresetPortfolio(critical_threshold=6.0).gate_weights(Xc)
    with pytest.raises(ValueError):
        pf.predict(Xc[:, :3])  # wrong feature width
    with pytest.raises(ValueError):
        CoresetPortfolio(critical_threshold=6.0).fit(
            Xc, yc[:-1], wc, Xg, yg, wg, scaler, rbf_gamma=gamma)  # y/w mismatch
    with pytest.raises(ValueError):
        CoresetPortfolio(critical_threshold=6.0).fit(
            Xc, yc, wc, Xg[:, :3], yg, wg, scaler, rbf_gamma=gamma)  # width clash


# ---------------------------------------------------------------------------
# ConformalCrystal
# ---------------------------------------------------------------------------


def _fit_krr(Xc, yc, wc, scaler, gamma):
    return KernelRidge(kernel="rbf", alpha=0.08, gamma=gamma).fit(
        scaler.transform(Xc), yc, sample_weight=wc)


def test_conformal_marginal_coverage_synthetic():
    """Split-conformal guarantee on exchangeable calibration rows."""
    X, y = make_stream(n=8000, seed=23)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=9, n_coreset=200, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)  # global model = the deployer
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), size=800, replace=False)
    cc = ConformalCrystal(k=15).fit(X[idx], y[idx], scaler=scaler, model=krr)
    iv = cc.predict_interval(X[:3000], krr, alpha=0.10)
    cov = float(np.mean((y[:3000] >= iv["low"]) & (y[:3000] <= iv["high"])))
    assert 0.85 <= cov <= 0.95, f"marginal coverage {cov:.3f} outside tolerance"
    assert iv["info"]["n_calibration"] == 800
    assert iv["info"]["rank"] == 721  # ceil(0.9 * 801) = 721


def test_conformal_alarm_zone_group_coverage():
    """Alarm-zone Mondrian: inside the zone BOTH bounds carry >= 1-alpha
    coverage even when the regressor systematically under-predicts peaks
    (the upper signed-residual quantile absorbs the bias)."""
    X, y = make_stream(n=9000, seed=31)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=4, n_coreset=200, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)
    rng = np.random.default_rng(2)
    idx = rng.choice(len(X), size=1200, replace=False)
    cc = ConformalCrystal(k=15, mondrian="alarm_zone", critical_threshold=6.0,
                          zone_fraction=0.10).fit(
        X[idx], y[idx], scaler=scaler, model=krr)
    iv = cc.predict_interval(X[:3000], krr, alpha=0.10)
    assert iv["info"]["mondrian"] == "alarm_zone"
    assert not iv["info"]["group_fallback"]
    in_zone = iv["info"]["row_group"]
    assert int(np.sum(in_zone)) >= 100  # the zone is not degenerate
    cov_zone = float(np.mean(
        (y[:3000][in_zone] >= iv["low"][in_zone])
        & (y[:3000][in_zone] <= iv["high"][in_zone])))
    assert cov_zone >= 0.87, f"alarm-zone coverage {cov_zone:.3f} < 1-alpha-tol"
    # The alarm-side (upper) guarantee is the deployment-critical one.
    up_zone = float(np.mean(y[:3000][in_zone] <= iv["high"][in_zone]))
    assert up_zone >= 0.90, f"alarm-zone upper coverage {up_zone:.3f}"


def test_conformal_alarm_zone_contains_true_peaks():
    """The zone must actually capture true-critical rows (a zone that never
    contains a peak is useless regardless of its coverage guarantee)."""
    X, y = make_stream(n=8000, seed=17)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=6, n_coreset=200, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)
    rng = np.random.default_rng(3)
    idx = rng.choice(len(X), size=1200, replace=False)
    cc = ConformalCrystal(k=15, mondrian="alarm_zone", critical_threshold=6.0,
                          zone_fraction=0.10).fit(
        X[idx], y[idx], scaler=scaler, model=krr)
    iv = cc.predict_interval(X[:3000], krr, alpha=0.10)
    in_zone = iv["info"]["row_group"]
    true_crit = y[:3000] >= 6.0
    n_crit = int(np.sum(true_crit))
    captured = int(np.sum(true_crit & in_zone))
    assert n_crit > 50
    assert captured >= 0.5 * n_crit, (
        f"zone captured {captured}/{n_crit} true peaks"
    )


def test_conformal_min_group_fallback_is_flagged():
    """A too-small calibration set must fall back to the pooled quantile and
    SAY SO in info, not silently pretend the group quantile is calibrated."""
    X, y = make_stream(n=2000, seed=5)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=8, n_coreset=60, threshold=6.0)
    krr = _fit_krr(Xc, yc, wc, scaler, gamma)
    rng = np.random.default_rng(4)
    idx = rng.choice(len(X), size=30, replace=False)
    cc = ConformalCrystal(k=5, mondrian="threshold", critical_threshold=6.0,
                          min_group=10).fit(
        X[idx], y[idx], scaler=scaler, model=krr)
    iv = cc.predict_interval(X[:200], krr, alpha=0.10)
    assert iv["info"]["group_fallback"] is True


def test_conformal_adaptive_widens_far_from_crystal():
    X, y = make_stream(n=5000, seed=19)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=12, n_coreset=150, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)
    rng = np.random.default_rng(6)
    idx = rng.choice(len(X), size=500, replace=False)
    cc = ConformalCrystal(k=10).fit(X[idx], y[idx], scaler=scaler, model=krr)
    plain = cc.predict_interval(X[:500], krr, alpha=0.10)
    cc_a = ConformalCrystal(k=10, adaptive=True).fit(
        X[idx], y[idx], scaler=scaler, model=krr)
    adapt = cc_a.predict_interval(X[:500], krr, alpha=0.10)
    # adaptive rescales by kNN distance / no-op when distances equal the median
    assert np.all(adapt["half_width"] > 0)
    assert not np.allclose(plain["half_width"], adapt["half_width"])
    s = cc.calibration_summary
    for key in ("n_calibration", "residual_median", "residual_q90",
                "residual_max", "knn_median", "adaptive", "k"):
        assert key in s


def test_conformal_is_deterministic_and_contract_correct():
    X, y = make_stream(n=3000, seed=29)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=13, n_coreset=120, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(X), size=400, replace=False)
    Xq = X[:200]
    out = []
    for _ in range(2):
        cc = ConformalCrystal(k=12).fit(
            X[idx], y[idx], scaler=scaler, model=krr)
        out.append(cc.predict_interval(Xq, krr, alpha=0.10))
    assert np.array_equal(out[0]["low"], out[1]["low"])
    assert np.array_equal(out[0]["high"], out[1]["high"])
    iv = out[0]
    assert np.all(iv["low"] <= iv["pred"]) and np.all(iv["pred"] <= iv["high"])
    assert np.allclose(iv["half_width"], 0.5 * (iv["high"] - iv["low"]))


def test_conformal_validation_errors():
    X, y = make_stream(n=1500, seed=37)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=14, n_coreset=80, threshold=6.0)
    krr = _fit_krr(Xg, yg, wg, scaler, gamma)
    with pytest.raises(ValueError):
        ConformalCrystal(k=1)
    with pytest.raises(ValueError):
        ConformalCrystal(mondrian="nope")
    with pytest.raises(ValueError):
        ConformalCrystal(mondrian="threshold")  # threshold missing
    with pytest.raises(ValueError):
        ConformalCrystal(mondrian="threshold", critical_threshold=6.0,
                         min_group=1)
    with pytest.raises(ValueError):
        ConformalCrystal(mondrian="alarm_zone", critical_threshold=6.0,
                         zone_fraction=0.9)
    with pytest.raises(RuntimeError):
        ConformalCrystal().predict_interval(Xg, krr)
    cc = ConformalCrystal().fit(Xg, yg, scaler=scaler, model=krr)
    with pytest.raises(ValueError):
        cc.predict_interval(Xg, krr, alpha=0.0)
    with pytest.raises(ValueError):
        cc.predict_interval(Xg, krr, alpha=1.0)
    with pytest.raises(ValueError):
        cc.predict_interval(Xg[:, :3], krr)
    with pytest.raises(ValueError):
        ConformalCrystal().fit(Xg, yg, scaler=scaler, model=None)


# ---------------------------------------------------------------------------
# Integration: portfolio + conformal on one stream (the deployment story)
# ---------------------------------------------------------------------------


def test_integration_portfolio_then_conformal_intervals():
    X, y = make_stream(n=7000, seed=41)
    Xc, yc, wc, Xg, yg, wg, scaler, gamma = make_arms(
        X, y, seed=15, n_coreset=180, threshold=6.0)
    pf = CoresetPortfolio(critical_threshold=6.0).fit(
        Xc, yc, wc, Xg, yg, wg, scaler, rbf_gamma=gamma, model_seed=3)

    class _PortfolioModel:
        """ConformalCrystal scores any fitted .predict model - the portfolio
        itself is one."""

        def predict(self, Z):
            return pf.predict(scaler.inverse_transform(Z), streaming=True)

    rng = np.random.default_rng(8)
    idx = rng.choice(len(X), size=600, replace=False)
    cc = ConformalCrystal(k=12, mondrian="alarm_zone", critical_threshold=6.0,
                          zone_fraction=0.10).fit(
        X[idx], y[idx], scaler=scaler, model=_PortfolioModel())
    Xq = X[:1000]
    yq = y[:1000]
    iv = cc.predict_interval(Xq, _PortfolioModel(), alpha=0.10)
    pred = pf.predict(Xq, streaming=True)
    assert np.allclose(iv["pred"], pred)
    cov = float(np.mean((yq >= iv["low"]) & (yq <= iv["high"])))
    assert cov >= 0.85, f"portfolio-conformal coverage {cov:.3f}"

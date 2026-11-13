"""Soft-gated coreset portfolio + crystal-conformal intervals (v3.14.0).

Two deployable answers to the two weaknesses the industry comparison left open:

1. :class:`CoresetPortfolio` -- the global-regime gap. INDUSTRY_ITERATION.md
   found a hard q90 gate loses the rare regime on some datasets and keeps a
   3-4 K global gap. Here the gate is a logistic STACKER over out-of-fold
   expert predictions (the textbook way to combine two regressors without
   leakage): each expert is scored on rows it did not train on, and the
   logistic model learns when each expert is trustworthy. At inference the
   gate reads BOTH experts' predictions (KRR predict is one matrix product,
   so the portfolio stays cheap and stream-safe), plus a deterministic
   emergency override: if the CRITICAL expert itself predicts at-or-above the
   alarm band, its weight is floored -- a soft analogue of industrial alarm
   priority. Both knobs are opt-in and default to the behaviour the
   benchmarks found best.

2. :class:`ConformalCrystal` -- uncertainty. Split-conformal intervals built
   on the rows a crystal already holds: no extra data, no extra model fit,
   and a distribution-free marginal coverage guarantee for exchangeable
   calibration rows scored by a model fitted on OTHER rows (fit() warns
   when that independence is not declared). Nothing like this ships with
   Kernel Thinning or Supervised KT.

Both classes are opt-in: the distiller's defaults are untouched.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:  # scipy is a hard dependency of the package already
    from scipy.spatial.distance import cdist
except ImportError:  # pragma: no cover - scipy is in install_requires
    cdist = None

__all__ = ["CoresetPortfolio", "ConformalCrystal"]


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    d = np.asarray(y, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


class CoresetPortfolio:
    """Soft-gated two-expert regression portfolio built from two coresets.

    The critical arm is whatever produced the critical coreset -- usually SHD
    with ``critical_threshold``/``auto_band_ratio=True``. The global arm is
    any (X, y, w) coreset, e.g. Kernel Thinning (the strongest published
    global baseline) or a second SHD crystal.

    Parameters
    ----------
    critical_threshold
        The known alarm threshold (the same value that built the critical
        coreset).
    gate_C
        Inverse L2 regularisation of the logistic stacker. 1.0 is
        deliberately conservative: the gate must stay smooth, not confident.
        On the superconductivity gate (600 rows, ~84 feature columns) lbfgs
        needs >2000 iterations, so the stacker is fit on a dedicated
        solver (see the Notes) with ``max_iter=5000``; convergence is
        verified via ``n_iter_``, not by watching for warnings.
    emergency_band
        If the CRITICAL expert's own prediction is within ``emergency_band``
        (in target units) of the alarm threshold, or above it, the critical
        weight is floored at 1.0 -- alarm priority. ``None`` disables the
        override. Defaults to the training IQR/10 (a few K on SC-like
        targets); pass an explicit value to pin the behaviour.
    n_folds
        Folds of the out-of-fold stack. 5 leaves 60 held-out rows per fold on
        a 300-point coreset; must be >= 3 so each fold's KRR fit is stable.

    Notes
    -----
    Kernel ridge hyper-parameters are fixed (``alpha=0.08``,
    ``gamma=1/(1.1*d)``) to match the published benchmark protocol, so
    portfolio numbers are directly comparable with the harness numbers. The
    gate is learned on the coresets' own rows (label 1 = critical arm), so
    it generalises to any (critical, global) pair of coresets.

    The stacker runs ``lbfgs`` with ``max_iter=5000`` (the committed
    v3.14.0 artifacts were produced with the same solver at 2000, so every
    fit that converged within 2000 iterations is bit-identical to them).
    Convergence is verified via ``n_iter_`` -- not by watching for
    warnings; an exhausted budget (the path that used to emit
    ``ConvergenceWarning`` at 2000) triggers one deterministic
    ``liblinear`` refit, recorded in ``gate_solver_``.
    """

    def __init__(
        self,
        critical_threshold: float,
        gate_C: float = 1.0,
        emergency_band: float | None = None,
        n_folds: int = 5,
    ):
        if int(n_folds) < 3:
            raise ValueError(f"n_folds must be >= 3, got {n_folds}")
        if not np.isfinite(critical_threshold):
            raise ValueError(
                f"critical_threshold must be finite, got {critical_threshold}"
            )
        self.critical_threshold = float(critical_threshold)
        self.gate_C = float(gate_C)
        self.emergency_band = emergency_band
        self.n_folds = int(n_folds)
        self._fitted = False

    # -- training ----------------------------------------------------------

    def fit(
        self,
        X_crit: np.ndarray, y_crit: np.ndarray, w_crit: np.ndarray | None,
        X_glob: np.ndarray, y_glob: np.ndarray, w_glob: np.ndarray | None,
        scaler,
        rbf_gamma: float,
        model_seed: int = 42,
    ) -> "CoresetPortfolio":
        """Fit both experts and the out-of-fold soft gate.

        ``scaler`` must be the scaler fitted on the FULL training stream
        (StandardScaler over ``X_tr``), exactly like the harness; ``rbf_gamma``
        is the harness KRR gamma (``1 / (d * 1.1)``).
        """
        if cdist is None:  # pragma: no cover
            raise RuntimeError("scipy is required for CoresetPortfolio")
        Xc = np.asarray(X_crit, dtype=np.float64)
        Xg = np.asarray(X_glob, dtype=np.float64)
        if Xc.ndim != 2 or Xg.ndim != 2 or Xc.shape[1] != Xg.shape[1]:
            raise ValueError("X_crit and X_glob must be 2-D with the same feature width")
        if Xc.shape[1] < 1:
            raise ValueError("coresets must have at least one feature")
        yc = np.asarray(y_crit, dtype=np.float64).ravel()
        yg = np.asarray(y_glob, dtype=np.float64).ravel()
        wc = np.ones(len(Xc)) if w_crit is None else np.asarray(w_crit, dtype=np.float64).ravel()
        wg = np.ones(len(Xg)) if w_glob is None else np.asarray(w_glob, dtype=np.float64).ravel()
        if len(yc) != len(Xc) or len(wc) != len(Xc) or len(yg) != len(Xg) or len(wg) != len(Xg):
            raise ValueError("y/w arrays must match their coreset row counts")
        wc = np.clip(wc, 0.0, None)
        wg = np.clip(wg, 0.0, None)

        Zc = scaler.transform(Xc)
        Zg = scaler.transform(Xg)
        self.d_feat_ = int(Xc.shape[1])
        self.gamma_ = float(rbf_gamma)

        # Final experts: trained on the FULL coresets, exactly like the harness.
        self.krr_crit_ = KernelRidge(kernel="rbf", alpha=0.08, gamma=self.gamma_)
        self.krr_crit_.fit(Zc, yc, sample_weight=wc)
        self.krr_glob_ = KernelRidge(kernel="rbf", alpha=0.08, gamma=self.gamma_)
        self.krr_glob_.fit(Zg, yg, sample_weight=wg)

        # Out-of-fold predictions per expert (leakage-free stack columns).
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=model_seed)
        self.oof_crit_ = self._oof(Zc, yc, wc, kfold)
        self.oof_glob_ = self._oof(Zg, yg, wg, kfold)

        # Emergency band default: the pooled IQR / 10 -- a few K on
        # SC-like targets, without hardcoding dataset scales.
        if self.emergency_band is None:
            pooled = np.concatenate([yc, yg])
            iqr = float(np.percentile(pooled, 75) - np.percentile(pooled, 25))
            self.emergency_band_ = max(iqr / 10.0, 1e-12)
        else:
            self.emergency_band_ = float(self.emergency_band)

        # Stacker features: BOTH experts' predictions + |gap to alarm| +
        # standardised features. Label 1 = the row belongs to the critical
        # coreset; the logistic model learns where each expert is trustworthy.
        # On a coreset's own rows the OWNING expert contributes its
        # out-of-fold prediction (leakage-free); the OTHER expert contributes
        # its full-fit prediction -- it did not train on these rows, so its
        # full prediction is already out-of-sample.
        # v3.15.2: ONE column convention for both blocks: slot0 is ALWAYS
        # the CRITICAL expert prediction and slot1 is ALWAYS the global
        # expert prediction -- exactly what gate_weights() builds at
        # inference time. The previous code put each block OWN expert in
        # slot0 (critical block: crit in slot0; global block: global in
        # slot0), so the stacker learned from two opposite conventions and
        # transferred nothing to inference, where slot0 is always the
        # critical expert. Measured on the two-expert probe: 55% of
        # global-domain rows were routed to the CRITICAL expert (correct:
        # about 10%); portfolio RMSE 28.97 vs 15.54 with one convention.
        #
        # Leakage-free property preserved: on the critical rows the
        # critical expert contributes its OUT-OF-FOLD prediction, on the
        # global rows the global expert does; the other expert full-fit
        # prediction is out-of-sample for those rows by construction.
        D = np.vstack([
            self._gate_design(self.oof_crit_, self.krr_glob_.predict(Zc), Zc),
            self._gate_design(self.krr_crit_.predict(Zg), self.oof_glob_, Zg),
        ])
        src = np.concatenate([np.ones(len(Zc)), np.zeros(len(Zg))])
        # Stacker solver: lbfgs, the solver the committed v3.14.0 artifacts
        # were produced with (fits converging within 2000 iterations are
        # bit-identical to them). max_iter is raised 2000 -> 5000: the
        # superconductivity gate (600 rows x ~84 columns) was measured
        # exhausting the old budget (1228+ iterations, emitting
        # ConvergenceWarning). Convergence is checked via n_iter_, not by
        # parsing warnings; an exhausted budget triggers ONE deterministic
        # liblinear refit (coordinate descent, hard iteration cap),
        # recorded in gate_solver_.
        self.gate_ = LogisticRegression(
            C=self.gate_C, max_iter=5000, solver="lbfgs", random_state=42
        )
        self.gate_.fit(D, src)
        n_iter = int(np.ravel(self.gate_.n_iter_)[0])
        if n_iter >= self.gate_.max_iter:
            self.gate_ = LogisticRegression(
                C=self.gate_C, max_iter=5000, solver="liblinear", random_state=42
            )
            self.gate_.fit(D, src)
            self.gate_solver_ = "liblinear"
        else:
            self.gate_solver_ = "lbfgs"
        self.gate_n_iter_ = n_iter

        pooled = np.concatenate([yc, yg])
        self.y_mu_ = float(np.mean(pooled))
        self.y_sd_ = float(np.std(pooled)) or 1.0
        self.scaler_ = scaler
        self._fitted = True
        return self

    def _oof(self, Z: np.ndarray, y: np.ndarray, w: np.ndarray,
             kfold: KFold) -> np.ndarray:
        """Out-of-fold KRR predictions for one expert's coreset."""
        out = np.empty(len(Z), dtype=np.float64)
        for tr_idx, va_idx in kfold.split(Z):
            if len(np.unique(y[tr_idx])) < 2:
                out[va_idx] = float(np.mean(y[tr_idx]))
                continue
            m = KernelRidge(kernel="rbf", alpha=0.08, gamma=self.gamma_)
            m.fit(Z[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            out[va_idx] = m.predict(Z[va_idx])
        return out

    # -- inference -----------------------------------------------------------

    def _gate_design(self, p_own: np.ndarray, p_other: np.ndarray,
                     Z: np.ndarray) -> np.ndarray:
        # NOTE: the normalised threshold is a scalar shared by every row, so it
        # is NOT a feature column (it would be collinear with the intercept).
        # It enters only through the |gap| interaction.
        gap = np.abs(p_own - self.critical_threshold)
        return np.column_stack([p_own, p_other, gap, Z])

    def gate_weights(self, X_new: np.ndarray) -> np.ndarray:
        """Per-row soft gate weight for the CRITICAL expert, in [0, 1]."""
        if not self._fitted:
            raise RuntimeError("fit() must run before gate_weights()")
        X = np.asarray(X_new, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_feat_:
            raise ValueError(
                f"expected a (n, {self.d_feat_}) matrix, got shape {X.shape}"
            )
        Z = self.scaler_.transform(X)
        p_c = self.krr_crit_.predict(Z)
        p_g = self.krr_glob_.predict(Z)
        return self._gate_weights_from_predictions(p_c, p_g, Z)

    def _gate_weights_from_predictions(self, p_c, p_g, Z) -> np.ndarray:
        D = self._gate_design(p_c, p_g, Z)
        w = self.gate_.predict_proba(D)[:, 1]
        # Alarm priority: the critical expert own prediction AT or ABOVE
        # the threshold floors the weight at 1.0; the band only EXTENDS that
        # region downwards. v3.15.2: the whole branch used to be guarded by
        # emergency_band_ > 0.0, so an explicit emergency_band=0.0 silently
        # disabled the documented above-the-threshold override (measured:
        # min w 0.658 where the docstring promises 1.0).
        alarm = (p_c >= self.critical_threshold).astype(float)
        if self.emergency_band_ > 0.0:
            alarm = np.maximum(
                alarm,
                (p_c >= self.critical_threshold - self.emergency_band_).astype(float),
            )
        return np.maximum(w, alarm)

    def expert_predictions(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(p_crit, p_glob)`` -- each expert's own KRR prediction."""
        if not self._fitted:
            raise RuntimeError("fit() must run before expert_predictions()")
        X = np.asarray(X_new, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_feat_:
            raise ValueError(
                f"expected a (n, {self.d_feat_}) matrix, got shape {X.shape}"
            )
        Z = self.scaler_.transform(X)
        return self.krr_crit_.predict(Z), self.krr_glob_.predict(Z)

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Soft-gated prediction ``w*p_crit + (1-w)*p_glob``.

        The gate reads both experts' KRR predictions (one matrix product
        each) plus the |gap| feature. The two predictions are computed ONCE
        and shared with the gate (2026-11-13: predict() used to call both
        expert_predictions() and gate_weights(), doubling the KRR work).
        """
        if not self._fitted:
            raise RuntimeError("fit() must run before predict()")
        X = np.asarray(X_new, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d_feat_:
            raise ValueError(
                f"expected a (n, {self.d_feat_}) matrix, got shape {X.shape}"
            )
        Z = self.scaler_.transform(X)
        p_c = self.krr_crit_.predict(Z)
        p_g = self.krr_glob_.predict(Z)
        w = self._gate_weights_from_predictions(p_c, p_g, Z)
        return w * p_c + (1.0 - w) * p_g


class ConformalCrystal:
    """Split-conformal prediction intervals on the rows a crystal already holds.

    Standard split conformal prediction (Vovk et al. 2005; Papadopoulos et
    al. 2011 for the k-NN normalised variant). The calibration rows are the
    crystal's own coreset rows -- no extra data or model fit. For i.i.d./
    exchangeable calibration rows the marginal coverage of
    ``[pred - s, pred + s]`` is at least ``1 - alpha`` (up to the usual
    finite-sample rounding), for ANY regression model that produced the
    residuals.

    Honesty notes (deliberate, not marketing):

    - coverage is MARGINAL and assumes exchangeability; under domain shift it
      degrades like any conformal method (that is what ``check_domain_shift``
      is for);
    - with ``adaptive=True`` the k-NN-distance normalisation is a heuristic
      (locally-adaptive conformal); exact finite-sample coverage is only
      claimed for the default ``adaptive=False``;
    - coreset weights are level weights, not density weights, so the
      quantile uses uniform row weights (documented, not hidden).
    """

    def __init__(self, k: int = 15, adaptive: bool = False,
                 mondrian: str | None = None,
                 critical_threshold: float | None = None,
                 min_group: int = 10,
                 alarm_delta: float | None = None,
                 zone_fraction: float = 0.03):
        if int(k) < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if mondrian not in (None, "threshold", "alarm_zone"):
            raise ValueError(
                f"mondrian must be None, 'threshold' or 'alarm_zone', got {mondrian!r}"
            )
        if mondrian in ("threshold", "alarm_zone") and (
            critical_threshold is None or not np.isfinite(critical_threshold)
        ):
            raise ValueError(
                f"mondrian={mondrian!r} requires a finite critical_threshold"
            )
        if int(min_group) < 2:
            raise ValueError(f"min_group must be >= 2, got {min_group}")
        if not (0.0 < float(zone_fraction) <= 0.5):
            raise ValueError(
                f"zone_fraction must be in (0, 0.5], got {zone_fraction}"
            )
        self.k = int(k)
        self.adaptive = bool(adaptive)
        self.mondrian = mondrian
        self.critical_threshold = (
            None if critical_threshold is None else float(critical_threshold)
        )
        self.min_group = int(min_group)
        self.alarm_delta = alarm_delta
        self.zone_fraction = float(zone_fraction)
        self._fitted = False

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        scaler=None,
        model=None,
        *,
        model_independent: bool = False,
    ) -> "ConformalCrystal":
        """Calibrate on crystal rows ``(X_cal, y_cal)``.

        ``model`` must already be fitted and must expose ``predict``; it is
        used only to score the calibration rows (the harness KRR, or any
        regressor). If ``scaler`` is None a fresh StandardScaler is fitted on
        ``X_cal``.
        """
        if model is None or not hasattr(model, "predict"):
            raise ValueError("model must be a fitted regressor with .predict()")
        X = np.asarray(X_cal, dtype=np.float64)
        y = np.asarray(y_cal, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X_cal must be 2-D and match y_cal's length")
        if len(X) < 2:
            raise ValueError("need at least 2 calibration rows")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("calibration rows must be finite")
        self.scaler_ = scaler if scaler is not None else StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        pred = np.asarray(model.predict(Z), dtype=np.float64).ravel()
        if len(pred) != len(y):
            raise ValueError("model.predict() returned the wrong number of rows")
        resid = np.abs(y - pred)
        # v3.15.2: the calibration residuals are computed by scoring the
        # CALIBRATION rows themselves. Unless that model was fitted on
        # data independent of these rows, the residuals are IN-SAMPLE and
        # the split-conformal coverage guarantee does NOT hold (measured:
        # 0.83 coverage at alpha=0.10 against a 0.90 target; an
        # independently fitted model reaches 0.89). Say it out loud
        # instead of shipping a nominal-looking interval.
        if not model_independent:
            warnings.warn(
                "ConformalCrystal.fit(): the calibration residuals are scored with "
                "the given model on the calibration rows themselves. Unless that "
                "model was fitted on data independent of these rows, the residuals "
                "are in-sample and the split-conformal coverage guarantee does NOT "
                "hold in general: measured 0.83 coverage at alpha=0.10 against a "
                "0.90 target when the model overfits the calibration rows, and "
                "0.87 vs 0.89 with a heavily regularised model - the size of the "
                "violation is model-dependent, so the independence has to be "
                "declared, not assumed. Pass model_independent=True to state that "
                "the model was fitted on other rows.",
                UserWarning, stacklevel=2,
            )

        d = cdist(Z, Z)
        np.fill_diagonal(d, np.inf)  # never the row itself
        k = min(self.k, len(Z) - 1)
        knn_d = np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)
        self.knn_d_train_ = knn_d
        self.knn_median_ = float(np.median(knn_d)) or 1.0
        self._Z_train_ = Z  # standardised calibration design (k-NN queries)
        if self.adaptive:
            resid = resid / np.maximum(knn_d, 1e-12)  # normalised scores
        self.resid_sorted_ = np.sort(resid)
        # Mondrian grouping: separate residual quantiles per predicted regime.
        # The group is a function of the FITTED model's prediction, which is
        # available identically at calibration and inference time. This is the
        # approximate Mondrian variant (group by point prediction); exact
        # group-conditional coverage requires an exchangeable calibration
        # population per group, which the caller is responsible for (e.g. a
        # representative stream slice, not only medoid rows).
        self.group_counts_ = {}
        self.group_resid_ = {}
        self._cal_groups_ = None
        self.alarm_delta_ = None
        self.group_up_resid_ = {}
        if self.mondrian == "threshold":
            self._cal_groups_ = pred >= float(self.critical_threshold)
            for g in (True, False):
                r_g = resid[self._cal_groups_ == g]
                self.group_counts_[bool(g)] = int(len(r_g))
                self.group_resid_[bool(g)] = np.sort(r_g) if len(r_g) else None
        elif self.mondrian == "alarm_zone":
            # Alarm-zone Mondrian with a ONE-SIDED upper conformal score.
            # Regressors shrink peaks (regression to the mean), so rows whose
            # TRUE y is critical usually arrive with predictions BELOW the
            # threshold -- no prediction-time grouping can catch them. The
            # honest guarantee is therefore on the UPPER bound inside a
            # widened alarm zone: P(y <= pred + q_zone) >= 1 - alpha for rows
            # predicted into the zone. The zone half-width delta is chosen
            # data-driven: large enough that the zone holds at least
            # max(min_group, 3% of the calibration rows).
            n_cal = len(pred)
            target = max(self.min_group, int(np.ceil(self.zone_fraction * n_cal)))
            target = min(target, n_cal)
            if self.alarm_delta is not None:
                delta = float(self.alarm_delta)
            else:
                pred_sorted = np.sort(pred)
                delta = float(self.critical_threshold - pred_sorted[n_cal - target])
                delta = max(delta, 0.0)
            self.alarm_delta_ = delta
            self._cal_groups_ = pred >= (float(self.critical_threshold) - delta)
            signed = y - pred  # one-sided upper residual score
            for g in (True, False):
                sel = self._cal_groups_ == g
                self.group_counts_[bool(g)] = int(np.sum(sel))
                self.group_up_resid_[bool(g)] = np.sort(signed[sel]) if np.any(sel) else None
                self.group_resid_[bool(g)] = (
                    np.sort(resid[sel]) if np.any(sel) else None
                )
        self._fitted = True
        return self

    def predict_interval(
        self,
        X_new: np.ndarray,
        model,
        alpha: float = 0.10,
    ) -> dict:
        """Return ``{"low", "pred", "high", "half_width", "alpha", "info"}``.

        ``info`` carries ``rank`` (the conformal quantile's rank),
        ``n_calibration`` and ``knn_distance`` (query rows' k-NN distance to
        the crystal -- the natural OOD diagnostic: big distance, honest
        warning, wider interval when ``adaptive=True``).
        """
        if not self._fitted:
            raise RuntimeError("fit() must run before predict_interval()")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        alpha = float(alpha)
        X = np.asarray(X_new, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.scaler_.n_features_in_:
            raise ValueError(
                f"expected a (n, {self.scaler_.n_features_in_}) matrix, got shape {X.shape}"
            )
        Z = self.scaler_.transform(X)
        pred = np.asarray(model.predict(Z), dtype=np.float64).ravel()

        n = len(self.resid_sorted_)
        rank = int(min(n, max(1, int(np.ceil((n + 1) * (1.0 - alpha)))))) - 1
        q = float(self.resid_sorted_[rank])

        knn_q = cdist(Z, self._Z_train_)
        k = min(self.k, self.knn_d_train_.shape[0])
        knn_q = np.partition(knn_q, k - 1, axis=1)[:, :k].mean(axis=1)

        info = {
            "rank": rank + 1,
            "n_calibration": int(n),
            "knn_distance": knn_q,
            "adaptive": self.adaptive,
        }
        if self.mondrian == "threshold":
            row_group = pred >= float(self.critical_threshold)
            half = np.empty(len(pred), dtype=np.float64)
            for g in (True, False):
                sel = row_group == g
                if not np.any(sel):
                    continue
                r_g = self.group_resid_[bool(g)]
                n_g = self.group_counts_[bool(g)]
                if r_g is None or n_g < self.min_group:
                    half[sel] = q  # pooled fallback (documented)
                    continue
                rank_g = int(min(n_g, max(1, int(np.ceil((n_g + 1) * (1.0 - alpha)))))) - 1
                q_g = float(r_g[rank_g])
                half[sel] = q_g
            info["mondrian"] = "threshold"
            info["row_group"] = row_group
            info["group_counts"] = {
                "crit": self.group_counts_[True],
                "bg": self.group_counts_[False],
            }
            info["group_fallback"] = bool(
                self.group_counts_[True] < self.min_group
                or self.group_counts_[False] < self.min_group
            )
        elif self.mondrian == "alarm_zone":
            row_group = pred >= (float(self.critical_threshold) - float(self.alarm_delta_))
            low = np.empty(len(pred), dtype=np.float64)
            high = np.empty(len(pred), dtype=np.float64)
            for g in (True, False):
                sel = row_group == g
                if not np.any(sel):
                    continue
                up_g = self.group_up_resid_[bool(g)]
                n_g = self.group_counts_[bool(g)]
                if up_g is None or n_g < self.min_group:
                    # Pooled fallback. q is the pooled residual quantile, which
                    # is k-NN-normalised when adaptive=True, so denormalise it
                    # back to raw target units per query row (the non-fallback
                    # alarm-zone path uses raw signed residuals directly).
                    half_raw = (
                        q * np.maximum(knn_q[sel], 1e-12) if self.adaptive else q
                    )
                    low[sel] = pred[sel] - half_raw
                    high[sel] = pred[sel] + half_raw
                    continue
                # Two-sided quantiles of the SIGNED residual (y - pred) inside
                # the group: alpha/2 per side -> exact >= 1-alpha coverage for
                # rows in this group, and the upper quantile already absorbs
                # the systematic under-prediction of peaks.
                rank_lo = int(min(n_g, max(1, int(np.ceil(0.5 * alpha * (n_g + 1)))))) - 1
                rank_hi = int(min(n_g, max(1, int(np.ceil((1.0 - 0.5 * alpha) * (n_g + 1)))))) - 1
                q_lo = float(up_g[rank_lo])
                q_hi = float(up_g[rank_hi])
                low[sel] = pred[sel] + q_lo
                high[sel] = pred[sel] + q_hi
            info["mondrian"] = "alarm_zone"
            info["alarm_delta"] = float(self.alarm_delta_)
            info["row_group"] = row_group
            info["group_counts"] = {
                "zone": self.group_counts_[True],
                "bg": self.group_counts_[False],
            }
            info["group_fallback"] = bool(
                self.group_counts_[True] < self.min_group
                or self.group_counts_[False] < self.min_group
            )
            half_zone = np.maximum(high - low, 0.0) / 2.0
        else:
            half = np.full(len(pred), q)

        if self.adaptive and self.mondrian != "alarm_zone":
            half = half * np.maximum(knn_q, 1e-12)
        if self.mondrian == "alarm_zone":
            return {
                "low": low,
                "pred": pred,
                "high": high,
                "half_width": half_zone,
                "alpha": alpha,
                "info": info,
            }
        return {
            "low": pred - half,
            "pred": pred,
            "high": pred + half,
            "half_width": half,
            "alpha": alpha,
            "info": info,
        }

    @property
    def calibration_summary(self) -> dict:
        """Diagnostics for verify_claims / audit trails."""
        if not self._fitted:
            raise RuntimeError("fit() must run before calibration_summary")
        r = self.resid_sorted_
        return {
            "n_calibration": int(len(r)),
            "residual_median": float(np.median(r)),
            "residual_q90": float(np.quantile(r, 0.90)),
            "residual_max": float(r[-1]),
            "knn_median": self.knn_median_,
            "adaptive": self.adaptive,
            "k": self.k,
        }

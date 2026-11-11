"""Benchmark: does include_deltas actually help on transient telemetry?

Design (honest and cheap):
- Synthetic 4D AR(1) sensor stream. Bursts are short (3-step) vibration windows
  where the innovation variance spikes 6x and y jumps by +3.5 + 0.8*x3. Marginally
  the burst rows sit in the *same* X distribution as the background (same levels),
  so the only geometric signature of a burst is its rate of change dX. This is the
  exact regime include_deltas was built for.
- Temporal split (first 6,400 rows train, last 2,600 test) - no shuffling, because
  the stream is ordered time.
- Per seed (0..19): build two crystals of M=200 from the SAME stream, identical
  except include_deltas=False vs True (paired by construction). Metrics: burst
  capture rate in the crystal, KRR/GBM RMSE on all test rows and on test burst
  rows, overall R^2. Random coreset M=200 is recorded as capture context only.
- Paired Wilcoxon signed-rank: deltas-on vs deltas-off.

Results -> bench_deltas_results.json (single source of truth).

Usage:
    python bench_deltas.py [--seeds 20]
Writes bench_deltas_v2_results.json (default). The historical
bench_deltas_results.json is never overwritten by default.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from shd import SpectralHologramDistiller


# --- stream -------------------------------------------------------------------


def make_ar1_stream(n=9000, d=4, seed=42, burst_prob=0.006, burst_len=3):
    """AR(1) telemetry with rare vibration bursts.

    X_t = 0.85 X_{t-1} + 0.527 eps_t (unit stationary variance). During a burst
    window the innovation variance inflates 6x and y gains a transient shift.
    Returns X, y, burst_mask.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, d))
    X[0] = rng.normal(size=d)
    burst = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if not burst[t - 1] and rng.random() < burst_prob:
            burst[t : min(t + burst_len, n)] = True
        eps = rng.normal(0.0, np.sqrt(6.0 if burst[t] else 1.0), size=d)
        X[t] = 0.85 * X[t - 1] + 0.527 * eps
    burst[0] = False

    base = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2]
    y = base + 0.1 * rng.normal(size=n)
    y[burst] += 3.5 + 0.8 * X[burst, 3]
    return X, y, burst


# --- crystal builders -----------------------------------------------------------


def build_shd(X_tr, y_tr, seed, m, batch, include_deltas):
    dist = SpectralHologramDistiller(
        crystal_size=m,
        high_band_ratio=0.26,
        sketch_dim=256,
        tail_quantile=84.0,
        phase_kappa=3.2,
        random_state=seed,
        include_deltas=include_deltas,
    )
    for i in range(0, len(X_tr), batch):
        dist.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    dist.finalize_crystal()
    return (
        dist.crystal_X,
        np.asarray(dist.crystal_Y, dtype=np.float64),
        dist.gamma_weights,
    )


def build_random(X_tr, y_tr, seed, m):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=m, replace=False)
    return X_tr[idx], y_tr[idx], None


def capture_count(crystal_X, pool_X):
    """How many rows of `pool` appear verbatim in the crystal (medoids are real rows)."""
    key = lambda a: set(map(tuple, np.round(a, 6)))
    ck = key(crystal_X)
    return int(sum(1 for row in map(tuple, np.round(pool_X, 6)) if row in ck))


# --- model evaluation ------------------------------------------------------------


def eval_models(X_c, y_c, w_c, scaler, model_seed, Z_te, y_te, Z_te_b, y_te_b, rbf_gamma, d):
    Z_c = scaler.transform(X_c)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=rbf_gamma)
    krr.fit(Z_c, y_c, sample_weight=w_c)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=model_seed
    )
    # Fit on the SAME scaled space the predictions use (Z_te). HGBR is
    # invariant to monotonic per-feature scaling at fit, but fitting on raw
    # X_c while predicting on scaled Z_te applies raw-unit split thresholds
    # to standardised values and corrupts the trees.
    gbm.fit(Z_c, y_c, sample_weight=w_c)
    return {
        "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, krr.predict(Z_te)))),
        "krr_rmse_burst": float(np.sqrt(mean_squared_error(y_te_b, krr.predict(Z_te_b)))),
        "krr_r2": float(r2_score(y_te, krr.predict(Z_te))),
        "gbm_rmse_all": float(np.sqrt(mean_squared_error(y_te, gbm.predict(Z_te)))),
        "gbm_rmse_burst": float(np.sqrt(mean_squared_error(y_te_b, gbm.predict(Z_te_b)))),
        "gbm_r2": float(r2_score(y_te, gbm.predict(Z_te))),
    }


# --- main ------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--crystal-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=800)
    ap.add_argument("--output", default="bench_deltas_v2_results.json")
    ap.add_argument("--n-train", type=int, default=6400)
    ap.add_argument("--n-test", type=int, default=2600)
    ap.add_argument("--model-seed", type=int, default=123)
    args = ap.parse_args()

    t0 = time.perf_counter()
    X, y, burst = make_ar1_stream(n=args.n_train + args.n_test)
    n_tr = args.n_train
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    burst_tr, burst_te = burst[:n_tr], burst[n_tr:]
    X_te_b, y_te_b = X_te[burst_te], y_te[burst_te]

    scaler = StandardScaler().fit(X_tr)
    Z_te = scaler.transform(X_te)
    Z_te_b = scaler.transform(X_te_b)
    rbf_gamma = 1.0 / (X.shape[1] * 1.1)

    print(
        f"[deltas-bench] AR(1) stream: train={n_tr} (burst rows {int(burst_tr.sum())} = "
        f"{burst_tr.mean() * 100:.1f}%), test={args.n_test} (burst rows {int(burst_te.sum())})"
    )

    records = []
    for seed in range(args.seeds):
        rec = {"seed": seed}

        Xc_off, yc_off, w_off = build_shd(X_tr, y_tr, seed, args.crystal_size, args.batch, include_deltas=False)
        Xc_on, yc_on, w_on = build_shd(X_tr, y_tr, seed, args.crystal_size, args.batch, include_deltas=True)

        rec["off"] = eval_models(
            Xc_off, yc_off, w_off, scaler, args.model_seed, Z_te, y_te, Z_te_b, y_te_b, rbf_gamma, X.shape[1]
        )
        rec["on"] = eval_models(
            Xc_on, yc_on, w_on, scaler, args.model_seed, Z_te, y_te, Z_te_b, y_te_b, rbf_gamma, X.shape[1]
        )
        rec["off"]["burst_captured"] = capture_count(Xc_off, X_tr[burst_tr])
        rec["on"]["burst_captured"] = capture_count(Xc_on, X_tr[burst_tr])

        Xc_r, yc_r, _ = build_random(X_tr, y_tr, seed, args.crystal_size)
        rec["random_burst_captured"] = capture_count(Xc_r, X_tr[burst_tr])

        records.append(rec)
        print(
            f"[deltas] seed {seed:2d}: burst captured off={rec['off']['burst_captured']:3d} "
            f"on={rec['on']['burst_captured']:3d} random={rec['random_burst_captured']:3d} | "
            f"KRR burst off={rec['off']['krr_rmse_burst']:.2f} on={rec['on']['krr_rmse_burst']:.2f} | "
            f"KRR all off={rec['off']['krr_rmse_all']:.3f} on={rec['on']['krr_rmse_all']:.3f}"
        )

    def med(key, variant=None):
        a = np.array(
            [r[key] if variant is None else r[variant][key] for r in records], float
        )
        return float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))

    summary = {}
    for variant in ("off", "on"):
        for key in ("burst_captured", "krr_rmse_all", "krr_rmse_burst", "gbm_rmse_all", "gbm_rmse_burst", "krr_r2"):
            m, iqr = med(key, variant)
            summary[f"{variant}_{key}_median"] = m
            summary[f"{variant}_{key}_iqr"] = iqr
    m, iqr = med("random_burst_captured")
    summary["random_burst_captured_median"] = m
    summary["random_burst_captured_iqr"] = iqr

    wilcoxon_tests = {}
    for key, higher_better in (("burst_captured", True), ("krr_rmse_burst", False), ("krr_rmse_all", False),
                               ("gbm_rmse_burst", False), ("gbm_rmse_all", False)):
        a = np.array([r["on"][key] for r in records], float)
        b = np.array([r["off"][key] for r in records], float)
        wins = int(np.sum(a > b)) if higher_better else int(np.sum(a < b))
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = 1.0
        wilcoxon_tests[f"deltas_on_vs_off_{key}"] = {"wins": wins, "p_value": p}

    out = {
        "summary": {
            "n_seeds": args.seeds,
            "dataset": "AR(1) telemetry with vibration bursts (synthetic)",
            "crystal_size": args.crystal_size,
            "n_train": n_tr,
            "n_test": args.n_test,
            "burst_share_train_pct": round(float(burst_tr.mean() * 100), 2),
            "burst_share_test_pct": round(float(burst_te.mean() * 100), 2),
            "elapsed_sec": time.perf_counter() - t0,
            **summary,
        },
        "wilcoxon_deltas_on_vs_off": wilcoxon_tests,
        "records": records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {args.output} ({out['summary']['elapsed_sec']:.1f}s)")
    print(f"  burst capture (med): off={summary['off_burst_captured_median']:.1f} "
          f"on={summary['on_burst_captured_median']:.1f} random={summary['random_burst_captured_median']:.1f} "
          f"(of {int(burst_tr.sum())} burst rows)")
    for key in ("krr_rmse_burst", "krr_rmse_all", "gbm_rmse_burst", "gbm_rmse_all"):
        print(f"  {key:14s}: off={summary['off_' + key + '_median']:.3f} on={summary['on_' + key + '_median']:.3f} "
              f"| Wilcoxon wins={wilcoxon_tests['deltas_on_vs_off_' + key]['wins']}/{args.seeds} "
              f"p={wilcoxon_tests['deltas_on_vs_off_' + key]['p_value']:.2e}")


if __name__ == "__main__":
    main()

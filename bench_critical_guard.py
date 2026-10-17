"""Critical-aware SHD + global guard on PM2.5 (M=350 budget).

Combines the two best levers found so far:
- critical-aware phase band: y >= known threshold, 10% of cells (best critical
  in the 5-seed sweep: 207.98 vs 235.64 base);
- 50 K-Means global-guard medoids to recover background accuracy.

Compares against the M=350 baselines from bench_budget_pm25_results.json.

Usage:
    python bench_critical_guard.py --seeds 5
Writes bench_critical_guard_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kmeans, eval_models
from shd import SpectralHologramDistiller


def build_critical_shd(X_tr, y_tr, seed, crit, m=300, batch=2500):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.10, sketch_dim=256,
        tail_quantile=84.0, critical_threshold=crit, auto_band_ratio=False,
        random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, int(np.sum(d.crystal_Y >= crit))


def run(seeds, m_shd=300, m_global=50, batch=2500, model_seed=42):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_tr.shape[1] * 1.1)

    out = {"dataset": "pm25", "n_train": int(len(X_tr)), "n_test_crit": int(len(X_te_crit)),
           "crit_threshold": crit, "m_shd": m_shd, "m_global": m_global,
           "n_seeds": len(seeds), "records": [], "summary": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        Xc, yc, wc, n_rare = build_critical_shd(X_tr, y_tr, seed, crit, m_shd, batch)
        rec = {"seed": int(seed)}
        rec["crit_shd"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                      X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["crit_shd"]["rare_captured"] = n_rare

        Xg, yg, wg = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m_global)
        if wg is None:
            wg = np.ones(len(Xg))
        Xp = np.vstack([Xc, Xg])
        yp = np.concatenate([yc, yg])
        wp = np.concatenate([wc, wg])
        wp /= np.mean(wp)
        rec["crit_guard"] = eval_models(Xp, yp, wp, scaler, model_seed, X_te, Z_te, y_te,
                                        X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["crit_guard"]["rare_captured"] = int(np.sum(yp >= crit))
        out["records"].append(rec)
        print(f"seed {seed}: crit_shd={rec['crit_shd']['krr_rmse_crit']:.1f} "
              f"crit_guard={rec['crit_guard']['krr_rmse_crit']:.1f} "
              f"all_shd={rec['crit_shd']['krr_rmse_all']:.1f} "
              f"all_guard={rec['crit_guard']['krr_rmse_all']:.1f} "
              f"gbm_guard={rec['crit_guard']['gbm_rmse_all']:.1f}", flush=True)

    for arm in ("crit_shd", "crit_guard"):
        out["summary"][arm] = {
            "krr_crit_median": float(np.median([r[arm]["krr_rmse_crit"] for r in out["records"]])),
            "krr_all_median": float(np.median([r[arm]["krr_rmse_all"] for r in out["records"]])),
            "gbm_crit_median": float(np.median([r[arm]["gbm_rmse_crit"] for r in out["records"]])),
            "gbm_all_median": float(np.median([r[arm]["gbm_rmse_all"] for r in out["records"]])),
            "rare_median": float(np.median([r[arm]["rare_captured"] for r in out["records"]])),
        }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--global-size", type=int, default=50)
    ap.add_argument("--output", default="bench_critical_guard_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)), m_global=args.global_size)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:10s} crit={v['krr_crit_median']:.1f} all={v['krr_all_median']:.1f} "
              f"gbm_all={v['gbm_all_median']:.1f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

"""PM2.5 hybrid experiments: can a global guard recover the background?

The tuning sweep showed narrower tail bands make PM2.5 worse, so the global
loss is not just band width. This script tests three cheap fixes on top of the
published SHD crystal:

- ``shd_uniform``: ignore SHD sample weights (uniform).
- ``shd_clip3``: winsorize SHD weights at 3x median.
- ``shd_swap50``: replace the 50 lowest-weight SHD medoids with 50 K-Means
  global-guard medoids (total M=300).
- ``shd_plus50``: append 50 global medoids (M=350).
- ``shd_250_plus50``: SHD M=250 + 50 global medoids (M=300).

Usage:
    python bench_pm25_hybrid.py --seeds 5
Writes bench_pm25_hybrid_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kmeans, build_shd, eval_models


def clip_weights(w, factor=3.0):
    w = np.asarray(w, dtype=np.float64).copy()
    cap = factor * np.median(w)
    w = np.minimum(w, cap)
    return w / np.mean(w)


def run(seeds, m=300, m_global=50, batch=2500, model_seed=42):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    out = {"dataset": "pm25", "n_total": int(len(X)), "n_test_crit": int(len(X_te_crit)),
           "crit_threshold": crit, "m": m, "m_global": m_global,
           "n_seeds": len(seeds), "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        rec = {"seed": int(seed)}
        rec["shd"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                 X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd"]["rare_captured"] = n_rare

        rec["shd_uniform"] = eval_models(Xc, yc, None, scaler, model_seed, X_te, Z_te, y_te,
                                         X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_uniform"]["rare_captured"] = n_rare

        w_clip = clip_weights(wc, 3.0)
        rec["shd_clip3"] = eval_models(Xc, yc, w_clip, scaler, model_seed, X_te, Z_te, y_te,
                                       X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_clip3"]["rare_captured"] = n_rare

        Xg, yg, wg = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m_global)
        if wg is None:
            wg = np.ones(len(Xg))
        # swap: drop the 50 lowest-weight SHD points, append global guard
        order = np.argsort(wc)
        keep = order[m_global:]
        X_swap = np.vstack([Xc[keep], Xg])
        y_swap = np.concatenate([yc[keep], yg])
        w_swap = np.concatenate([wc[keep], wg])
        w_swap /= np.mean(w_swap)
        rec["shd_swap50"] = eval_models(X_swap, y_swap, w_swap, scaler, model_seed, X_te, Z_te, y_te,
                                        X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_swap50"]["rare_captured"] = int(np.sum(y_swap >= crit))

        # clipped weights + global guard
        order_c = np.argsort(w_clip)
        keep_c = order_c[m_global:]
        X_cs = np.vstack([Xc[keep_c], Xg])
        y_cs = np.concatenate([yc[keep_c], yg])
        w_cs = np.concatenate([w_clip[keep_c], wg])
        w_cs /= np.mean(w_cs)
        rec["shd_clip_swap50"] = eval_models(X_cs, y_cs, w_cs, scaler, model_seed, X_te, Z_te, y_te,
                                             X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_clip_swap50"]["rare_captured"] = int(np.sum(y_cs >= crit))

        X_plus = np.vstack([Xc, Xg])
        y_plus = np.concatenate([yc, yg])
        w_plus = np.concatenate([wc, wg])
        w_plus /= np.mean(w_plus)
        rec["shd_plus50"] = eval_models(X_plus, y_plus, w_plus, scaler, model_seed, X_te, Z_te, y_te,
                                        X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_plus50"]["rare_captured"] = int(np.sum(y_plus >= crit))

        Xc250, yc250, wc250, n_rare250 = build_shd(X_tr, y_tr, seed, m - m_global, batch, crit)
        X_250 = np.vstack([Xc250, Xg])
        y_250 = np.concatenate([yc250, yg])
        w_250 = np.concatenate([wc250, wg])
        w_250 /= np.mean(w_250)
        rec["shd_250_plus50"] = eval_models(X_250, y_250, w_250, scaler, model_seed, X_te, Z_te, y_te,
                                            X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_250_plus50"]["rare_captured"] = int(np.sum(y_250 >= crit))

        out["records"].append(rec)
        print(f"seed {seed}: shd={rec['shd']['krr_rmse_crit']:.1f} "
              f"swap={rec['shd_swap50']['krr_rmse_crit']:.1f} "
              f"crit_all={rec['shd']['krr_rmse_all']:.1f} "
              f"swap_all={rec['shd_swap50']['krr_rmse_all']:.1f} "
              f"plus_all={rec['shd_plus50']['krr_rmse_all']:.1f}", flush=True)

    arms = [k for k in out["records"][0] if k != "seed"]
    out["summary"] = {}
    for arm in arms:
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
    ap.add_argument("--output", default="bench_pm25_hybrid_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:16s} crit={v['krr_crit_median']:.1f} all={v['krr_all_median']:.1f} "
              f"gbm_all={v['gbm_all_median']:.1f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

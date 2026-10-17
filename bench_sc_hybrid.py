"""SC hybrid control: does the global guard hurt the headline critical result?

Same arms as the PM2.5 hybrid experiment, on UCI Superconductivity:
- SHD base M=300
- uniform / clipped weights
- swap 50 lowest-weight SHD medoids for 50 K-Means guard medoids (M=300)
- append 50 (M=350) / SHD 250 + 50 (M=300)

Usage:
    python bench_sc_hybrid.py --seeds 5
Writes bench_sc_hybrid_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_pm25_hybrid import clip_weights
from bench_seeds import (
    build_kmeans,
    build_shd,
    eval_models,
    load_superconduct,
)


def run(seeds, m=300, m_global=50, batch=2500, model_seed=42):
    X, y, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    out = {"dataset": "superconduct", "n_train": int(len(X_tr)), "n_test_crit": int(len(X_te_crit)),
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
        order = np.argsort(wc)
        keep = order[m_global:]
        X_swap = np.vstack([Xc[keep], Xg])
        y_swap = np.concatenate([yc[keep], yg])
        w_swap = np.concatenate([wc[keep], wg])
        w_swap /= np.mean(w_swap)
        rec["shd_swap50"] = eval_models(X_swap, y_swap, w_swap, scaler, model_seed, X_te, Z_te, y_te,
                                        X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_swap50"]["rare_captured"] = int(np.sum(y_swap >= crit))

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
        print(f"seed {seed}: shd={rec['shd']['krr_rmse_crit']:.2f} "
              f"swap={rec['shd_swap50']['krr_rmse_crit']:.2f} "
              f"shd_all={rec['shd']['krr_rmse_all']:.2f} swap_all={rec['shd_swap50']['krr_rmse_all']:.2f}",
              flush=True)

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
    ap.add_argument("--output", default="bench_sc_hybrid_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:16s} crit={v['krr_crit_median']:.2f} all={v['krr_all_median']:.2f} "
              f"gbm_all={v['gbm_all_median']:.2f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

"""Budget-matched test: SHD + global guard at M=350 vs baselines at M=350.

The hybrid experiments compared SHD+50 (M=350) to baselines at M=300, which is
not a fair memory budget. This script gives every arm M=350:

- Random / K-Means / KT / SKT at M=350.
- SHD at M=350.
- SHD at M=300 + 50 K-Means global-guard medoids (M=350).

Usage:
    python bench_budget.py --dataset superconduct --seeds 5
    python bench_budget.py --dataset pm25 --seeds 5
Writes bench_budget_<dataset>_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import (
    build_kt,
    build_kmeans,
    build_random,
    build_shd,
    build_skt,
    eval_models,
    load_superconduct,
)


def load_data(name):
    if name == "pm25":
        X, y, title = load_pm25()
        n_tr = int(round(0.80 * len(X)))
        return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], float(np.percentile(y[:n_tr], 97.0)), title
    X, y, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_tr, y_tr, X_te, y_te, float(crit), title


def run(dataset, seeds, m=350, m_shd=300, batch=2500, model_seed=42):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_tr.shape[1] * 1.1)

    out = {"dataset": dataset, "title": title, "m": m, "m_shd": m_shd,
           "n_train": int(len(X_tr)), "n_test_crit": int(len(X_te_crit)),
           "crit_threshold": float(crit), "n_seeds": len(seeds), "records": [],
           "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        for arm, builder in (
            ("random", lambda: build_random(X_tr, y_tr, seed, m)),
            ("kmeans", lambda: build_kmeans(X_tr, y_tr, Z_tr, seed, m)),
            ("kt", lambda: build_kt(X_tr, y_tr, Z_tr, seed, m)),
            ("skt", lambda: build_skt(X_tr, y_tr, Z_tr, seed, m)),
        ):
            Xc, yc, wc = builder()
            rec[arm] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                   X_te_crit, Z_te_crit, y_te_crit, gamma)
            rec[arm]["rare_captured"] = int(np.sum(yc >= crit))

        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        rec["shd"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                 X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd"]["rare_captured"] = n_rare

        Xh, yh, wh, _ = build_shd(X_tr, y_tr, seed, m_shd, batch, crit)
        m_g = m - m_shd
        Xg, yg, wg = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m_g)
        if wg is None:
            wg = np.ones(len(Xg))
        X_plus = np.vstack([Xh, Xg])
        y_plus = np.concatenate([yh, yg])
        w_plus = np.concatenate([wh, wg])
        w_plus /= np.mean(w_plus)
        rec["shd_guard"] = eval_models(X_plus, y_plus, w_plus, scaler, model_seed, X_te, Z_te, y_te,
                                       X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_guard"]["rare_captured"] = int(np.sum(y_plus >= crit))

        out["records"].append(rec)
        print(f"seed {seed}: shd={rec['shd']['krr_rmse_crit']:.3f} "
              f"guard={rec['shd_guard']['krr_rmse_crit']:.3f} "
              f"kt={rec['kt']['krr_rmse_crit']:.3f} "
              f"shd_all={rec['shd']['krr_rmse_all']:.3f} "
              f"guard_all={rec['shd_guard']['krr_rmse_all']:.3f}", flush=True)

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
    ap.add_argument("--dataset", choices=("superconduct", "pm25"), default="pm25")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--m", type=int, default=350)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run(args.dataset, list(range(args.seeds)), m=args.m)
    path = args.output or f"bench_budget_{args.dataset}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:10s} crit={v['krr_crit_median']:.2f} all={v['krr_all_median']:.2f} "
              f"gbm_all={v['gbm_all_median']:.2f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

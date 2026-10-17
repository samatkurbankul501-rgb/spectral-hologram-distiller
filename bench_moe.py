"""SHD-MoE: route critical rows to SHD, background to a global coreset.

The hybrid guard improves PM2.5 global but its critical gain is seed-dependent.
This experiment keeps two small experts and a hard gate:

- SHD crystal (rare-focused) -> KRR/GBM predictions.
- K-Means global coreset (background) -> KRR/GBM predictions.
- Gate: use SHD where the global prediction is in the upper tail
  (pred >= critical threshold or a train-y quantile); otherwise global.

This is a deployment mode, not a change to the SHD coreset. It tests whether a
two-expert system is Pareto-better than either expert alone.

Usage:
    python bench_moe.py --dataset pm25 --seeds 5
    python bench_moe.py --dataset superconduct --seeds 5
Writes bench_moe_<dataset>_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kmeans, build_shd, load_superconduct


def load_data(name):
    if name == "pm25":
        X, y, title = load_pm25()
        n_tr = int(round(0.80 * len(X)))
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]
        crit = float(np.percentile(y_tr, 97.0))
        return X_tr, y_tr, X_te, y_te, crit, title
    X, y, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_tr, y_tr, X_te, y_te, float(crit), title


def fit_predict(Xc, yc, wc, Z_te, d_feat, scaler):
    Zc = scaler.transform(Xc)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (d_feat * 1.1))
    krr.fit(Zc, yc, sample_weight=wc)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42
    )
    gbm.fit(Zc, yc, sample_weight=wc)
    return krr.predict(Z_te), gbm.predict(Z_te)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run(dataset, seeds, m=300, m_global=300, batch=2500):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    d_feat = X_tr.shape[1]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    gates = {
        "crit": crit,
        "q90": float(np.percentile(y_tr, 90.0)),
        "q95": float(np.percentile(y_tr, 95.0)),
        "q97": float(np.percentile(y_tr, 97.0)),
    }
    out = {"dataset": dataset, "title": title, "n_train": int(len(X_tr)),
           "n_test": int(len(X_te)), "n_test_crit": int(np.sum(crit_mask)),
           "crit_threshold": crit, "m": m, "m_global": m_global,
           "n_seeds": len(seeds), "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        Xs, ys, ws, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        Xg, yg, wg = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m_global)
        if wg is None:
            wg = np.ones(len(Xg))
        p_s_krr, p_s_gbm = fit_predict(Xs, ys, ws, Z_te, d_feat, scaler)
        p_g_krr, p_g_gbm = fit_predict(Xg, yg, wg, Z_te, d_feat, scaler)
        rec = {"seed": int(seed), "shd": {"krr_rmse_all": rmse(y_te, p_s_krr),
                                          "krr_rmse_crit": rmse(y_te[crit_mask], p_s_krr[crit_mask]),
                                          "gbm_rmse_all": rmse(y_te, p_s_gbm),
                                          "gbm_rmse_crit": rmse(y_te[crit_mask], p_s_gbm[crit_mask]),
                                          "rare_captured": n_rare},
               "kmeans": {"krr_rmse_all": rmse(y_te, p_g_krr),
                          "krr_rmse_crit": rmse(y_te[crit_mask], p_g_krr[crit_mask]),
                          "gbm_rmse_all": rmse(y_te, p_g_gbm),
                          "gbm_rmse_crit": rmse(y_te[crit_mask], p_g_gbm[crit_mask]),
                          "rare_captured": int(np.sum(yg >= crit))}}
        for gate_name, gate_val in gates.items():
            # Use the global KRR prediction as the gate; same gate for both KRR/GBM.
            g = p_g_krr >= gate_val
            p_krr = np.where(g, p_s_krr, p_g_krr)
            p_gbm = np.where(g, p_s_gbm, p_g_gbm)
            rec[f"moe_{gate_name}"] = {
                "krr_rmse_all": rmse(y_te, p_krr),
                "krr_rmse_crit": rmse(y_te[crit_mask], p_krr[crit_mask]),
                "gbm_rmse_all": rmse(y_te, p_gbm),
                "gbm_rmse_crit": rmse(y_te[crit_mask], p_gbm[crit_mask]),
                "gate_fraction": float(np.mean(g)),
                "rare_captured": n_rare,
            }
        out["records"].append(rec)
        print(f"seed {seed}: shd crit={rec['shd']['krr_rmse_crit']:.2f} "
              f"all={rec['shd']['krr_rmse_all']:.2f} | "
              f"moe_crit crit={rec['moe_crit']['krr_rmse_crit']:.2f} "
              f"all={rec['moe_crit']['krr_rmse_all']:.2f} | "
              f"moe_q95 all={rec['moe_q95']['krr_rmse_all']:.2f}", flush=True)
    arms = [k for k in out["records"][0] if k != "seed"]
    out["summary"] = {}
    for arm in arms:
        out["summary"][arm] = {
            "krr_all_median": float(np.median([r[arm]["krr_rmse_all"] for r in out["records"]])),
            "krr_crit_median": float(np.median([r[arm]["krr_rmse_crit"] for r in out["records"]])),
            "gbm_all_median": float(np.median([r[arm]["gbm_rmse_all"] for r in out["records"]])),
            "gbm_crit_median": float(np.median([r[arm]["gbm_rmse_crit"] for r in out["records"]])),
            "rare_median": float(np.median([r[arm]["rare_captured"] for r in out["records"]])),
        }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pm25", "superconduct"), default="pm25")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run(args.dataset, list(range(args.seeds)))
    path = args.output or f"bench_moe_{args.dataset}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:12s} crit={v['krr_crit_median']:.2f} all={v['krr_all_median']:.2f} "
              f"gbm_all={v['gbm_all_median']:.2f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

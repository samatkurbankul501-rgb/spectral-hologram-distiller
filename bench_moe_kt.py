"""SHD (critical-aware) + KT portfolio with a prediction gate on PM2.5.

SHD is the best critical expert; KT is the best global expert. This script
keeps both crystals (600 points total) and routes by the predicted target:

- gate on SHD prediction: rows with pred_shd >= threshold use SHD, others KT;
- gate on KT prediction: same, but the gate is KT's prediction.

If the gate routes the alarm regime to SHD and the background to KT, the
portfolio can be Pareto-better than either expert alone.

Usage:
    python bench_moe_kt.py --seeds 5
Writes bench_moe_kt_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kt, load_superconduct
from shd import SpectralHologramDistiller
from sklearn.model_selection import train_test_split


def load_data(name):
    if name == "pm25":
        X, y, title = load_pm25()
        n_tr = int(round(0.80 * len(X)))
        return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], float(np.percentile(y[:n_tr], 97.0)), title
    X, y, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_tr, y_tr, X_te, y_te, float(crit), title


def build_critical_shd(X_tr, y_tr, seed, crit, m=300, batch=2500):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.10, sketch_dim=256,
        tail_quantile=84.0, critical_threshold=crit, auto_band_ratio=True,
        random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights


def fit_predict(Zc, yc, wc, Z_te, d_feat):
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (d_feat * 1.1))
    krr.fit(Zc, yc, sample_weight=wc)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42
    )
    gbm.fit(Zc, yc, sample_weight=wc)
    return krr.predict(Z_te), gbm.predict(Z_te)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def run(seeds, m=300, batch=2500, dataset="pm25"):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
    d_feat = X_tr.shape[1]
    gates = {"crit": crit, "q90": float(np.percentile(y_tr, 90.0)),
             "q95": float(np.percentile(y_tr, 95.0))}

    out = {"dataset": dataset, "n_test_crit": int(np.sum(crit_mask)),
           "crit_threshold": crit, "m_shd": m, "m_kt": m,
           "n_seeds": len(seeds), "records": [], "summary": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        Xs, ys, ws = build_critical_shd(X_tr, y_tr, seed, crit, m, batch)
        Xk, yk, wk = build_kt(X_tr, y_tr, Z_tr, seed + 5000, m)
        if wk is None:
            wk = np.ones(len(Xk))
        Zs, Zk = scaler.transform(Xs), scaler.transform(Xk)
        p_s_krr, p_s_gbm = fit_predict(Zs, ys, ws, Z_te, d_feat)
        p_k_krr, p_k_gbm = fit_predict(Zk, yk, wk, Z_te, d_feat)
        rec = {"seed": int(seed)}
        rec["shd"] = {"krr_rmse_all": rmse(y_te, p_s_krr),
                      "krr_rmse_crit": rmse(y_te[crit_mask], p_s_krr[crit_mask]),
                      "gbm_rmse_all": rmse(y_te, p_s_gbm),
                      "gbm_rmse_crit": rmse(y_te[crit_mask], p_s_gbm[crit_mask])}
        rec["kt"] = {"krr_rmse_all": rmse(y_te, p_k_krr),
                     "krr_rmse_crit": rmse(y_te[crit_mask], p_k_krr[crit_mask]),
                     "gbm_rmse_all": rmse(y_te, p_k_gbm),
                     "gbm_rmse_crit": rmse(y_te[crit_mask], p_k_gbm[crit_mask])}
        for gate_name, gate_val in gates.items():
            for gate_src, gate_pred in (("shd", p_s_krr), ("kt", p_k_krr)):
                g = gate_pred >= gate_val
                p_krr = np.where(g, p_s_krr, p_k_krr)
                p_gbm = np.where(g, p_s_gbm, p_k_gbm)
                rec[f"moe_{gate_src}_{gate_name}"] = {
                    "krr_rmse_all": rmse(y_te, p_krr),
                    "krr_rmse_crit": rmse(y_te[crit_mask], p_krr[crit_mask]),
                    "gbm_rmse_all": rmse(y_te, p_gbm),
                    "gbm_rmse_crit": rmse(y_te[crit_mask], p_gbm[crit_mask]),
                    "gate_fraction": float(np.mean(g)),
                }
        out["records"].append(rec)
        print(f"seed {seed}: shd={rec['shd']['krr_rmse_crit']:.1f}/{rec['shd']['krr_rmse_all']:.1f} "
              f"kt={rec['kt']['krr_rmse_crit']:.1f}/{rec['kt']['krr_rmse_all']:.1f} "
              f"moe_shd_crit={rec['moe_shd_crit']['krr_rmse_crit']:.1f}/"
              f"{rec['moe_shd_crit']['krr_rmse_all']:.1f}", flush=True)

    arms = [k for k in out["records"][0] if k != "seed"]
    for arm in arms:
        out["summary"][arm] = {
            "krr_crit_median": float(np.median([r[arm]["krr_rmse_crit"] for r in out["records"]])),
            "krr_all_median": float(np.median([r[arm]["krr_rmse_all"] for r in out["records"]])),
            "gbm_crit_median": float(np.median([r[arm]["gbm_rmse_crit"] for r in out["records"]])),
            "gbm_all_median": float(np.median([r[arm]["gbm_rmse_all"] for r in out["records"]])),
        }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pm25", "superconduct"), default="pm25")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run(list(range(args.seeds)), dataset=args.dataset)
    path = args.output or f"bench_moe_kt_{args.dataset}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:16s} crit={v['krr_crit_median']:.1f} all={v['krr_all_median']:.1f} "
              f"gbm_all={v['gbm_all_median']:.1f}")


if __name__ == "__main__":
    main()

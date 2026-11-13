"""Second real dataset: Beijing PM2.5 sensor telemetry.

The paper's only real dataset is UCI Superconductivity; the reactor/BO/delta
streams are synthetic. This benchmark uses the real UCI Beijing PM2.5 dataset
(43,824 hourly observations, 2010-2014) with a time-ordered 80/20 split, so the
test stream is genuinely in the future. The critical regime is the top 3% of
training PM2.5 (rare pollution peaks).

Usage:
    python bench_pm25.py --seeds 5
Writes bench_pm25_results.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.request

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import (
    build_kt,
    build_kmeans,
    build_random,
    build_shd,
    build_skt,
    eval_models,
)

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
       "PRSA_data_2010.1.1-2014.12.31.csv")


def load_pm25():
    # Full UCI CSV committed under data/ so regens run offline; the truncated
    # data/pm25.csv cache belongs to bench_passport_reference and is NOT this file.
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "pm25_full.csv")
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            text = r.read().decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    cbwd_values = ["NW", "NE", "SE", "cv"]
    data = []
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        try:
            y = float(r[5])
            nums = [float(r[i]) for i in (1, 2, 3, 4, 6, 7, 8, 10, 11, 12)]
        except (ValueError, IndexError):
            continue
        cb = r[9]
        if cb not in cbwd_values:
            continue
        onehot = [1.0 if cb == v else 0.0 for v in cbwd_values]
        data.append(nums + onehot + [y])
    arr = np.asarray(data, dtype=np.float64)
    return arr[:, :-1], arr[:, -1], "Beijing PM2.5 (real sensor telemetry)"


def run(seeds, m=300, batch=2500, model_seed=42):
    X, y, title = load_pm25()
    # Time-ordered split: the test period is strictly after the training period.
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)
    print(f"{title}: n={len(X):,} train={len(X_tr):,} test={len(X_te):,} "
          f"crit_thr={crit:.1f} test_crit={len(X_te_crit):,}", flush=True)

    out = {"dataset": "pm25", "title": title, "n_total": int(len(X)),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "n_features": int(X.shape[1]), "n_seeds": len(seeds), "records": [],
           "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        rec["shd"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                 X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd"]["rare_captured"] = n_rare

        Xc, yc, wc = build_random(X_tr, y_tr, seed, m)
        rec["random"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                    X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["random"]["rare_captured"] = int(np.sum(yc >= crit))

        Xc, yc, wc = build_kmeans(X_tr, y_tr, Z_tr, seed, m)
        rec["kmeans"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                    X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["kmeans"]["rare_captured"] = int(np.sum(yc >= crit))

        Xc, yc, wc = build_kt(X_tr, y_tr, Z_tr, seed, m)
        rec["kt"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["kt"]["rare_captured"] = int(np.sum(yc >= crit))

        Xc, yc, wc = build_skt(X_tr, y_tr, Z_tr, seed, m)
        rec["skt"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                 X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["skt"]["rare_captured"] = int(np.sum(yc >= crit))
        out["records"].append(rec)
        print(f"seed {seed}: shd={rec['shd']['krr_rmse_crit']:.2f} "
              f"kt={rec['kt']['krr_rmse_crit']:.2f} "
              f"skt={rec['skt']['krr_rmse_crit']:.2f} "
              f"km={rec['kmeans']['krr_rmse_crit']:.2f} "
              f"rare={n_rare}", flush=True)

    tests = {}
    for pair in ("random", "kmeans", "kt", "skt"):
        for metric in ("krr_rmse_all", "krr_rmse_crit", "krr_r2",
                       "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"):
            higher = metric.endswith("_r2")
            a = np.array([r["shd"][metric] for r in out["records"]])
            b = np.array([r[pair][metric] for r in out["records"]])
            diff = a - b
            stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
            wins = int(np.sum(diff > 0)) if higher else int(np.sum(diff < 0))
            tests[f"shd_vs_{pair}_{metric}"] = {
                "median_shd": float(np.median(a)), "median_" + pair: float(np.median(b)),
                "wins_shd": wins, "statistic": float(stat), "p_value": float(p),
            }
    out["wilcoxon"] = tests
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="bench_pm25_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for pair in ("random", "kmeans", "kt", "skt"):
        k = f"shd_vs_{pair}_krr_rmse_crit"
        if k in res["wilcoxon"]:
            v = res["wilcoxon"][k]
            print(f"  {pair:7s} crit: SHD {v['median_shd']:.2f} vs {v['median_' + pair]:.2f} "
                  f"wins={v['wins_shd']}/{len(res['records'])} p={v['p_value']:.3g}")


if __name__ == "__main__":
    main()

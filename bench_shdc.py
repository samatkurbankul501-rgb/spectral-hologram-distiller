"""20-seed confirmation of SHD-C (critical-aware auto band).

The critical-aware mode was piloted on 5 seeds. This script runs the single
SHD-C configuration (critical_threshold = known alarm threshold,
auto_band_ratio=True) for 20 seeds on SC or PM2.5.

Usage:
    python bench_shdc.py --dataset superconduct --seeds 20
    python bench_shdc.py --dataset pm25 --seeds 20
Writes bench_shdc_<dataset>_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import eval_models, load_superconduct
from shd import SpectralHologramDistiller


def load_data(name):
    if name == "pm25":
        X, y, title = load_pm25()
        n_tr = int(round(0.80 * len(X)))
        return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], float(np.percentile(y[:n_tr], 97.0)), title
    X, y, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_tr, y_tr, X_te, y_te, float(crit), title


def run(dataset, seeds, m=300, batch=2500, model_seed=42):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_tr.shape[1] * 1.1)

    out = {"dataset": dataset, "title": title, "n_train": int(len(X_tr)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "m": m, "n_seeds": len(seeds), "records": [], "summary": {},
           "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        d = SpectralHologramDistiller(
            crystal_size=m, sketch_dim=256, random_state=seed,
            critical_threshold=crit, auto_band_ratio=True,
        )
        for i in range(0, len(X_tr), batch):
            d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
        d.finalize_crystal()
        met = eval_models(d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64),
                          d.gamma_weights, scaler, model_seed, X_te, Z_te, y_te,
                          X_te_crit, Z_te_crit, y_te_crit, gamma)
        met["rare_captured"] = int(np.sum(d.crystal_Y >= crit))
        out["records"].append(met)
        print(f"seed {seed}: crit={met['krr_rmse_crit']:.3f} "
              f"all={met['krr_rmse_all']:.3f} gbm={met['gbm_rmse_all']:.3f} "
              f"rare={met['rare_captured']}", flush=True)

    for metric in ("krr_rmse_crit", "krr_rmse_all", "gbm_rmse_crit", "gbm_rmse_all", "rare_captured"):
        vals = np.array([r[metric] for r in out["records"]], dtype=np.float64)
        out["summary"][metric] = {
            "median": float(np.median(vals)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("superconduct", "pm25"), default="superconduct")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run(args.dataset, list(range(args.seeds)))
    path = args.output or f"bench_shdc_{args.dataset}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({res['elapsed_sec']:.0f}s)")
    for k, v in res["summary"].items():
        print(f"  {k:16s} {v['median']:.3f} / IQR {v['iqr']:.3f}")


if __name__ == "__main__":
    main()

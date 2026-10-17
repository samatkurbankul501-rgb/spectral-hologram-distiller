"""Critical-aware phase band: use a known alarm threshold instead of a score percentile.

The default SHD reserves 26% of cells using the 84th score percentile. For a
domain with a known critical threshold (e.g. PM2.5 >= 324, Tc >= 77 K), the
band should be defined directly on y. This script compares:

- base: score percentile (tail_quantile=84, high_band_ratio=0.26)
- crit_auto: y >= threshold, cells = 2x observed critical fraction (3%-30%)
- crit_h10 / crit_h26: y >= threshold, fixed 10% / 26% cells

Usage:
    python bench_critical_aware.py --dataset pm25 --seeds 5
    python bench_critical_aware.py --dataset superconduct --seeds 5
Writes bench_critical_aware_<dataset>_results.json.
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


def build(X_tr, y_tr, seed, crit, params):
    d = SpectralHologramDistiller(crystal_size=300, sketch_dim=256, random_state=seed, **params)
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i : i + 2500], y_tr[i : i + 2500])
    d.finalize_crystal()
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, int(np.sum(d.crystal_Y >= crit))


def run(dataset, seeds):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_tr.shape[1] * 1.1)

    configs = {
        "base": dict(tail_quantile=84.0, high_band_ratio=0.26),
        "crit_auto": dict(critical_threshold=crit, auto_band_ratio=True),
        "crit_h10": dict(critical_threshold=crit, high_band_ratio=0.10),
        "crit_h26": dict(critical_threshold=crit, high_band_ratio=0.26),
        "crit_h05": dict(critical_threshold=crit, high_band_ratio=0.05),
    }
    out = {"dataset": dataset, "title": title, "crit_threshold": crit,
           "n_train": int(len(X_tr)), "n_test_crit": int(len(X_te_crit)),
           "n_seeds": len(seeds), "configs": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for name, params in configs.items():
        recs = []
        for seed in seeds:
            Xc, yc, wc, n_rare = build(X_tr, y_tr, seed, crit, params)
            met = eval_models(Xc, yc, wc, scaler, 42, X_te, Z_te, y_te,
                              X_te_crit, Z_te_crit, y_te_crit, gamma)
            met["rare_captured"] = n_rare
            recs.append(met)
        out["configs"][name] = {
            "params": {k: (None if v is None else v) for k, v in params.items()},
            "records": recs,
            "krr_crit_median": float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_all_median": float(np.median([r["krr_rmse_all"] for r in recs])),
            "gbm_crit_median": float(np.median([r["gbm_rmse_crit"] for r in recs])),
            "gbm_all_median": float(np.median([r["gbm_rmse_all"] for r in recs])),
            "rare_median": float(np.median([r["rare_captured"] for r in recs])),
        }
        print(f"{name}: crit={out['configs'][name]['krr_crit_median']:.2f} "
              f"all={out['configs'][name]['krr_all_median']:.2f} "
              f"gbm={out['configs'][name]['gbm_all_median']:.2f} "
              f"rare={out['configs'][name]['rare_median']:.0f}", flush=True)
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pm25", "superconduct"), default="pm25")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    res = run(args.dataset, list(range(args.seeds)))
    path = args.output or f"bench_critical_aware_{args.dataset}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({res['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()

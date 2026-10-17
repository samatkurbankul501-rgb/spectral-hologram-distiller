"""Sensitivity / ablation of SHD's architectural constants.

For each constant we run the full distill -> KRR/GBM evaluation pipeline on the
4D phase-transition reactor (fast, 5 seeds) and report critical-regime and
background RMSE plus rare-regime capture. One constant is varied at a time;
every other constant stays at its published default.

Usage:
    python bench_ablation.py [--seeds 5] [--dataset reactor|superconduct]
Writes bench_ablation_results.json (single source of truth).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bench_seeds import eval_models, load_reactor, load_superconduct
from shd import SpectralHologramDistiller
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULTS = dict(
    phase_kappa=3.2,
    high_band_ratio=0.26,
    voronoi_temper=0.55,
    herding_mix=0.75,
    center_lr=0.25,
    fista_l2_reg=0.20,
    omega_low_scale=0.75,
    omega_high_scale=2.10,
    tail_quantile=84.0,
)

# one-at-a-time grids (value lists always include the default)
GRIDS = {
    "phase_kappa": [0.0, 1.6, 3.2, 6.4],
    "high_band_ratio": [0.10, 0.26, 0.50],
    "voronoi_temper": [0.0, 0.30, 0.55, 1.00],
    "herding_mix": [1.0, 0.75, 0.5, 0.0],
    "center_lr": [0.05, 0.25, 0.50],
    "fista_l2_reg": [0.05, 0.20, 0.60],
    "omega_low_scale": [0.35, 0.75, 1.50],
    "omega_high_scale": [0.75, 2.10, 4.20],
    "tail_quantile": [74.0, 84.0, 94.0],
}


def build(X_tr, y_tr, seed, m, batch, params, crit_threshold):
    d = SpectralHologramDistiller(
        crystal_size=m, sketch_dim=256, random_state=seed, **params
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    n_rare = int(np.sum(d.crystal_Y >= crit_threshold))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare


def run(dataset, seeds, m):
    if dataset == "reactor":
        X_raw, y_raw, crit_threshold, title = load_reactor()
    else:
        X_raw, y_raw, crit_threshold, title = load_superconduct()

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit_threshold
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma_rbf = 1.0 / (X_raw.shape[1] * 1.1)

    out = {"dataset": dataset, "title": title, "n_seeds": len(seeds),
           "crystal_size": m, "defaults": DEFAULTS, "sweeps": {}}

    for const, values in GRIDS.items():
        rows = []
        for val in values:
            params = dict(DEFAULTS)
            params[const] = val
            recs = []
            t0 = time.perf_counter()
            for seed in seeds:
                Xc, yc, wc, n_rare = build(X_tr, y_tr, seed, m, 2500, params, crit_threshold)
                met = eval_models(Xc, yc, wc, scaler, 42, X_te, Z_te, y_te,
                                  X_te_crit, Z_te_crit, y_te_crit, gamma_rbf)
                met["rare_captured"] = n_rare
                recs.append(met)
            row = {"value": val}
            for metric in ("krr_rmse_crit", "krr_rmse_all", "gbm_rmse_crit", "gbm_rmse_all", "rare_captured"):
                vals = np.array([r[metric] for r in recs])
                row[metric + "_median"] = float(np.median(vals))
                row[metric + "_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            row["elapsed_sec"] = time.perf_counter() - t0
            rows.append(row)
            print(f"[{dataset}] {const}={val}: crit={row['krr_rmse_crit_median']:.3f} "
                  f"all={row['krr_rmse_all_median']:.3f} rare={row['rare_captured_median']:.0f} "
                  f"({row['elapsed_sec']:.0f}s)")
        out["sweeps"][const] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dataset", choices=["reactor", "superconduct"], default="reactor")
    ap.add_argument("--output", default="bench_ablation_results.json")
    ap.add_argument("--only", default="", help="comma-separated constant names")
    args = ap.parse_args()

    global GRIDS
    if args.only:
        keep = {k.strip() for k in args.only.split(",")}
        GRIDS = {k: v for k, v in GRIDS.items() if k in keep}

    res = run(args.dataset, list(range(args.seeds)), m=300)
    path = args.output
    merged = {}
    try:
        with open(path, encoding="utf-8") as f:
            merged = json.load(f)
    except FileNotFoundError:
        pass
    merged[res["dataset"]] = res
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()

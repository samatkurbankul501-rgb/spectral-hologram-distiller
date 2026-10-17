"""PM2.5 tuning sweep: tail trigger and phase-band width.

The default SHD (tail_quantile=84, high_band_ratio=0.26) over-allocates to a
broad high-y band; PM2.5's critical regime is only the top 3%. This sweep tests
narrower triggers/bands, with and without the residual term, before any
adaptive-band code change.

Usage:
    python bench_pm25_sweep.py --seeds 5
Writes bench_pm25_sweep_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import (
    build_kt,
    build_kmeans,
    build_random,
    build_skt,
    eval_models,
)
from shd import SpectralHologramDistiller

CONFIGS = {
    "base_t84_h26": dict(tail_quantile=84.0, high_band_ratio=0.26),
    "t90_h10": dict(tail_quantile=90.0, high_band_ratio=0.10),
    "t94_h10": dict(tail_quantile=94.0, high_band_ratio=0.10),
    "t94_h05": dict(tail_quantile=94.0, high_band_ratio=0.05),
    "t97_h05": dict(tail_quantile=97.0, high_band_ratio=0.05),
    "t97_h03": dict(tail_quantile=97.0, high_band_ratio=0.03),
    "t97_h05_nores": dict(tail_quantile=97.0, high_band_ratio=0.05, use_residual=False),
}


def build_shd_custom(X_tr, y_tr, seed, m, batch, crit, params):
    d = SpectralHologramDistiller(
        crystal_size=m, sketch_dim=256, random_state=seed, **params
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    n_rare = int(np.sum(d.crystal_Y >= crit))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare


def run(seeds, m=300, batch=2500, model_seed=42):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    out = {"dataset": "pm25", "title": title, "n_total": int(len(X)),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "n_seeds": len(seeds), "configs": {}, "baselines": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()

    # Baselines are config-independent: compute once per seed.
    for seed in seeds:
        rec = {}
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
        out["baselines"][str(seed)] = rec
        print(f"baselines seed {seed} done ({time.perf_counter()-t0:.0f}s)", flush=True)

    for name, params in CONFIGS.items():
        recs = []
        for seed in seeds:
            Xc, yc, wc, n_rare = build_shd_custom(X_tr, y_tr, seed, m, batch, crit, params)
            met = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                              X_te_crit, Z_te_crit, y_te_crit, gamma)
            met["rare_captured"] = n_rare
            recs.append(met)
        out["configs"][name] = {
            "params": params,
            "records": recs,
            "krr_crit_median": float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_all_median": float(np.median([r["krr_rmse_all"] for r in recs])),
            "gbm_crit_median": float(np.median([r["gbm_rmse_crit"] for r in recs])),
            "gbm_all_median": float(np.median([r["gbm_rmse_all"] for r in recs])),
            "rare_median": float(np.median([r["rare_captured"] for r in recs])),
        }
        print(f"{name}: crit={out['configs'][name]['krr_crit_median']:.2f} "
              f"all={out['configs'][name]['krr_all_median']:.2f} "
              f"rare={out['configs'][name]['rare_median']:.0f}", flush=True)
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="bench_pm25_sweep_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()

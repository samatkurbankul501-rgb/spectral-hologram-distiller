"""Dual-scale RFF ablation.

The paper claims a dual-scale Bochner sketch as a contribution but never
ablates it. This script compares the published dual-scale projector against
single-scale projectors (both halves forced to the low or high bandwidth) on
the 4D reactor, 10 seeds.

Usage:
    python bench_rff.py --seeds 10
Writes bench_rff_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import eval_models, load_reactor
from shd import SpectralHologramDistiller

CONFIGS = {
    "dual_default": dict(omega_low_scale=0.75, omega_high_scale=2.10),
    "single_low": dict(omega_low_scale=0.75, omega_high_scale=0.75),
    "single_high": dict(omega_low_scale=2.10, omega_high_scale=2.10),
}


def run(seeds, m=300, batch=2500):
    X_raw, y_raw, crit, title = load_reactor()
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_raw.shape[1] * 1.1)

    out = {"dataset": "reactor", "title": title, "n_seeds": len(seeds),
           "crystal_size": m, "configs": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for name, params in CONFIGS.items():
        recs = []
        for seed in seeds:
            d = SpectralHologramDistiller(
                crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
                tail_quantile=84.0, phase_kappa=3.2, random_state=seed, **params,
            )
            for i in range(0, len(X_tr), batch):
                d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
            d.finalize_crystal()
            met = eval_models(d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64),
                              d.gamma_weights, scaler, 42, X_te, Z_te, y_te,
                              X_te_crit, Z_te_crit, y_te_crit, gamma)
            met["rare_captured"] = int(np.sum(d.crystal_Y >= crit))
            recs.append(met)
        out["configs"][name] = {
            "params": params,
            "records": recs,
            "krr_rmse_crit_median": float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_rmse_crit_iqr": float(np.percentile([r["krr_rmse_crit"] for r in recs], 75)
                                        - np.percentile([r["krr_rmse_crit"] for r in recs], 25)),
            "krr_rmse_all_median": float(np.median([r["krr_rmse_all"] for r in recs])),
            "rare_captured_median": float(np.median([r["rare_captured"] for r in recs])),
        }
        print(f"{name}: crit={out['configs'][name]['krr_rmse_crit_median']:.3f} "
              f"all={out['configs'][name]['krr_rmse_all_median']:.3f}", flush=True)
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--output", default="bench_rff_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()

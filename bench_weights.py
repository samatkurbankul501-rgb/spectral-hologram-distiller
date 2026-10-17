"""Uniform-vs-gamma weight ablation for SHD crystals.

The paper claims that FISTA herding weights are part of the contribution.  This
benchmark isolates the *selection* from the *weighting*: every seed builds one
SHD crystal, then the same crystal is evaluated twice -- once with the crystal's
``gamma_weights`` and once with uniform weights.  A paired Wilcoxon test
compares the two weightings on identical medoids.

Usage:
    python bench_weights.py --seeds 20
    python bench_weights.py --datasets reactor --seeds 20
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import build_shd, eval_models, load_reactor, load_superconduct

METRICS = ("krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2")


def run_dataset(key: str, seeds: list[int], m: int, batch: int, model_seed: int) -> dict:
    t0 = time.perf_counter()
    if key == "superconduct":
        X_raw, y_raw, crit_threshold, title = load_superconduct()
    else:
        X_raw, y_raw, crit_threshold, title = load_reactor()

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit_threshold
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]

    scaler = StandardScaler().fit(X_tr)
    Z_te = scaler.transform(X_te)
    Z_te_crit = scaler.transform(X_te_crit)
    rbf_gamma = 1.0 / (X_raw.shape[1] * 1.1)

    records = []
    out_path = f"bench_weights_{key}_results.json"
    for seed in seeds:
        ts = time.perf_counter()
        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit_threshold)
        gamma = eval_models(
            Xc, yc, wc, scaler, model_seed,
            X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma,
        )
        uniform = eval_models(
            Xc, yc, np.ones_like(wc), scaler, model_seed,
            X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma,
        )
        records.append(
            {
                "seed": int(seed),
                "rare_captured": int(n_rare),
                "gamma": gamma,
                "uniform": uniform,
            }
        )
        print(
            f"[{key}] seed {seed:2d}: KRR crit gamma={gamma['krr_rmse_crit']:.2f} "
            f"uniform={uniform['krr_rmse_crit']:.2f} | GBM crit "
            f"{gamma['gbm_rmse_crit']:.2f} vs {uniform['gbm_rmse_crit']:.2f} "
            f"({time.perf_counter() - ts:.1f}s)",
            flush=True,
        )
        # Checkpoint after every seed so an interrupted long run is not lost.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": key, "title": title, "crystal_size": m,
                       "n_seeds": len(records), "records": records}, f, indent=1)

    summary = {}
    wilcoxon_tests = {}
    for metric in METRICS:
        a = np.array([r["uniform"][metric] for r in records], dtype=np.float64)
        b = np.array([r["gamma"][metric] for r in records], dtype=np.float64)
        diff = a - b
        stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
        summary[metric] = {
            "gamma_median": float(np.median(b)),
            "gamma_iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
            "uniform_median": float(np.median(a)),
            "uniform_iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        }
        wilcoxon_tests[f"uniform_vs_gamma_{metric}"] = {
            "wins_uniform": int(np.sum(diff > 0)) if metric.endswith("_r2") else int(np.sum(diff < 0)),
            "wins_gamma": int(np.sum(diff < 0)) if metric.endswith("_r2") else int(np.sum(diff > 0)),
            "statistic": float(stat),
            "p_value": float(p),
        }

    result = {
        "dataset": key,
        "title": title,
        "crystal_size": m,
        "n_seeds": len(records),
        "crit_threshold": float(crit_threshold),
        "records": records,
        "summary": summary,
        "wilcoxon": wilcoxon_tests,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"[{key}] saved -> {out_path} ({result['elapsed_sec']:.1f}s)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="superconduct,reactor")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--crystal-size", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--model-seed", type=int, default=42)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    for key in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        run_dataset(key, seeds, args.crystal_size, args.batch, args.model_seed)


if __name__ == "__main__":
    main()

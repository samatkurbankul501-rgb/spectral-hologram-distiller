"""PM2.5 band/tail rehabilitation sweep.

The published PM2.5 result (``bench_pm25_results.json``, 20 seeds) is negative:
at the default ``high_band_ratio=0.26`` / ``tail_quantile=84`` SHD ties the
baselines on the critical regime (13/20 vs KT, p=0.22) and loses significantly
on global KRR/GBM. The methodological audit raised two hypotheses:

1. the 26% phase-peak band is too wide for a 3% regime -> test ``br=0.10``;
2. the band targets the top-16% tail while the critical regime is the top 3%
   -> align ``tail_quantile=97`` with the actual regime percentile.

This benchmark runs both knobs and their combination against KMeans at the
same M=300 budget, time-ordered 80/20 split identical to ``bench_pm25.py``.
The default arm reproduces the published SHD rows bit-exactly (parity control).

Usage:
    python bench_pm25_rehab.py --seeds 20
Writes bench_pm25_rehab_v2_results.json (default). The historical
bench_pm25_rehab_results.json is never overwritten by default.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

from bench_seeds import build_kmeans, eval_models
from bench_pm25 import load_pm25
from shd import SpectralHologramDistiller

ARMS = [
    ("shd_default", 0.26, 84.0),   # published configuration (parity control)
    ("shd_br010", 0.10, 84.0),     # narrower band
    ("shd_tq97", 0.26, 97.0),      # band aligned with the 3% regime
    ("shd_br010_tq97", 0.10, 97.0),
]


def build_shd_variant(X_tr, y_tr, seed, m, batch, crit, br, tq):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=br, sketch_dim=256,
        tail_quantile=tq, phase_kappa=3.2, random_state=seed,
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
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)
    print(f"{title}: n={len(X):,} train={len(X_tr):,} test={len(X_te):,} "
          f"crit_thr={crit:.1f} test_crit={len(X_te_crit):,}", flush=True)

    out = {"dataset": "pm25_rehab", "title": title, "n_total": int(len(X)),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "n_features": int(X.shape[1]), "n_seeds": len(seeds),
           "m": int(m), "batch": int(batch), "model_seed": int(model_seed),
           "records": [], "wilcoxon": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        for arm, br, tq in ARMS:
            Xc, yc, wc, n_rare = build_shd_variant(X_tr, y_tr, seed, m, batch, crit, br, tq)
            rec[arm] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                   X_te_crit, Z_te_crit, y_te_crit, gamma)
            rec[arm]["rare_captured"] = n_rare
        Xc, yc, wc = build_kmeans(X_tr, y_tr, Z_tr, seed, m)
        rec["kmeans"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                    X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["kmeans"]["rare_captured"] = int(np.sum(yc >= crit))
        out["records"].append(rec)
        print(f"seed {seed:2d}: " + " ".join(
            f"{arm}={rec[arm]['krr_rmse_crit']:.1f}" for arm, _, _ in ARMS)
            + f" km={rec['kmeans']['krr_rmse_crit']:.1f}", flush=True)

    for arm, _, _ in ARMS:
        for metric in ("krr_rmse_all", "krr_rmse_crit", "krr_r2",
                       "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"):
            higher = metric.endswith("_r2")
            a = np.array([r[arm][metric] for r in out["records"]])
            b = np.array([r["kmeans"][metric] for r in out["records"]])
            diff = a - b
            stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
            wins = int(np.sum(diff > 0)) if higher else int(np.sum(diff < 0))
            out["wilcoxon"][f"{arm}_vs_kmeans_{metric}"] = {
                "median_shd": float(np.median(a)), "median_kmeans": float(np.median(b)),
                "wins_shd": wins, "statistic": float(stat), "p_value": float(p),
            }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--output", default="bench_pm25_rehab_v2_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for arm, _, _ in ARMS:
        k = f"{arm}_vs_kmeans_krr_rmse_crit"
        v = res["wilcoxon"][k]
        g = res["wilcoxon"][f"{arm}_vs_kmeans_gbm_rmse_all"]
        print(f"  {arm:16s} crit: {v['median_shd']:.1f} vs {v['median_kmeans']:.1f} "
              f"wins={v['wins_shd']}/{len(res['records'])} p={v['p_value']:.3g} | "
              f"gbm_all: {g['median_shd']:.1f} vs {g['median_kmeans']:.1f} "
              f"wins={g['wins_shd']}/{len(res['records'])}")


if __name__ == "__main__":
    main()

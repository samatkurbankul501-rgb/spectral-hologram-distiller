"""Does the SHIPPED global guard earn its place on Beijing PM2.5?

Why this benchmark exists (v3.15.1):

* ``pm25_global`` / ``sparse_auto`` ship ``global_guard=True``, but NO committed
  artifact measured the shipped code path. The only PM2.5 "guard wins"
  numbers came from ``bench_critical_guard.py``, which implements a MANUAL
  K-Means concatenation -- a different mechanism from
  ``SpectralHologramDistiller.build_global_guard`` (which swaps the
  lowest-weight crystal rows in place and damps the donor weights by the
  residual-aware factor).
* Worse: with the shipped default ``lattice_rebuild_interval=0`` the guard
  had no donor rows at all and ``save_hologram()`` RAISED, so the presets
  could not even be saved. That is fixed in v3.15.1 (the guard now owns a
  uniform donor reservoir fed on every batch).

What this benchmark measures (PM2.5, time-ordered 80/20 split, same protocol
as ``bench_pm25.py``):

* ``shd_default``  -- SHD, no guard (the published negative result);
* ``shd_guarded``  -- the SAME distiller, saved through the SHIPPED path
  (``global_guard=True`` -> ``save_hologram()`` -> the guard runs inside
  ``save_hologram``). This is the code path the presets actually use;
* ``kmeans``       -- equal-budget K-Means, the global-accuracy reference.

Both SHD arms use the same M=300 budget and the same seed, so the guard is
the ONLY difference between them -- a paired comparison.

Usage:
    python bench_global_guard.py --seeds 5
Writes bench_global_guard_results.json (default).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kmeans, eval_models
from shd import SpectralHologramDistiller


def build_shd_guarded(X_tr, y_tr, seed, m, batch, n_global, weight):
    """Distil, then save through the SHIPPED guard path.

    ``save_hologram()`` runs the guard when ``global_guard=True`` (the
    documented behaviour), so this exercises exactly what a user of the
    ``pm25_global`` preset executes -- no manual donor logic.
    """
    d = SpectralHologramDistiller(
        crystal_size=m, sketch_dim=256, tail_quantile=97.0,
        phase_kappa=3.2, random_state=seed,
        global_guard=True, global_guard_n=n_global, global_guard_weight=weight,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "crystal.hologram.npz")
        d.save_hologram(path)
        from shd import HologramCrystalLoader
        ld = HologramCrystalLoader(path)
        Xc, yc, wc = ld.get_training_data()
    return np.asarray(Xc), np.asarray(yc), np.asarray(wc), d._guard_report


def build_shd_plain(X_tr, y_tr, seed, m, batch, tail_q):
    """SHD without the guard -- the published negative-result arm."""
    d = SpectralHologramDistiller(
        crystal_size=m, sketch_dim=256, tail_quantile=tail_q,
        phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    return (np.asarray(d.crystal_X), np.asarray(d.crystal_Y),
            np.asarray(d.gamma_weights))


def run(seeds, m=300, batch=2500, model_seed=42, n_global=150, weight=0.35,
        tail_q=97.0):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_tr.shape[1] * 1.1)

    out = {"dataset": "pm25", "title": title, "n_total": int(len(X)),
           "n_train": int(n_tr), "n_test": int(len(X_te)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "n_features": int(X_tr.shape[1]), "m": m, "batch": batch,
           "model_seed": model_seed, "n_seeds": len(seeds),
           "guard": {"n_global": n_global, "weight": weight,
                     "path": "shipped: save_hologram() with global_guard=True"},
           "tail_quantile": tail_q, "records": [], "summary": {},
           "wilcoxon": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        Xc, yc, wc = build_shd_plain(X_tr, y_tr, seed, m, batch, tail_q)
        rec["shd_plain"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te,
                                       y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_plain"]["rare_captured"] = int(np.sum(yc >= crit))
        Xg, yg, wg, report = build_shd_guarded(X_tr, y_tr, seed, m, batch,
                                              n_global, weight)
        rec["shd_guarded"] = eval_models(Xg, yg, wg, scaler, model_seed, X_te,
                                         Z_te, y_te, X_te_crit, Z_te_crit,
                                         y_te_crit, gamma)
        rec["shd_guarded"]["rare_captured"] = int(np.sum(yg >= crit))
        rec["guard_report"] = report
        Xk, yk, wk = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m)
        rec["kmeans"] = eval_models(Xk, yk, wk, scaler, model_seed, X_te, Z_te,
                                    y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["kmeans"]["rare_captured"] = int(np.sum(yk >= crit))
        out["records"].append(rec)
        print(f"seed {seed}: plain all={rec['shd_plain']['krr_rmse_all']:.2f} "
              f"guarded all={rec['shd_guarded']['krr_rmse_all']:.2f} "
              f"kmeans all={rec['kmeans']['krr_rmse_all']:.2f}", flush=True)

    def med(arm, key):
        return float(np.median([r[arm][key] for r in out["records"]]))

    for arm in ("shd_plain", "shd_guarded", "kmeans"):
        out["summary"][arm] = {
            "krr_rmse_all": med(arm, "krr_rmse_all"),
            "krr_rmse_crit": med(arm, "krr_rmse_crit"),
            "krr_r2": med(arm, "krr_r2"),
            "gbm_rmse_all": med(arm, "gbm_rmse_all"),
            "gbm_rmse_crit": med(arm, "gbm_rmse_crit"),
            "rare_captured_median": float(np.median([r[arm]["rare_captured"]
                                                       for r in out["records"]])),
        }

    for metric in ("krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all"):
        a = np.array([r["shd_guarded"][metric] for r in out["records"]])
        b = np.array([r["shd_plain"][metric] for r in out["records"]])
        wins = int(np.sum(a < b))
        try:
            p = float(wilcoxon(a, b).pvalue)
        except Exception:
            p = float("nan")
        out["wilcoxon"][f"guarded_vs_plain_{metric}"] = {
            "median_guarded": float(np.median(a)),
            "median_plain": float(np.median(b)),
            "wins_guarded": wins, "n": len(a), "p_value": p}
        c = np.array([r["kmeans"][metric] for r in out["records"]])
        wins_k = int(np.sum(a < c))
        try:
            p_k = float(wilcoxon(a, c).pvalue)
        except Exception:
            p_k = float("nan")
        out["wilcoxon"][f"guarded_vs_kmeans_{metric}"] = {
            "median_guarded": float(np.median(a)),
            "median_kmeans": float(np.median(c)),
            "wins_guarded": wins_k, "n": len(a), "p_value": p_k}

    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="bench_global_guard_results.json")
    ap.add_argument("--n-global", type=int, default=150)
    ap.add_argument("--weight", type=float, default=0.35)
    args = ap.parse_args()
    res = run(list(range(args.seeds)), n_global=args.n_global, weight=args.weight)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
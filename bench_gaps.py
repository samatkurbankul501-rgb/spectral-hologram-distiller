"""Methodological-gap pilot for SHD.

Closes the two decisive confounds in the 20-seed UCI Superconductivity table:

1. ``shd_sub4800`` runs SHD on exactly the same 4,800-point i.i.d. subsample
   that Kernel Thinning / Supervised KT consume, so the KT comparison is no
   longer confounded by data access.
2. ``stratified`` and ``stratified_ht`` are simple tail-stratified baselines:
   26% of the budget from the top-26% y band and 74% random background.
   ``stratified`` uses uniform weights (rare-focused); ``stratified_ht`` uses
   Horvitz-Thompson stratum weights so the coreset represents the full stream.

Also supports multiple train/test split seeds (the published 20 seeds vary only
the coreset, not the data split) and reports Holm-adjusted p-values.

Usage:
    python bench_gaps.py --splits 42 --seeds 1
    python bench_gaps.py --splits 42,43,44 --seeds 5
Writes bench_gaps_results.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import (
    _kt_subsample_and_halvings,
    build_kt,
    build_kmeans,
    build_random,
    build_shd,
    build_skt,
    eval_models,
    load_superconduct,
)


def kt_sub_idx(n_train: int, m: int, seed: int) -> np.ndarray:
    """Reproduce exactly the subsample drawn inside ``build_kt``."""
    rng = np.random.default_rng(seed)
    n_sub, _h = _kt_subsample_and_halvings(n_train, m)
    return rng.choice(n_train, size=n_sub, replace=False)


def build_stratified(X_tr, y_tr, seed, m, tail_frac=0.26, weighted=False):
    """26% top-y band + random background, with optional HT weights."""
    rng = np.random.default_rng(seed + 100_000)
    q = float(np.quantile(y_tr, 1.0 - tail_frac))
    tail = np.flatnonzero(y_tr >= q)
    bg = np.flatnonzero(y_tr < q)
    n_tail = max(1, int(round(m * tail_frac)))
    n_tail = min(n_tail, len(tail))
    n_bg = m - n_tail
    if n_bg > len(bg):
        n_bg = len(bg)
        n_tail = m - n_bg
    idx_tail = rng.choice(tail, size=n_tail, replace=False)
    idx_bg = rng.choice(bg, size=n_bg, replace=False)
    idx = np.concatenate([idx_tail, idx_bg])
    if not weighted:
        return X_tr[idx], y_tr[idx], None
    w = np.empty(m, dtype=np.float64)
    w[:n_tail] = len(tail) / max(n_tail, 1)
    w[n_tail:] = len(bg) / max(n_bg, 1)
    w /= np.mean(w)
    return X_tr[idx], y_tr[idx], w


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def summarise(records, split_seed=None):
    """Median/IQR + paired Wilcoxon tests over ``records``.

    Independence caveat: a Wilcoxon signed-rank test assumes independent
    pairs. When ``records`` pools several train/test splits (the pooled
    summary), the coreset seeds WITHIN a split share the same test set, so
    those pairs are correlated and the pooled p-value is anti-conservative
    (pseudoreplication). The per-split summaries are the independent-unit
    inference; with 5 seeds the signed-rank p floor is 2/2**5 = 0.0625, so a
    single split cannot reach p < 0.05. Only ``shd_vs_<pair>`` tests are
    computed here (SHD_sub4800 is compared against full SHD, not against the
    baselines).
    """
    arms = [k for k in records[0] if k not in ("seed", "split_seed")]
    out = {"n_records": len(records), "arms": arms, "medians": {}}
    for arm in arms:
        out["medians"][arm] = {}
        for metric in ("krr_rmse_all", "krr_rmse_crit", "krr_r2",
                       "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2", "rare_captured"):
            vals = np.array([r[arm][metric] for r in records], dtype=np.float64)
            out["medians"][arm][metric] = {
                "median": float(np.median(vals)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            }
    pairs = [p for p in ("random", "kmeans", "kt", "skt", "stratified",
                         "stratified_ht", "shd_sub4800") if p in arms]
    tests = {}
    for pair in pairs:
        for metric in ("krr_rmse_all", "krr_rmse_crit", "krr_r2",
                       "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"):
            higher = metric.endswith("_r2")
            a = np.array([r["shd"][metric] for r in records])
            b = np.array([r[pair][metric] for r in records])
            diff = a - b
            stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
            wins = int(np.sum(diff > 0)) if higher else int(np.sum(diff < 0))
            tests[f"shd_vs_{pair}_{metric}"] = {
                "median_shd": float(np.median(a)),
                "median_" + pair: float(np.median(b)),
                "wins_shd": wins,
                "statistic": float(stat),
                "p_value": float(p),
            }
    # Holm across the whole paired-test family reported here.
    keys = list(tests)
    adjusted = holm([tests[k]["p_value"] for k in keys])
    for k, p_adj in zip(keys, adjusted):
        tests[k]["p_holm"] = float(p_adj)
    out["tests"] = tests
    return out


def run(split_seeds, seeds, m=300, batch=2500, model_seed=42):
    X_raw, y_raw, crit, title = load_superconduct()
    if X_raw.shape[1] != 81:
        raise RuntimeError("UCI Superconductivity fetch failed (fallback dataset); aborting")
    out = {
        "dataset": "superconduct",
        "title": title,
        "crystal_size": m,
        "split_seeds": list(split_seeds),
        "seeds": list(seeds),
        "splits": {},
    }
    pooled = []
    t_all = time.perf_counter()
    for split_seed in split_seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y_raw, test_size=0.20, random_state=split_seed
        )
        crit_mask = y_te >= crit
        X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
        scaler = StandardScaler().fit(X_tr)
        Z_te = scaler.transform(X_te)
        Z_te_crit = scaler.transform(X_te_crit)
        Z_tr = scaler.transform(X_tr)
        rbf_gamma = 1.0 / (X_raw.shape[1] * 1.1)
        print(f"[split {split_seed}] train={len(X_tr):,} test={len(X_te):,} "
              f"crit={len(X_te_crit):,}", flush=True)

        recs = []
        for seed in seeds:
            rec = {"split_seed": int(split_seed), "seed": int(seed)}
            idx = kt_sub_idx(len(X_tr), m, seed)
            sub_X, sub_y = X_tr[idx], y_tr[idx]
            print(f"  seed {seed}: subsample={len(idx)}", flush=True)

            Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
            rec["shd"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                     X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)
            rec["shd"]["rare_captured"] = n_rare

            Xc, yc, wc, n_rare = build_shd(sub_X, sub_y, seed, m, batch, crit)
            rec["shd_sub4800"] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                             X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)
            rec["shd_sub4800"]["rare_captured"] = n_rare

            for arm, builder in (
                ("random", lambda: build_random(X_tr, y_tr, seed, m)),
                ("kmeans", lambda: build_kmeans(X_tr, y_tr, Z_tr, seed, m)),
                ("kt", lambda: build_kt(X_tr, y_tr, Z_tr, seed, m)),
                ("skt", lambda: build_skt(X_tr, y_tr, Z_tr, seed, m)),
                ("stratified", lambda: build_stratified(X_tr, y_tr, seed, m)),
                ("stratified_ht", lambda: build_stratified(X_tr, y_tr, seed, m, weighted=True)),
            ):
                Xc, yc, wc = builder()
                rec[arm] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                       X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)
                rec[arm]["rare_captured"] = int(np.sum(yc >= crit))
            recs.append(rec)
            pooled.append(rec)
            print(f"    shd={rec['shd']['krr_rmse_crit']:.2f} "
                  f"sub={rec['shd_sub4800']['krr_rmse_crit']:.2f} "
                  f"kt={rec['kt']['krr_rmse_crit']:.2f} "
                  f"skt={rec['skt']['krr_rmse_crit']:.2f} "
                  f"strat={rec['stratified']['krr_rmse_crit']:.2f} "
                  f"strat_ht={rec['stratified_ht']['krr_rmse_crit']:.2f}", flush=True)

        out["splits"][str(split_seed)] = {
            "n_train": int(len(X_tr)),
            "n_test_crit": int(len(X_te_crit)),
            "records": recs,
            "summary": summarise(recs, split_seed),
        }
        print(f"[split {split_seed}] done ({time.perf_counter() - t_all:.0f}s total)", flush=True)

    out["pooled_summary"] = summarise(pooled)
    out["elapsed_sec"] = time.perf_counter() - t_all
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="42", help="comma-separated split seeds")
    ap.add_argument("--seeds", type=int, default=1, help="number of coreset seeds 0..n-1")
    ap.add_argument("--output", default="bench_gaps_results.json")
    args = ap.parse_args()
    split_seeds = [int(s.strip()) for s in args.splits.split(",") if s.strip()]
    res = run(split_seeds, list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    ps = res["pooled_summary"]["tests"]
    for key in ("shd_vs_kt_krr_rmse_crit", "shd_vs_skt_krr_rmse_crit",
                "shd_vs_stratified_krr_rmse_crit", "shd_vs_stratified_ht_krr_rmse_crit"):
        if key in ps:
            print(f"  {key}: p={ps[key]['p_value']:.3g} p_holm={ps[key]['p_holm']:.3g} "
                  f"wins={ps[key]['wins_shd']}/{res['pooled_summary']['n_records']}")


if __name__ == "__main__":
    main()

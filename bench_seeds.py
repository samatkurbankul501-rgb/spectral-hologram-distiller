"""Multi-seed statistical validation for SHD v3.7.

Design (honest and cheap):
- Train/test split is FIXED (random_state=42, same as the published single-seed
  numbers). The Oracle baseline therefore does not depend on the seed and is
  fitted once.
- Only the coreset-construction randomness varies per seed: the SHD projector
  and lattice seed, the Random selection, and the K-Means seed.
- For each seed we record KRR and GBM metrics (all + rare regime), then run
  paired Wilcoxon signed-rank tests: SHD vs Random, SHD vs K-Means.

Results are dumped to bench_seeds_results.json — the single source of truth.

Usage:
    python bench_seeds.py [--seeds 20] [--dataset superconduct|reactor|both]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shd import SpectralHologramDistiller


# --- datasets ----------------------------------------------------------------


def load_reactor(n=20000, seed=42):
    """Synthetic 4D stream with a rare phase-transition regime (hot band)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    crit_threshold = float(np.percentile(y, 97.0))  # rare-regime cut on y
    return X, y, crit_threshold, "Reactor 4D (phase transition)"


def load_superconduct():
    """UCI Superconductivity (21,263 x 81). Falls back to California housing."""
    from sklearn.datasets import fetch_california_housing, fetch_openml

    try:
        try:
            ds = fetch_openml(name="superconduct", version=1, as_frame=False, parser="auto")
        except ImportError:
            # parser='auto' needs pandas for dense data; liac-arff does not.
            ds = fetch_openml(name="superconduct", version=1, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=np.float64)
        y = np.asarray(ds.target, dtype=np.float64)
        if X.shape[1] == 81 and len(X) > 20000:
            return X, y, 77.0, "UCI Superconductivity"
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[warn] OpenML fetch failed ({e}); falling back to California housing")
    ds = fetch_california_housing()
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(ds.target, dtype=np.float64)
    return X, y, float(np.percentile(y, 97.0)), "California housing (fallback)"


# --- coreset builders ---------------------------------------------------------


def build_shd(X_tr, y_tr, seed, m, batch, crit_threshold):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
    d.finalize_crystal()
    n_rare = int(np.sum(d.crystal_Y >= crit_threshold))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare


def build_random(X_tr, y_tr, seed, m):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_tr), size=m, replace=False)
    return X_tr[idx], y_tr[idx], None


def build_kmeans(X_tr, y_tr, Z_tr, seed, m):
    km = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=seed, n_init=3).fit(Z_tr)
    counts = np.bincount(km.labels_, minlength=m).astype(np.float64) + 1.0
    w = counts / np.mean(counts)
    idx = np.argmin(cdist(km.cluster_centers_, Z_tr, metric="sqeuclidean"), axis=1)
    return X_tr[idx], y_tr[idx], w


# --- model evaluation ----------------------------------------------------------


def eval_models(X_c, y_c, w_c, scaler, model_seed, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma):
    Z_c = scaler.transform(X_c)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=rbf_gamma)
    krr.fit(Z_c, y_c, sample_weight=w_c)
    pred_all = krr.predict(Z_te)
    pred_crit = krr.predict(Z_te_crit)
    krr_row = (
        float(np.sqrt(mean_squared_error(y_te, pred_all))),
        float(np.sqrt(mean_squared_error(y_te_crit, pred_crit))),
        float(r2_score(y_te, pred_all)),
    )

    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=model_seed
    )
    gbm.fit(X_c, y_c, sample_weight=w_c)
    g_all = gbm.predict(X_te)
    g_crit = gbm.predict(X_te_crit)
    gbm_row = (
        float(np.sqrt(mean_squared_error(y_te, g_all))),
        float(np.sqrt(mean_squared_error(y_te_crit, g_crit))),
        float(r2_score(y_te, g_all)),
    )
    return {
        "krr_rmse_all": krr_row[0], "krr_rmse_crit": krr_row[1], "krr_r2": krr_row[2],
        "gbm_rmse_all": gbm_row[0], "gbm_rmse_crit": gbm_row[1], "gbm_r2": gbm_row[2],
    }


# --- per-dataset experiment -----------------------------------------------------


def run_dataset(key, seeds, m, batch, model_seed):
    t0 = time.perf_counter()
    if key == "superconduct":
        X_raw, y_raw, crit_threshold, title = load_superconduct()
    else:
        X_raw, y_raw, crit_threshold, title = load_reactor()

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit_threshold
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]

    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te = scaler.transform(X_te)
    Z_te_crit = scaler.transform(X_te_crit)

    d_feat = X_raw.shape[1]
    rbf_gamma = 1.0 / (d_feat * 1.1)

    # Oracle: seed-independent, fit once.
    oracle = {"krr": None, "gbm": None}
    n_oracle = min(9000, len(X_tr))
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=rbf_gamma)
    krr.fit(scaler.transform(X_tr[:n_oracle]), y_tr[:n_oracle])
    oracle["krr"] = (
        float(np.sqrt(mean_squared_error(y_te, krr.predict(Z_te)))),
        float(np.sqrt(mean_squared_error(y_te_crit, krr.predict(Z_te_crit)))),
        float(r2_score(y_te, krr.predict(Z_te))),
    )
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=model_seed
    )
    gbm.fit(X_tr, y_tr)
    oracle["gbm"] = (
        float(np.sqrt(mean_squared_error(y_te, gbm.predict(X_te)))),
        float(np.sqrt(mean_squared_error(y_te_crit, gbm.predict(X_te_crit)))),
        float(r2_score(y_te, gbm.predict(X_te))),
    )
    print(f"[{key}] {title}: n_train={len(X_tr):,}, test_crit={len(X_te_crit):,}, oracle done "
          f"({time.perf_counter() - t0:.1f}s)")

    records = []
    for seed in seeds:
        t_s = time.perf_counter()
        rec = {"seed": seed}

        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit_threshold)
        rec["shd"] = eval_models(Xc, yc, wc, scaler, model_seed,
                                 X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)
        rec["shd"]["rare_captured"] = n_rare

        Xc, yc, wc = build_random(X_tr, y_tr, seed, m)
        rec["random"] = eval_models(Xc, yc, wc, scaler, model_seed,
                                    X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)

        Xc, yc, wc = build_kmeans(X_tr, y_tr, Z_tr, seed, m)
        rec["kmeans"] = eval_models(Xc, yc, wc, scaler, model_seed,
                                    X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)

        records.append(rec)
        print(f"[{key}] seed {seed:2d}: shd krr_crit={rec['shd']['krr_rmse_crit']:.2f} "
              f"rand={rec['random']['krr_rmse_crit']:.2f} km={rec['kmeans']['krr_rmse_crit']:.2f} "
              f"rare={n_rare} ({time.perf_counter() - t_s:.1f}s)")

    # Paired Wilcoxon tests.
    tests = {}
    for metric in ["krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"]:
        higher_is_better = metric.endswith("_r2")
        for pair in ["random", "kmeans"]:
            a = np.array([r["shd"][metric] for r in records])
            b = np.array([r[pair][metric] for r in records])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(a, b)
            wins = int(np.sum(diff > 0)) if higher_is_better else int(np.sum(diff < 0))
            tests[f"shd_vs_{pair}_{metric}"] = {
                "median_shd": float(np.median(a)),
                "iqr_shd": float(np.percentile(a, 75) - np.percentile(a, 25)),
                "median_" + pair: float(np.median(b)),
                "iqr_" + pair: float(np.percentile(b, 75) - np.percentile(b, 25)),
                "wins_shd": wins,
                "statistic": float(stat),
                "p_value": float(p),
            }

    summary = {
        "dataset": key,
        "title": title,
        "crystal_size": m,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "split_random_state": 42,
        "oracle": {"krr_rmse_all": oracle["krr"][0], "krr_rmse_crit": oracle["krr"][1],
                   "krr_r2": oracle["krr"][2], "gbm_rmse_all": oracle["gbm"][0],
                   "gbm_rmse_crit": oracle["gbm"][1], "gbm_r2": oracle["gbm"][2]},
        "records": records,
        "wilcoxon": tests,
        "elapsed_sec": time.perf_counter() - t0,
    }
    print(f"[{key}] done in {summary['elapsed_sec']:.1f}s")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--dataset", choices=["superconduct", "reactor", "both", "summary"], default="both")
    ap.add_argument("--model-seed", type=int, default=42)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    path = "bench_seeds_results.json"
    out = {"config": {"n_seeds": args.seeds, "model_seed": args.model_seed}, "results": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            prev = json.load(f)
            if isinstance(prev.get("results"), dict):
                out["results"].update(prev["results"])
    except FileNotFoundError:
        pass
    if args.dataset == "summary":
        keys = list(out["results"].keys())
    else:
        keys = ["superconduct", "reactor"] if args.dataset == "both" else [args.dataset]
        for key in keys:
            out["results"][key] = run_dataset(key, seeds, m=300, batch=2500, model_seed=args.model_seed)

    # Recompute Wilcoxon + IQR across all stored datasets to guarantee schema uniformity.
    for key, res in out["results"].items():
        records = res["records"]
        tests = {}
        for metric in ["krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"]:
            higher_is_better = metric.endswith("_r2")
            for pair in ["random", "kmeans"]:
                a = np.array([r["shd"][metric] for r in records])
                b = np.array([r[pair][metric] for r in records])
                diff = a - b
                stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
                wins = int(np.sum(diff > 0)) if higher_is_better else int(np.sum(diff < 0))
                tests[f"shd_vs_{pair}_{metric}"] = {
                    "median_shd": float(np.median(a)),
                    "iqr_shd": float(np.percentile(a, 75) - np.percentile(a, 25)),
                    "median_" + pair: float(np.median(b)),
                    "iqr_" + pair: float(np.percentile(b, 75) - np.percentile(b, 25)),
                    "wins_shd": wins,
                    "statistic": float(stat),
                    "p_value": float(p),
                }
        res["wilcoxon"] = tests

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path}")

    # Compact summary.
    for key in out["results"]:
        r = out["results"][key]
        rares = [rec["shd"]["rare_captured"] for rec in r["records"]]
        print(f"\n=== {key}: {r['title']} (n_seeds={r['n_seeds']}) ===")
        print(f"Oracle KRR: all={r['oracle']['krr_rmse_all']:.2f} crit={r['oracle']['krr_rmse_crit']:.2f} r2={r['oracle']['krr_r2']:.4f}")
        print(f"Oracle GBM: all={r['oracle']['gbm_rmse_all']:.2f} crit={r['oracle']['gbm_rmse_crit']:.2f} r2={r['oracle']['gbm_r2']:.4f}")
        print(f"Rare captured by SHD: median={np.median(rares):.1f} (IQR={np.percentile(rares,75)-np.percentile(rares,25):.1f})")
        for metric in ["krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"]:
            for pair in ["random", "kmeans"]:
                t = r["wilcoxon"][f"shd_vs_{pair}_{metric}"]
                print(
                    f"  {metric:14s} vs {pair:6s}: shd {t['median_shd']:.3f} (IQR {t['iqr_shd']:.3f}) "
                    f"vs {t['median_' + pair]:.3f} (IQR {t['iqr_' + pair]:.3f}) | "
                    f"wins {t['wins_shd']:2d}/{r['n_seeds']} | p={t['p_value']:.2e}"
                )


if __name__ == "__main__":
    main()

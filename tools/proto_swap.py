"""PM2.5 swap-grid prototype (2026-10-22).

proto_refine/proto_raps established: refinement in phase space fails
(child cells leak into the band), the residual-augmented phase space (RAPS)
helps both metrics modestly, and swap50 (drop the 50 lowest-weight medoids,
add 50 k-means medoids in RAW standardized space) helps both metrics
strongly. This grid answers:

1. How many slots should the global guard get? swap in {25, 50, 75, 100, 150}.
2. Donor space: k-means on RAW standardized features vs k-means in SHD's
   internal normalized space (the sketch input) -- is the sketch metric
   distortion the root cause of the background deficit?
3. Does RAPS + swap compose?
4. Sanity: does the dropped tail really carry the lowest weights, and does
   the band survive the swap (rare_captured)?

Arms: shd, swap25..swap150, rswap50 (raps+swap50), dsketch50 (donor in SHD
space), kmeans, skt, random.

Usage:
    python tools/proto_swap.py --seeds 5
Writes tools/_proto_swap.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench_pm25 import load_pm25  # noqa: E402
from bench_seeds import (  # noqa: E402
    build_kmeans,
    build_random,
    build_shd,
    build_skt,
    eval_models,
)
from proto_raps import build_raps  # noqa: E402
from shd import SpectralHologramDistiller  # noqa: E402


def build_kmeans_in_shd_space(X_tr, y_tr, seed, m_k):
    """k-means medoids in the distiller's INTERNAL normalized space.

    Difference vs build_kmeans (raw StandardScaler space): the distiller
    normalizes with its own running X_mean/X_std and feeds the RFF sketch;
    if the sketch metric distortion is the root cause of the background
    deficit, this donor should behave worse than the raw-space donor.
    """
    d0 = SpectralHologramDistiller(
        crystal_size=50, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    d0.partial_fit(X_tr[:2500], y_tr[:2500])
    Z_shd = d0._normalize_X(X_tr, update_state=False)
    km = MiniBatchKMeans(n_clusters=m_k, batch_size=2048, random_state=seed, n_init=3).fit(Z_shd)
    counts = np.bincount(km.labels_, minlength=m_k).astype(np.float64) + 1.0
    w = counts / np.mean(counts)
    idx = np.argmin(cdist(km.cluster_centers_, Z_shd, metric="sqeuclidean"), axis=1)
    return X_tr[idx], y_tr[idx], w


def make_swap(Xc, yc, wc, Xg, yg, wg, m_swap):
    order = np.argsort(wc)
    keep = order[m_swap:]
    X_s = np.vstack([Xc[keep], Xg])
    y_s = np.concatenate([yc[keep], yg])
    w_s = np.concatenate([wc[keep], wg])
    return X_s, y_s, w_s / w_s.mean()


def run(seeds, m=300, batch=2500, out_path="tools/_proto_swap.json"):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    arms = ["shd", "swap25", "swap50", "swap75", "swap100", "swap150",
            "rswap50", "dsketch50", "kmeans", "skt", "random"]
    out = {"dataset": "pm25", "m": m, "n_seeds": len(seeds), "arms": arms,
           "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        c = {}
        Xc, yc, wc, _ = build_shd(X_tr, y_tr, seed, m, batch, crit)
        c["shd"] = (Xc, yc, wc)
        Xg_raw, yg_raw, wg_raw = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, 150)
        if wg_raw is None:
            wg_raw = np.ones(len(Xg_raw))
        Xg_sket, yg_sket, wg_sket = build_kmeans_in_shd_space(X_tr, y_tr, seed, 150)
        if wg_sket is None:
            wg_sket = np.ones(len(Xg_sket))
        for k_s in (25, 50, 75, 100, 150):
            c[f"swap{k_s}"] = make_swap(Xc, yc, wc, Xg_raw[:k_s], yg_raw[:k_s],
                                        wg_raw[:k_s], k_s)
        rap = build_raps(X_tr, y_tr, seed, m, batch, crit, lam=1.0)[:3]
        c["rswap50"] = make_swap(rap[0], rap[1], rap[2], Xg_raw[:50], yg_raw[:50],
                                 wg_raw[:50], 50)
        c["dsketch50"] = make_swap(Xc, yc, wc, Xg_sket[:50], yg_sket[:50],
                                   wg_sket[:50], 50)
        Xk, yk, wk = build_kmeans(X_tr, y_tr, Z_tr, seed, m)
        c["kmeans"] = (Xk, yk, wk)
        Xs, ys, ws = build_skt(X_tr, y_tr, Z_tr, seed, m)
        c["skt"] = (Xs, ys, ws)
        Xr, yr, wr = build_random(X_tr, y_tr, seed, m)
        c["random"] = (Xr, yr, wr)

        for arm in arms:
            Xa, ya, wa = c[arm]
            r = eval_models(Xa, ya, wa, scaler, 42, X_te, Z_te, y_te,
                            X_te_crit, Z_te_crit, y_te_crit, gamma)
            r["rare_captured"] = int(np.sum(np.asarray(ya) >= crit))
            rec[arm] = r
        out["records"].append(rec)
        print(f"seed {seed}: " + " ".join(
            f"{arm}={rec[arm]['krr_rmse_all']:.1f}/{rec[arm]['krr_rmse_crit']:.0f}"
            for arm in arms), flush=True)

    print("\nMEDIANS (krr_all / krr_crit / krr_r2 / gbm_all / gbm_crit / rare):")
    for arm in arms:
        f = lambda k: float(np.median([r[arm][k] for r in out["records"]]))  # noqa: E731
        print(f"  {arm:9s} all={f('krr_rmse_all'):7.2f} crit={f('krr_rmse_crit'):7.1f} "
              f"r2={f('krr_r2'):6.3f} gbm_all={f('gbm_rmse_all'):7.2f} "
              f"gbm_crit={f('gbm_rmse_crit'):7.1f} rare={f('rare_captured'):.0f}")
    print("\nWINS vs shd (lower better, r2 higher better):")
    for arm in arms[1:]:
        w = {k: int(np.sum([r[arm][k] < r["shd"][k] for r in out["records"]]))
             for k in ("krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all", "gbm_rmse_crit")}
        wr2 = int(np.sum([r[arm]["krr_r2"] > r["shd"]["krr_r2"] for r in out["records"]]))
        print(f"  {arm:9s} {w} r2_wins={wr2}/{len(seeds)}")

    out["elapsed_sec"] = time.perf_counter() - t0
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved -> {out_path} ({out['elapsed_sec']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="tools/_proto_swap.json")
    args = ap.parse_args()
    run(list(range(args.seeds)), out_path=args.output)


if __name__ == "__main__":
    main()

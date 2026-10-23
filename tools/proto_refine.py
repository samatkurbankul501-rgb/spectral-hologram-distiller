"""PM2.5 refinement prototype (Tony-Stark round, 2026-10-22).

Diagnosis (tools/diag_pm25_weights.py): the PM2.5 global loss is a
GEOMETRY deficit, not a weight deficit -- uniform weights improve all
(94.9 -> 91.7) but still lose to random (88.7), and 26% of the crystal is
reserved for the rare band. The hybrid probe (bench_pm25_hybrid.py) showed
swap50 (drop the 50 lowest-weight medoids, add 50 k-means medoids) helps
BOTH metrics, pointing at a background-resolution deficit.

Mechanism tested here: LATTICE REFINEMENT. After the standard lattice
build, run a few Lloyd iterations on a random sample of rows in the merged
phase space [Z || y_scaled] and split the cells with the largest
count * spread (rich get richer): each split cell donates one medoid to a
new cell seeded at its farthest member. The crystal budget M stays 300 --
quality is redistributed, not added.

Arms (M=300 unless noted):
- shd            published pipeline (h=0.26) -- baseline
- ref10          refine_steps=10, refine_sample=2500, split=50
- ref10_s05      same with y-scale 0.5 (weaker y coordinate in refinement)
- swap50         hybrid reference
- ref_swap       refine + swap50
- skt_joint      supervised kernel-thinning on [Z || y_std] (external
  reference: is the deficit fixable by a fundamentally different sampler?)
- random         i.i.d. reference

Usage:
    python tools/proto_refine.py --seeds 5
Writes tools/_proto_refine.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
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
from shd import SpectralHologramDistiller  # noqa: E402


def build_shd_refined(X_tr, y_tr, seed, m, batch, crit, refine_steps=10,
                      refine_sample=2500, n_split=50, y_scale=1.0,
                      kappa=3.2):
    """Standard SHD lattice + post-hoc refinement of the largest cells.

    The refinement re-runs Lloyd on [Z || y_std * y_scale] over a sample of
    the training rows, then splits the n_split cells with the largest
    member count. Splitting is 'rich get richer': a cell with many members
    donates its medoid to a new cell seeded at the member farthest from the
    (recomputed) centre, so dense background regions get more resolution
    while the crystal size stays M.
    """
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=kappa, random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i:i + batch], y_tr[i:i + batch])
    d.finalize_crystal()
    if refine_steps <= 0:
        n_rare = int(np.sum(np.asarray(d.crystal_Y, dtype=np.float64) >= crit))
        return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare

    # --- refinement -----------------------------------------------------
    rng = np.random.default_rng(seed + 77)
    idx_s = rng.choice(len(X_tr), size=min(refine_sample, len(X_tr)), replace=False)
    Z_s = d._normalize_X(X_tr[idx_s], update_state=False)
    y_s = y_tr[idx_s].astype(np.float64)
    y_scaled = (y_s - d.y_mean) / d.y_std * y_scale
    ZY = np.hstack([Z_s, y_scaled[:, None]])

    centers = d.centers_ZY.copy()
    for _ in range(refine_steps):
        labels = np.argmin(cdist(ZY, centers, metric="sqeuclidean"), axis=1)
        moved = 0.0
        for c_i in range(len(centers)):
            mask = labels == c_i
            if np.any(mask):
                new_c = np.mean(ZY[mask], axis=0)
                moved = max(moved, float(np.sum((new_c - centers[c_i]) ** 2)))
                centers[c_i] = new_c
        if moved < 1e-10:
            break

    labels = np.argmin(cdist(ZY, centers, metric="sqeuclidean"), axis=1)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.float64)
    # cells to split: largest counts, but only cells with >= 2 sampled members
    order = np.argsort(counts)[::-1]
    split = []
    for c_i in order:
        if len(split) >= n_split:
            break
        if counts[c_i] >= 2:
            split.append(int(c_i))
    for c_i in split:
        mask = np.where(labels == c_i)[0]
        if len(mask) < 2:
            continue
        # medoid of the cell = member closest to the centre (this is the
        # medoid the crystal would keep); the FARTHEST member seeds a child
        d_loc = np.sum((ZY[mask] - centers[c_i]) ** 2, axis=1)
        keep_local = int(np.argmin(d_loc))
        far_local = int(np.argmax(d_loc))
        child = ZY[mask[far_local]] + 1e-6 * rng.normal(size=ZY.shape[1])
        centers[c_i] = ZY[mask[keep_local]]
        centers = np.vstack([centers, child])

    # --- crystal from the refined lattice -------------------------------
    # Assign ALL training rows to the refined cells and take real medoids.
    Z_all = d._normalize_X(X_tr, update_state=False)
    y_all = y_tr.astype(np.float64)
    ZY_all = np.hstack([Z_all, ((y_all - d.y_mean) / d.y_std * y_scale)[:, None]])
    labels_all = np.argmin(cdist(ZY_all, centers, metric="sqeuclidean"), axis=1)
    counts_all = np.bincount(labels_all, minlength=len(centers)).astype(np.float64)
    M_eff = len(centers)
    med_X = np.zeros((M_eff, X_tr.shape[1]))
    med_Y = np.zeros(M_eff)
    med_Z = np.zeros((M_eff, Z_all.shape[1]))
    for c_i in range(M_eff):
        mask = np.where(labels_all == c_i)[0]
        if len(mask) == 0:
            # empty cell: nearest sampled row to the centre
            dd = np.sum((ZY_all - centers[c_i]) ** 2, axis=1)
            mask = np.array([int(np.argmin(dd))])
        d_loc = np.sum((ZY_all[mask] - centers[c_i]) ** 2, axis=1)
        med_X[c_i] = X_tr[mask[int(np.argmin(d_loc))]]
        med_Y[c_i] = y_all[mask[int(np.argmin(d_loc))]]
        med_Z[c_i] = Z_all[mask[int(np.argmin(d_loc))]]

    # FISTA weights on the refined crystal (same recipe as finalize_crystal)
    ZY_m = np.hstack([med_Z, ((med_Y - d.y_mean) / d.y_std * y_scale)[:, None]])
    U_m = d._spectral_transform(med_X, update_state=False)[1]
    voronoi_prob = counts_all / counts_all.sum()
    Psi_hat = d.Psi_sum / d.n_seen
    G_U = U_m @ U_m.T
    K = (G_U ** 2) + d.fista_l2_reg * np.eye(M_eff)
    b = np.sum((U_m @ Psi_hat) * U_m, axis=1) + d.fista_l2_reg * voronoi_prob
    L_lip = float(np.linalg.norm(K, ord=2)) + 1e-6
    gamma = voronoi_prob.copy()
    z = gamma.copy()
    t_step = 1.0
    for _ in range(250):
        grad = K @ z - b
        gamma_next = np.maximum(1e-5, z - grad / L_lip)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_step ** 2))
        z = gamma_next + ((t_step - 1.0) / t_next) * (gamma_next - gamma)
        gamma, t_step = gamma_next, t_next
    gamma_prob = gamma / gamma.sum()
    tempered = voronoi_prob ** d.voronoi_temper
    tempered /= tempered.sum()
    w = (d.herding_mix * tempered + (1.0 - d.herding_mix) * gamma_prob) * M_eff
    w = w / w.mean()
    n_rare = int(np.sum(med_Y >= crit))
    return med_X, med_Y, w, n_rare


def swap50(Xc, yc, wc, X_tr, y_tr, Z_tr, seed, m_swap=50):
    Xg, yg, wg = build_kmeans(X_tr, y_tr, Z_tr, seed + 5000, m_swap)
    if wg is None:
        wg = np.ones(len(Xg))
    order = np.argsort(wc)
    keep = order[m_swap:]
    X_s = np.vstack([Xc[keep], Xg])
    y_s = np.concatenate([yc[keep], yg])
    w_s = np.concatenate([wc[keep], wg])
    return X_s, y_s, w_s / w_s.mean()


def run(seeds, m=300, batch=2500, out_path="tools/_proto_refine.json"):
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

    arms = ["shd", "ref10", "ref10_s05", "swap50", "ref_swap", "skt_joint", "random"]
    out = {"dataset": "pm25", "m": m, "n_seeds": len(seeds), "arms": arms,
           "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        c = {}

        Xc, yc, wc, _ = build_shd(X_tr, y_tr, seed, m, batch, crit)
        c["shd"] = (Xc, yc, wc)
        c["ref10"] = build_shd_refined(X_tr, y_tr, seed, m, batch, crit,
                                        refine_steps=10, n_split=50, y_scale=1.0)[:3]
        c["ref10_s05"] = build_shd_refined(X_tr, y_tr, seed, m, batch, crit,
                                           refine_steps=10, n_split=50, y_scale=0.5)[:3]
        c["swap50"] = swap50(Xc, yc, wc, X_tr, y_tr, Z_tr, seed)
        c["ref_swap"] = swap50(*c["ref10"][:3], X_tr, y_tr, Z_tr, seed)
        # skt on the joint [Z || y_std] space via build_skt (it already
        # appends a scaled y column internally)
        Xj, yj, wj = build_skt(X_tr, y_tr, Z_tr, seed, m)
        c["skt_joint"] = (Xj, yj, wj)
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

    print("\nMEDIANS (krr_all / krr_crit / krr_r2 / gbm_all / rare):")
    for arm in arms:
        f = lambda k: float(np.median([r[arm][k] for r in out["records"]]))  # noqa: E731
        print(f"  {arm:10s} all={f('krr_rmse_all'):7.2f} crit={f('krr_rmse_crit'):7.1f} "
              f"r2={f('krr_r2'):6.3f} gbm_all={f('gbm_rmse_all'):7.2f} "
              f"gbm_crit={f('gbm_rmse_crit'):7.1f} rare={f('rare_captured'):.0f}")
    # wins vs shd
    print("\nWINS vs shd (lower better, r2 higher better):")
    for arm in arms[1:]:
        w = {k: int(np.sum([r[arm][k] < r["shd"][k] for r in out["records"]]))
             for k in ("krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all")}
        wr2 = int(np.sum([r[arm]["krr_r2"] > r["shd"]["krr_r2"] for r in out["records"]]))
        print(f"  {arm:10s} {w} r2_wins={wr2}/{len(seeds)}")

    out["elapsed_sec"] = time.perf_counter() - t0
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved -> {out_path} ({out['elapsed_sec']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="tools/_proto_refine.json")
    args = ap.parse_args()
    run(list(range(args.seeds)), out_path=args.output)


if __name__ == "__main__":
    main()

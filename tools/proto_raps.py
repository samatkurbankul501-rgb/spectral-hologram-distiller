"""PM2.5 residual-augmented-phase-space (RAPS) prototype (2026-10-22).

Two-pass geometry: pass 1 is the published SHD pipeline; its KRR residuals
on the TRAIN pool become an extra phase coordinate for pass 2's lattice:

    X_aug = [X || lam * z(residual)]     (features, 14 -> 15 columns)
    y     = real y                        (unchanged)

so the Voronoi lattice allocates resolution where the first model is
wrong, while the crystal still stores real rows with real targets and the
downstream models train on the ORIGINAL 14 features -- the published
single-model evaluation protocol is untouched. This is geometry, not an
ensemble: no test data, no prediction-space blending.

Arms: shd (published), raps10 (lam=1.0), raps05 (lam=0.5), reb15k
(lattice_rebuild_interval=15000 on the published pipeline), kmeans, skt,
random.

Usage:
    python tools/proto_raps.py --seeds 5
Writes tools/_proto_raps.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.kernel_ridge import KernelRidge
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


def build_shd_generic(X, y, seed, m, batch, crit, rebuild_interval=0):
    """build_shd with an optional lattice_rebuild_interval; returns (X, y, w, rare)."""
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
        lattice_rebuild_interval=rebuild_interval,
    )
    for i in range(0, len(X), batch):
        d.partial_fit(X[i:i + batch], y[i:i + batch])
    d.finalize_crystal()
    rare = int(np.sum(np.asarray(d.crystal_Y, dtype=np.float64) >= crit))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, rare


def build_raps(X_tr, y_tr, seed, m, batch, crit, lam=1.0):
    """Two-pass residual-augmented-phase-space distillation."""
    # pass 1: published pipeline
    Xc, yc, wc, _ = build_shd_generic(X_tr, y_tr, seed, m, batch, crit, 0)
    # residual field on the train pool (model 1 = KRR on the pass-1 crystal)
    from sklearn.preprocessing import StandardScaler as _SS
    scaler1 = _SS().fit(X_tr)
    Z_tr1, Z_c1 = scaler1.transform(X_tr), scaler1.transform(Xc)
    krr1 = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (X_tr.shape[1] * 1.1))
    krr1.fit(Z_c1, yc, sample_weight=wc)
    resid = y_tr - krr1.predict(Z_tr1)
    z_res = (resid - resid.mean()) / (resid.std() + 1e-12)

    # pass 2: same pipeline on the augmented feature space
    X_aug = np.hstack([X_tr, (lam * z_res)[:, None]])
    d2 = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X_aug), batch):
        d2.partial_fit(X_aug[i:i + batch], y_tr[i:i + batch])
    d2.finalize_crystal()
    Xc2 = d2.crystal_X[:, :-1]  # strip the residual column: models train on X
    yc2 = np.asarray(d2.crystal_Y, dtype=np.float64)
    rare = int(np.sum(yc2 >= crit))
    return Xc2, yc2, d2.gamma_weights, rare


def run(seeds, m=300, batch=2500, out_path="tools/_proto_raps.json"):
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

    arms = ["shd", "raps10", "raps05", "reb15k", "kmeans", "skt", "random"]
    out = {"dataset": "pm25", "m": m, "n_seeds": len(seeds), "arms": arms,
           "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        c = {}
        c["shd"] = build_shd_generic(X_tr, y_tr, seed, m, batch, crit, 0)[:3]
        c["raps10"] = build_raps(X_tr, y_tr, seed, m, batch, crit, lam=1.0)[:3]
        c["raps05"] = build_raps(X_tr, y_tr, seed, m, batch, crit, lam=0.5)[:3]
        c["reb15k"] = build_shd_generic(X_tr, y_tr, seed, m, batch, crit, 15000)[:3]
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
        print(f"  {arm:8s} all={f('krr_rmse_all'):7.2f} crit={f('krr_rmse_crit'):7.1f} "
              f"r2={f('krr_r2'):6.3f} gbm_all={f('gbm_rmse_all'):7.2f} "
              f"gbm_crit={f('gbm_rmse_crit'):7.1f} rare={f('rare_captured'):.0f}")
    print("\nWINS vs shd (lower better, r2 higher better):")
    for arm in arms[1:]:
        w = {k: int(np.sum([r[arm][k] < r["shd"][k] for r in out["records"]]))
             for k in ("krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all", "gbm_rmse_crit")}
        wr2 = int(np.sum([r[arm]["krr_r2"] > r["shd"]["krr_r2"] for r in out["records"]]))
        print(f"  {arm:8s} {w} r2_wins={wr2}/{len(seeds)}")

    out["elapsed_sec"] = time.perf_counter() - t0
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved -> {out_path} ({out['elapsed_sec']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="tools/_proto_raps.json")
    args = ap.parse_args()
    run(list(range(args.seeds)), out_path=args.output)


if __name__ == "__main__":
    main()

"""PM2.5 kappa-scale sweep (background run).

Question: does the kappa*y phase coordinate dominate the lattice geometry
and stretch background cells along y (hypothesis for the global deficit)?
Arms: kappa in {1.0, 2.0, 3.2, 6.0} via the published pipeline, plus
shd/kmeans/skt/random references. Writes tools/_proto_kappa.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
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


def build_shd_kappa(X, y, seed, m, batch, crit, kappa):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=kappa, random_state=seed,
    )
    for i in range(0, len(X), batch):
        d.partial_fit(X[i:i + batch], y[i:i + batch])
    d.finalize_crystal()
    rare = int(np.sum(np.asarray(d.crystal_Y, dtype=np.float64) >= crit))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, rare


def run(seeds=(0, 1, 2), m=300, batch=2500):
    X, y, _ = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    arms = ["k10", "k20", "k32", "k60", "shd", "kmeans", "skt", "random"]
    out = {"arms": arms, "records": [], "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        c = {}
        for ka, kv in (("k10", 1.0), ("k20", 2.0), ("k32", 3.2), ("k60", 6.0)):
            c[ka] = build_shd_kappa(X_tr, y_tr, seed, m, batch, crit, kv)[:3]
        c["shd"] = build_shd(X_tr, y_tr, seed, m, batch, crit)[:3]
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
    print("\nMEDIANS:")
    for arm in arms:
        f = lambda k: float(np.median([r[arm][k] for r in out["records"]]))  # noqa: E731
        print(f"  {arm:7s} all={f('krr_rmse_all'):7.2f} crit={f('krr_rmse_crit'):7.1f} "
              f"r2={f('krr_r2'):6.3f} gbm_all={f('gbm_rmse_all'):7.2f} rare={f('rare_captured'):.0f}")
    out["elapsed_sec"] = time.perf_counter() - t0
    with open("tools/_proto_kappa.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("saved -> tools/_proto_kappa.json")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    run(seeds=tuple(range(n)))

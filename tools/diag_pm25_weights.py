"""PM2.5 weight-structure diagnostics (Tony-Stark round, 2026-10-22).

Why does the SHD crystal lose the GLOBAL metrics on PM2.5 (krr_rmse_all
median 95.1 vs random 88.2 / kmeans 88.9, krr_r2 -0.017 vs 0.124) while
WINNING the critical metrics? Candidate causes:

1. WEIGHT SKEW: the crystal reserves 26% of cells for the high-y band
   (rare_median ~10-25 of 300) and the FISTA/gamma weights may concentrate
   mass on the band while the band's GEOMETRY (spiky high-PM2.5 winter
   nights) interpolates badly for the global model. The hybrid probe
   (bench_pm25_hybrid.py) showed swap50 (drop 50 lowest-weight medoids, add
   50 k-means medoids) improves BOTH all and crit -- pointing at a
   low-weight background deficit, not just weights.
2. BAND MASS: mass (sum of normalised weights) on the top-26% band vs the
   band's share of cells; random baseline spends ~26% of its 300 points in
   the band by construction (i.i.d.), SHD spends high_band_ratio * 300.
3. LOW-WEIGHT TAIL: how many medoids sit below w/median < 0.1, and how
   much y-range they cover (are the dropped ones informative?).

Prints per-seed: krr_all / krr_crit for shd, random, kmeans; band mass,
band cell share, low-weight counts, and the swap50 delta again for
reference. Writes tools/_pm25_diag.json for the report.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from bench_pm25 import load_pm25  # noqa: E402
from bench_seeds import build_kmeans, build_shd, eval_models  # noqa: E402


def run(seeds=(0, 1, 2, 3, 4), m=300, batch=2500):
    X, y, title = load_pm25()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)

    rows = []
    for seed in seeds:
        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        w = np.asarray(wc, dtype=np.float64)
        w = w / w.mean()

        # band = top 26% by y inside the crystal (the lattice's high band)
        k_top = max(1, int(round(len(yc) * 0.26)))
        band_idx = np.argsort(yc)[-k_top:]
        is_band = np.zeros(len(yc), dtype=bool)
        is_band[band_idx] = True

        mass_band = float(w[is_band].sum())
        share_band = float(is_band.mean())
        # low-weight tail
        med = float(np.median(w))
        n_low01 = int(np.sum(w < 0.1 * med))
        n_low03 = int(np.sum(w < 0.3 * med))
        mass_low01 = float(w[w < 0.1 * med].sum())
        # y-range coverage of the low-weight tail
        y_low = yc[w < 0.1 * med]
        y_hi = yc[w >= 0.1 * med]

        rec_shd = eval_models(Xc, yc, wc, scaler, 42, X_te, Z_te, y_te,
                              X_te_crit, Z_te_crit, y_te_crit, gamma)

        # uniform variant, same geometry
        rec_unif = eval_models(Xc, yc, None, scaler, 42, X_te, Z_te, y_te,
                               X_te_crit, Z_te_crit, y_te_crit, gamma)

        # random baseline for reference
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), size=m, replace=False)
        rec_rand = eval_models(X_tr[idx], y_tr[idx], None, scaler, 42,
                               X_te, Z_te, y_te,
                               X_te_crit, Z_te_crit, y_te_crit, gamma)

        rows.append({
            "seed": int(seed),
            "shd_all": rec_shd["krr_rmse_all"], "shd_crit": rec_shd["krr_rmse_crit"],
            "unif_all": rec_unif["krr_rmse_all"], "unif_crit": rec_unif["krr_rmse_crit"],
            "rand_all": rec_rand["krr_rmse_all"], "rand_crit": rec_rand["krr_rmse_crit"],
            "shd_r2": rec_shd["krr_r2"], "unif_r2": rec_unif["krr_r2"],
            "rand_r2": rec_rand["krr_r2"],
            "band_mass": mass_band, "band_share": share_band,
            "n_low01": n_low01, "n_low03": n_low03, "mass_low01": mass_low01,
            "y_low_min": float(y_low.min()) if len(y_low) else None,
            "y_low_max": float(y_low.max()) if len(y_low) else None,
            "y_keep_max": float(y_hi.max()),
            "rare": int(n_rare),
        })
        print(f"seed {seed}: shd_all={rec_shd['krr_rmse_all']:.1f} "
              f"unif_all={rec_unif['krr_rmse_all']:.1f} "
              f"rand_all={rec_rand['krr_rmse_all']:.1f} | "
              f"shd_crit={rec_shd['krr_rmse_crit']:.1f} "
              f"unif_crit={rec_unif['krr_rmse_crit']:.1f} "
              f"rand_crit={rec_rand['krr_rmse_crit']:.1f} | "
              f"band_mass={mass_band:.2f} share={share_band:.2f} "
              f"low01={n_low01} r2={rec_shd['krr_r2']:.3f}/{rec_unif['krr_r2']:.3f}"
              f"/{rec_rand['krr_r2']:.3f}", flush=True)

    def med(key):
        return float(np.median([r[key] for r in rows]))

    print("\nMEDIANS over seeds:")
    print(f"  krr_all: shd={med('shd_all'):.2f} unif={med('unif_all'):.2f} "
          f"rand={med('rand_all'):.2f}")
    print(f"  krr_crit: shd={med('shd_crit'):.2f} unif={med('unif_crit'):.2f} "
          f"rand={med('rand_crit'):.2f}")
    print(f"  krr_r2: shd={med('shd_r2'):.3f} unif={med('unif_r2'):.3f} "
          f"rand={med('rand_r2'):.3f}")
    print(f"  band_mass={med('band_mass'):.3f} band_share={med('band_share'):.3f} "
          f"n_low01={med('n_low01'):.0f} mass_low01={med('mass_low01'):.4f}")

    with open("tools/_pm25_diag.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "medians": {k: med(k) for k in
                               ("shd_all", "unif_all", "rand_all", "shd_crit",
                                "unif_crit", "rand_crit", "shd_r2", "unif_r2",
                                "rand_r2", "band_mass", "band_share",
                                "n_low01", "mass_low01")}},
                  fh, indent=1)
    print("saved -> tools/_pm25_diag.json")


if __name__ == "__main__":
    run(seeds=tuple(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)))

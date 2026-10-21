"""Does the budget shrink invalidate the published PM2.5 result?

bench_pm25.py fetches PM2.5 from the network (41,757 rows). The repo ships a
LOCAL copy with only 5,937 usable rows -- enough to reproduce the *structure*
of the experiment (same code path, same arms) even if the absolute numbers
differ.

What we test: at tail_quantile=97 the phase-peak class holds only ~3% of the
rows, but the band budget is M_high = 78 medoids. The comparison must
therefore be at EQUAL DELIVERED SIZE, not equal requested size -- exactly the
trap ARTIFACTS.md warns about for bench_budget_*.json.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import SpectralHologramDistiller  # noqa: E402


def load_local_pm25(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    cbwd_values = ["NW", "NE", "SE", "cv"]
    data = []
    for r in rows:
        try:
            y = float(r["pm2.5"])
            nums = [float(r[k]) for k in ("year", "month", "day", "hour", "DEWP", "TEMP",
                                         "PRES", "Iws", "Is", "Ir")]
        except (ValueError, TypeError, KeyError):
            continue
        if r["cbwd"] not in cbwd_values:
            continue
        onehot = [1.0 if r["cbwd"] == v else 0.0 for v in cbwd_values]
        data.append(nums + onehot + [y])
    arr = np.asarray(data, dtype=np.float64)
    return arr[:, :-1], arr[:, -1]


def kmeans_coreset(X, y, m, seed):
    from sklearn.cluster import MiniBatchKMeans

    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    km = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=seed, n_init=3).fit(Z)
    counts = np.bincount(km.labels_, minlength=m).astype(np.float64) + 1.0
    w = counts / counts.mean()
    d = ((Z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
    idx = np.argmin(d, axis=1)
    return X[idx], y[idx], w


def main():
    X, y = load_local_pm25(os.path.join(REPO, "data", "pm25.csv"))
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr, X_te, y_te = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    is_crit = y_te >= crit
    print(f"local PM2.5: n={len(X):,} train={n_tr:,} test={len(X_te):,} "
          f"crit_thr={crit:.1f} test_crit={int(is_crit.sum())}")
    print("(published run used 41,757 rows; same code path on the repo's local copy)\n")

    M = 300
    results = {}
    for label, tq, br in (("default tq=84", 84.0, 0.26),
                          ("tq=97 (published)", 97.0, 0.26),
                          ("tq=97, br=0.50", 97.0, 0.50)):
        for seed in range(5):
            d = SpectralHologramDistiller(crystal_size=M, high_band_ratio=br,
                                           sketch_dim=256, tail_quantile=tq,
                                           phase_kappa=3.2, random_state=seed)
            for i in range(0, n_tr, 2500):
                d.partial_fit(X_tr[i:i + 2500], y_tr[i:i + 2500])
            d.finalize_crystal()
            got = len(d.crystal_X)
            rare = int(np.sum(d.crystal_Y >= crit))
            results.setdefault(label, []).append((got, rare))
        g = [r[0] for r in results[label]]
        rc = [r[1] for r in results[label]]
        print(f"  {label:22s}: delivered median {int(np.median(g)):3d} / requested {M}   "
              f"rare captured median {np.median(rc):.1f}  (delivered: {g})")

    print("\nK-Means baseline always delivers exactly m:")
    for seed in range(3):
        Xc, yc, wc = kmeans_coreset(X_tr, y_tr, M, seed)
        print(f"  kmeans seed{seed}: {len(Xc)} rows, rare {int(np.sum(yc >= crit))}")

    print("\nEqual-delivered-size fairness check (tq=97):")
    got = int(np.median([r[0] for r in results["tq=97 (published)"]]))
    rare_shd = float(np.median([r[1] for r in results["tq=97 (published)"]]))
    rare_km = []
    for seed in range(5):
        Xc, yc, wc = kmeans_coreset(X_tr, y_tr, got, seed)
        rare_km.append(int(np.sum(yc >= crit)))
    print(f"  SHD tq=97 delivers {got} medoids, captures {rare_shd:.1f} critical")
    print(f"  K-Means at the SAME delivered size ({got}) captures {float(np.median(rare_km)):.1f} critical")


if __name__ == "__main__":
    main()

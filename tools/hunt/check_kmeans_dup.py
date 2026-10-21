"""Do the baseline coreset builders really deliver exactly m DISTINCT rows?

bench_seeds.py build_kmeans:
    idx = np.argmin(cdist(km.cluster_centers_, Z_tr), axis=1)
    return X_tr[idx], y_tr[idx], w
np.argmin picks ONE nearest row per centre. If two centres pick the SAME row,
the coreset contains DUPLICATES -> fewer than m distinct points, and the
baseline is trained on fewer effective samples than SHD (or on duplicates).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

rng = np.random.default_rng(0)
n, d, m = 17010, 81, 300
X = rng.normal(size=(n, d))
mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
Z = (X - mu) / sd
km = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=0, n_init=3).fit(Z)
idx = np.argmin(cdist(km.cluster_centers_, Z, metric="sqeuclidean"), axis=1)
print(f"requested m = {m}")
print(f"returned rows          = {len(idx)}")
print(f"DISTINCT rows returned = {len(np.unique(idx))}")
print(f"duplicate slots        = {len(idx) - len(np.unique(idx))}")
print(f"effective sample size  = {len(np.unique(idx))} instead of {m}")

# same check on the reactor generator
n2, d2 = 16000, 4
X2 = rng.normal(size=(n2, d2))
y2 = np.sin(2.0 * X2[:, 0]) + 0.5 * X2[:, 1] * X2[:, 2] + 0.1 * rng.normal(size=n2)
hot = np.abs(X2[:, 0]) > 1.8
y2[hot] += 6.0 + 1.5 * X2[hot, 3]
mu2, sd2 = X2.mean(axis=0), X2.std(axis=0) + 1e-9
Z2 = (X2 - mu2) / sd2
km2 = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=0, n_init=3).fit(Z2)
idx2 = np.argmin(cdist(km2.cluster_centers_, Z2, metric="sqeuclidean"), axis=1)
print(f"\nreactor: requested {m}, returned {len(idx2)}, DISTINCT {len(np.unique(idx2))}, "
      f"duplicates {len(idx2) - len(np.unique(idx2))}")

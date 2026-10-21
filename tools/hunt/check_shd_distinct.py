"""Are SHD's medoids DISTINCT? (the K-Means baseline's are not)"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

rng = np.random.default_rng(0)
n, d, M = 8000, 40, 300
X = rng.normal(size=(n, d))
y = 3.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=n)
hot = rng.random(n) < 0.186
y[hot] += 20.0
X[hot] += 0.8

dist = SpectralHologramDistiller(crystal_size=M, high_band_ratio=0.26, sketch_dim=128,
                                 tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, n, 2500):
    dist.partial_fit(X[i:i + 2500], y[i:i + 2500])
dist.finalize_crystal()
C = dist.crystal_X
uniq = np.unique(np.round(C, 9), axis=0)
print(f"SHD: returned {len(C)} medoids, DISTINCT {len(uniq)}, duplicates {len(C) - len(uniq)}", flush=True)

dm = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
nearest = dm.argmin(axis=0)
print(f"SHD medoids mapping to DISTINCT stream rows: {len(np.unique(nearest))} / {len(C)}", flush=True)

mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
Z = (X - mu) / sd
km = MiniBatchKMeans(n_clusters=M, batch_size=2048, random_state=0, n_init=3).fit(Z)
idx = np.argmin(cdist(km.cluster_centers_, Z, metric="sqeuclidean"), axis=1)
print(f"K-Means: returned {len(idx)} rows, DISTINCT {len(np.unique(idx))}, "
      f"duplicates {len(idx) - len(np.unique(idx))}", flush=True)
print(f"=> 'equal budget M=300' is really {len(uniq)} (SHD) vs "
      f"{len(np.unique(idx))} (K-Means) distinct points", flush=True)

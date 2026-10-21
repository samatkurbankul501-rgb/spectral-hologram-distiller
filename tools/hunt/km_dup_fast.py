"""K-Means baseline: how many DISTINCT rows does it actually deliver? (fast)"""
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

rng = np.random.default_rng(0)
n, d, m = 6000, 81, 300
X = rng.normal(size=(n, d))
mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
Z = (X - mu) / sd
res = []
for seed in range(4):
    km = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=seed, n_init=3).fit(Z)
    idx = np.argmin(cdist(km.cluster_centers_, Z, metric="sqeuclidean"), axis=1)
    u = len(np.unique(idx))
    res.append(u)
    print("seed %d: returned %d, DISTINCT %d, dup slots %d" % (seed, len(idx), u, len(idx) - u), flush=True)
print("median distinct = %d of %d requested (%.1f%%)" % (np.median(res), m, 100 * np.median(res) / m))

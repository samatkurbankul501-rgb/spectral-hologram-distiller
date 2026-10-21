"""Edge cases of save_hologram(rank_passport=...)"""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="rank_")
rng = np.random.default_rng(0)
n, d = 3000, 20
X = rng.normal(size=(n, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=n)
hot = rng.random(n) < 0.18
y[hot] += 8.0; X[hot] += 0.9
dist = SpectralHologramDistiller(crystal_size=120, random_state=7)
for i in range(0, n, 512):
    dist.partial_fit(X[i:i+512], y[i:i+512])
dist.finalize_crystal()

print("k (sketch_dim) =", dist.k)
for rank in (0, 1, -1, 32, 256, 300, 1000):
    p = os.path.join(tmp, f"r{rank}.npz")
    try:
        dist.save_hologram(p, rank_passport=rank)
        ld = HologramCrystalLoader(p)
        rep = ld.check_domain_shift(X[:512])
        print(f"  rank_passport={rank:5d} -> stored rank={ld.eigvecs.shape[1]:4d} "
              f"in-domain BW={rep['bw_distance']:.5f} shifted={rep['is_shifted']}")
    except Exception as e:
        print(f"  rank_passport={rank:5d} -> {type(e).__name__}: {e}")

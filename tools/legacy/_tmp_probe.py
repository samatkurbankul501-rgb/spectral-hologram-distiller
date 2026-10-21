# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import os, sys, time, tempfile
import numpy as np

sys.path.insert(0, str(REPO_PATH))
from shd import SpectralHologramDistiller, HologramCrystalLoader

rng = np.random.default_rng(0)
# 81-dim synthetic, 4500 warmup + a few batches
X = rng.normal(size=(6000, 81))
y = X[:, 0] + 0.1 * rng.normal(size=6000)
d = SpectralHologramDistiller(crystal_size=300, sketch_dim=256, random_state=0,
                              max_init_buffer=4500)
for i in range(0, len(X), 2000):
    d.partial_fit(X[i:i+2000], y[i:i+2000])
d.finalize_crystal()
tmp = tempfile.mkdtemp(prefix='shd_size_')
p = os.path.join(tmp, 'c.hologram.npz')
kb = d.save_hologram(p, include_projector=True)
print('crystal_shape', d.crystal_X.shape, 'file_kb', round(kb, 2))

ld = HologramCrystalLoader(p)
for B in (64, 1024, 5780):
    window = X[:B]
    ts = []
    for _ in range(5):
        r = ld.check_domain_shift(window)
        ts.append(r['latency_ms'])
    print(f'B={B:5d} latency_ms median={np.median(ts):.3f} bw={r["bw_distance"]:.4f} thr={r["threshold"]:.4f}')

# Delta-width ambiguity: non-delta 2-feature crystal fed a 1-feature window
X2 = rng.normal(size=(500, 2)); y2 = X2[:, 0]
d2 = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=1, max_init_buffer=200)
for i in range(0, 500, 200):
    d2.partial_fit(X2[i:i+200], y2[i:i+200])
d2.finalize_crystal()
p2 = os.path.join(tmp, 'c2.hologram.npz'); d2.save_hologram(p2)
ld2 = HologramCrystalLoader(p2)
try:
    r = ld2.check_domain_shift(rng.normal(size=(32, 1)))
    print('1-feature window into 2-feature non-delta crystal: ACCEPTED', r['bw_distance'])
except ValueError as e:
    print('1-feature window rejected:', e)

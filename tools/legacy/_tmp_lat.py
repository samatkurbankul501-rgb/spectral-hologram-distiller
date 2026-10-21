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
X = rng.normal(size=(6000, 81)); y = X[:, 0]
d = SpectralHologramDistiller(crystal_size=300, sketch_dim=256, random_state=0, max_init_buffer=4500)
for i in range(0, len(X), 2000):
    d.partial_fit(X[i:i+2000], y[i:i+2000])
d.finalize_crystal()
tmp = tempfile.mkdtemp(); p = os.path.join(tmp, 'c.npz'); d.save_hologram(p)
ld = HologramCrystalLoader(p)
for B in (64, 1024, 5780):
    w = X[:B]
    ld.check_domain_shift(w)
    ts = []
    for _ in range(20):
        t = time.perf_counter(); ld.check_domain_shift(w); ts.append((time.perf_counter()-t)*1000)
    print(f'B={B:5d} min={min(ts):.2f} p50={np.median(ts):.2f} p90={np.percentile(ts,90):.2f} max={max(ts):.2f}')

# component timing for B=64
w = X[:64]
t = time.perf_counter(); Xn, dX = ld._normalize_stream(w); t1 = time.perf_counter()
U = ld._spectral_transform(Xn); t2 = time.perf_counter()
Psi = U.T @ U; t3 = time.perf_counter()
Bsub = ld.eigvecs.T @ Psi @ ld.eigvecs; t4 = time.perf_counter()
ev = np.linalg.eigvalsh(Bsub); t5 = time.perf_counter()
print('components ms: normalize %.2f transform %.2f Psi %.2f proj %.2f eigh %.2f' % ((t1-t)*1e3,(t2-t1)*1e3,(t3-t2)*1e3,(t4-t3)*1e3,(t5-t4)*1e3))
print('Omega shape', d.Omega.shape, 'dtype', d.Omega.dtype)
np.show_config()

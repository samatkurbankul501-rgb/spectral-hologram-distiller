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
w = X[:64]
for _ in range(5):
    ld.check_domain_shift(w)

def tm(fn, n=50):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    return np.median(ts)

Z = ld._normalize_stream(w)
print('normalize', tm(lambda: ld._normalize_stream(w)))
print('embed', tm(lambda: np.sqrt(2/ld.k_dim)*np.cos(Z @ ld.Omega + ld.phase)))
U = np.sqrt(2/ld.k_dim)*np.cos(Z @ ld.Omega + ld.phase)
U = U / (np.linalg.norm(U, axis=1, keepdims=True)+1e-8)
print('Psi', tm(lambda: U.T @ U))
Psi = U.T @ U
print('proj', tm(lambda: ld.eigvecs.T @ Psi @ ld.eigvecs))
B = ld.eigvecs.T @ Psi @ ld.eigvecs
print('eigh32', tm(lambda: np.linalg.eigvalsh(B)))
print('sqrt32', tm(lambda: np.linalg.eigh(B)))
print('check', tm(lambda: ld.check_domain_shift(w)))

# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import os, sys, tempfile
import numpy as np
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(REPO_PATH / "tests"))
from test_smoke import make_stream
from shd import SpectralHologramDistiller, HologramCrystalLoader

X, y = make_stream(n=8000, seed=123)
d = SpectralHologramDistiller(
    crystal_size=120, high_band_ratio=0.26, sketch_dim=64,
    tail_quantile=84.0, phase_kappa=3.2, random_state=7,
)
for i in range(0, len(X), 1000):
    d.partial_fit(X[i:i+1000], y[i:i+1000])
rel = d.finalize_crystal()
path = os.path.join(tempfile.mkdtemp(prefix="shd_golden_"), "golden.npz")
kb = d.save_hologram(path)
print("n", len(d.crystal_X), "shape", d.crystal_X.shape)
print("y_median", repr(float(np.median(d.crystal_Y))))
print("w_median", repr(float(np.median(d.gamma_weights))))
print("rel_err", repr(float(rel)))
print("kb", kb)
print("summary", HologramCrystalLoader(path).summary())
X_shift = X[:200].copy()
X_shift[:, 0] = X_shift[:, 0] * 2.0 + 4.0
r = HologramCrystalLoader(path).check_domain_shift(X_shift)
print("bw", repr(r["bw_distance"]), "thr", repr(r["threshold"]), "shift", r["is_shifted"])

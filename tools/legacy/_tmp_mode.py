# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import sys, numpy as np
sys.path.insert(0, str(REPO_PATH))
from shd import SpectralHologramDistiller

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 4)); y = X[:, 0]
d = SpectralHologramDistiller(crystal_size=10, sketch_dim=16, random_state=0, max_init_buffer=200)
d.partial_fit(X[:200])              # label-free
try:
    d.partial_fit(X[200:], y[200:])  # switch to supervised
    d.finalize_crystal()
    print('mode switch accepted; crystal', d.crystal_X.shape, 'y range', float(np.min(d.crystal_Y)), float(np.max(d.crystal_Y)))
except Exception as e:
    print('mode switch rejected:', type(e).__name__, e)

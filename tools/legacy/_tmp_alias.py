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
X = rng.normal(size=(200, 4)); y = X[:, 0]
d = SpectralHologramDistiller(crystal_size=10, sketch_dim=16, random_state=0, max_init_buffer=4500)
d.partial_fit(X, y)
X[:] = 999.0
d.finalize_crystal()
print('crystal rows after mutating the input buffer:')
print(d.crystal_X[:3])
print('unique values:', np.unique(d.crystal_X)[:5])

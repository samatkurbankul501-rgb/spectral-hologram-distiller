# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import sys, os, tempfile
import numpy as np
sys.path.insert(0, str(REPO_PATH))
from shd import SpectralHologramDistiller, HologramCrystalLoader

for n, m, deltas in [(1, 10, False), (5, 300, False), (4, 300, True), (2, 50, False)]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    try:
        d = SpectralHologramDistiller(crystal_size=m, sketch_dim=32, include_deltas=deltas, random_state=0)
        d.partial_fit(X, y)
        d.finalize_crystal()
        p = os.path.join(tempfile.mkdtemp(prefix="shd_tiny_"), "c.npz")
        d.save_hologram(p)
        ld = HologramCrystalLoader(p)
        r = ld.check_domain_shift(X[:1])
        print(f"n={n} M={m} deltas={deltas}: crystal={len(d.crystal_X)} ok, radar={r['bw_distance']:.3f}, size={os.path.getsize(p)}B")
    except Exception as e:
        print(f"n={n} M={m} deltas={deltas}: ERROR {type(e).__name__}: {e}")

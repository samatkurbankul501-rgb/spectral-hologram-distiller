# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import os, sys, tempfile
import numpy as np

sys.path.insert(0, str(REPO_PATH))
from shd import SpectralHologramDistiller, HologramCrystalLoader
from shd.cli import _load_tabular

rng = np.random.default_rng(0)
n = 600
X = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)])
y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
y[::20] += 6.0  # rare band

d = SpectralHologramDistiller(crystal_size=40, high_band_ratio=0.26, sketch_dim=64,
                              tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, n, 200):
    d.partial_fit(X[i:i+200], y[i:i+200])
d.finalize_crystal()
print("crystal size:", len(d.crystal_X))

tmp = tempfile.mkdtemp(prefix="shd_cap_")
path = os.path.join(tmp, "c.hologram.npz")
d.save_hologram(path)
print("saved KB:", round(os.path.getsize(path) / 1024, 1))

for cap in [60, 30, 25, 20]:
    ld = HologramCrystalLoader(path)
    try:
        rep = ld.heal_domain_shift(X[:120] * 3.0 + 5.0, y[:120] + 2.0, n_inject=30, max_crystal_size=cap, random_state=1)
        print(f"cap={cap}: healed={rep.get('healed')} n_injected={rep['n_injected']} n_pruned={rep['n_pruned']} final={len(ld.X)} -> over_cap={len(ld.X) > cap}")
    except ValueError as e:
        print(f"cap={cap}: ValueError: {e}")

# CLI npz target-col check
tmp2 = tempfile.mkdtemp(prefix="shd_cli_npz_")
npz = os.path.join(tmp2, "x.npz")
np.savez(npz, X=np.column_stack([X, y]))
Xa, ya = _load_tabular(npz, target_col=-1)
print("npz without y, target_col=-1 -> y is None:", ya is None, "X shape:", None if Xa is None else Xa.shape)
npz2 = os.path.join(tmp2, "x2.npz")
np.savez(npz2, X=X, y=y)
Xb, yb = _load_tabular(npz2, target_col=0)
print("npz with y, target_col=0 -> y is None:", yb is None, "X shape:", None if Xb is None else Xb.shape, "y shape:", None if yb is None else yb.shape)

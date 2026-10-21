"""Probe set 3: CLI usability of the shipped artifact, and target-column integrity."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402

tmp = tempfile.mkdtemp(prefix="h3_")
env = dict(os.environ)
env["PYTHONPATH"] = "/sessions/affectionate-gallant-archimedes/mnt/outputs/_shim:" + REPO

print("=" * 72)
print("C) CLI on the SHIPPED legacy artifact (artifacts/superconduct_v37.hologram.npz)")
print("=" * 72)
legacy = os.path.join(REPO, "artifacts", "superconduct_v37.hologram.npz")
with np.load(legacy, allow_pickle=False) as z:
    print(f"  stored arrays: {sorted(z.files)}")
    print(f"  has projector: {'Omega' in z.files}   (README: 'predates the stored projector')")
    print(f"  crystal_X shape: {z['crystal_X'].shape}")

csvp = os.path.join(tmp, "in.csv")
np.savetxt(csvp, np.random.default_rng(0).normal(size=(64, 81)), delimiter=",")
for cmd in (["radar", "--crystal", legacy, "--input", csvp],
            ["heal", "--crystal", legacy, "--input", csvp, "--output", os.path.join(tmp, "h.npz")]):
    r = subprocess.run([sys.executable, "-m", "shd"] + cmd, cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    print(f"\n  $ shd {cmd[0]} -> exit {r.returncode}")
    for line in (r.stdout or "").strip().splitlines()[-5:]:
        print("    out:", line)
    for line in (r.stderr or "").strip().splitlines()[-4:]:
        print("    err:", line)

print("\n" + "=" * 72)
print("D) target-column integrity after a LABEL-FREE heal")
print("=" * 72)
rng = np.random.default_rng(1)
n, d = 3000, 6
X = rng.normal(size=(n, d))
y = 100.0 + 5.0 * X[:, 0] + 0.3 * rng.normal(size=n)     # realistic industrial units
dist = SpectralHologramDistiller(crystal_size=100, random_state=3)
for i in range(0, n, 500):
    dist.partial_fit(X[i:i + 500], y[i:i + 500])
dist.finalize_crystal()
p = os.path.join(tmp, "d.npz")
dist.save_hologram(p)
ld = HologramCrystalLoader(p)
old = ld.y.copy()
drift = X[:300] * 2.0 + 3.0
res = ld.heal_domain_shift(drift, None, n_inject=25, random_state=0, force=True)
new = ld.y
nnew = len(new) - len(old)
print(f"  original y range      : [{old.min():8.3f} .. {old.max():8.3f}]  (units of the target)")
print(f"  injected (novelty) ys : [{new[-nnew:].min():8.3f} .. {new[-nnew:].max():8.3f}]")
print(f"  -> the crystal now carries TWO INCOMPATIBLE target semantics")
print(f"  -> get_training_data() returns y = concatenation of both")
Xc, yc, wc = ld.get_training_data()
print(f"     combined y: [{yc.min():.3f} .. {yc.max():.3f}], n={len(yc)}")
# what does a downstream model see?
print(f"     -> a regression trained on this sees {nnew}/{len(yc)} rows with a different scale")

print("\n" + "=" * 72)
print("E) budget shrink -> the documented compression ratio is wrong")
print("=" * 72)
for tq in (97.0, 99.0, 99.5):
    dd = SpectralHologramDistiller(crystal_size=300, random_state=3, tail_quantile=tq)
    for i in range(0, n, 500):
        dd.partial_fit(X[i:i + 500], y[i:i + 500])
    dd.finalize_crystal()
    got = len(dd.crystal_X)
    print(f"  n_train={n}, M=300, tail_quantile={tq}: delivered {got} medoids "
          f"({got / n:.2%} of the stream, not {300 / n:.2%})")

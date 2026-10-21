"""Edge contracts around finalize_crystal() and the streaming API."""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import SpectralHologramDistiller

rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 5))
y = X[:, 0] * 2 + rng.normal(size=2000)

d = SpectralHologramDistiller(crystal_size=50, random_state=0)
for i in range(0, 2000, 400):
    d.partial_fit(X[i:i+400], y[i:i+400])
r1 = d.finalize_crystal()
print(f"finalize #1 -> rel spectral error {r1:.6f}, crystal {len(d.crystal_X)}")

# calling finalize twice
try:
    r2 = d.finalize_crystal()
    print(f"finalize #2 -> rel spectral error {r2:.6f}  (allowed)")
except Exception as e:
    print(f"finalize #2 -> {type(e).__name__}: {e}")

# absorbing MORE data after finalize: the lattice is already built, so the
# online path updates cell centres but the crystal is NOT rebuilt
d.partial_fit(X[:400] + 5.0, y[:400] + 9.0)
r3 = d.finalize_crystal()
print(f"after a post-finalize batch + re-finalize -> rel err {r3:.6f}, "
      f"crystal {len(d.crystal_X)}")
print("  (did the medoids change?)")
d2 = SpectralHologramDistiller(crystal_size=50, random_state=0)
for i in range(0, 2000, 400):
    d2.partial_fit(X[i:i+400], y[i:i+400])
d2.finalize_crystal()
same = np.allclose(d.crystal_X, d2.crystal_X)
print(f"  identical to a distiller that never saw the extra batch: {same}")

# empty stream
d3 = SpectralHologramDistiller(crystal_size=10)
try:
    d3.finalize_crystal()
    print("finalize on an empty stream -> OK (unexpected)")
except Exception as e:
    print(f"finalize on an empty stream -> {type(e).__name__}: {str(e)[:80]}")

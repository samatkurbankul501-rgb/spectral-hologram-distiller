"""What does the BW radar actually measure?

Hypothesis under test
---------------------
The passport stores Psi = E[u u^T] (the SECOND MOMENT of the RFF sketch rows),
not the covariance. So the metric is NOT translation-invariant: a pure MEAN
shift of the stream can move it a lot, while a change of the stream's SHAPE
(distribution shape, same mean) can move it little.

That distinction matters operationally: "the pump is running 6% hotter" (level
drift) and "the pump's operating distribution changed shape" (mode/regime
change) are different failures, and the radar should be honest about which one
it sees.

Experiment
----------
base        : the training stream
level shift : X + c * 1        (mean moves, shape preserved)
shape shift : rotate the cloud in a random orthonormal basis (mean preserved,
              covariance eigenvalues preserved -> second moment changes only
              through cross terms)
scale shift : X * s            (mean and variance move)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from scipy import linalg as sla

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402

tmp = tempfile.mkdtemp(prefix="sem_")
rng = np.random.default_rng(5)
n, d = 4000, 20
X = rng.normal(size=(n, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=n)
hot = rng.random(n) < 0.18
y[hot] += 8.0
X[hot] += 0.9

dist = SpectralHologramDistiller(crystal_size=150, random_state=7)
for i in range(0, n, 512):
    dist.partial_fit(X[i:i + 512], y[i:i + 512])
dist.finalize_crystal()
path = os.path.join(tmp, "c.npz")
dist.save_hologram(path)
ld = HologramCrystalLoader(path)

base = ld.check_domain_shift(X[:512])
print(f"in-distribution BW={base['bw_distance']:.6f}  thr={base['threshold']:.5f}\n")

# ---- level shift (mean moves, shape preserved) ------------------------------
print("LEVEL shift (mean moves, shape preserved):")
for c in (0.25, 0.5, 1.0, 2.0):
    rep = ld.check_domain_shift((X + c)[:512])
    print(f"  c={c:4.2f} (mean +{c:.2f}, std unchanged): BW={rep['bw_distance']:.6f} "
          f"shifted={rep['is_shifted']}")

# ---- shape shift (mean preserved) -------------------------------------------
print("\nSHAPE shift (mean preserved, orthogonal rotation):")
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
for t in (0.1, 0.25, 0.5, 1.0):
    Xs = X @ (np.eye(d) + t * (Q - Q.T))       # rotation, mean ~ preserved
    rep = ld.check_domain_shift(Xs[:512])
    moved = np.linalg.norm(Xs - X, axis=1).mean()
    print(f"  t={t:4.2f} (rows moved {moved:.2f}, mean preserved): BW={rep['bw_distance']:.6f} "
          f"shifted={rep['is_shifted']}")

# ---- scale shift ------------------------------------------------------------
print("\nSCALE shift (variance moves):")
for s in (1.1, 1.25, 1.5, 2.0):
    rep = ld.check_domain_shift((X * s)[:512])
    print(f"  s={s:4.2f}: BW={rep['bw_distance']:.6f} shifted={rep['is_shifted']}")

# ---- shape shift WITHOUT the mean (center first) ---------------------------
print("\nSHAPE shift on a CENTRED stream (mean exactly 0 in feature space):")
Xc = X - X.mean(axis=0)
for t in (0.25, 0.5, 1.0):
    Xs = Xc @ (np.eye(d) + t * (Q - Q.T))
    rep = ld.check_domain_shift(Xs[:512])
    print(f"  t={t:4.2f}: BW={rep['bw_distance']:.6f} shifted={rep['is_shifted']}")

# ---- how big is the sketch mean? -------------------------------------------
Z = np.clip((X[:512] - ld.X_mean) / ld.X_std, -8, 8)
U = np.sqrt(2.0 / dist.k) * np.cos(Z @ ld.Omega + ld.phase)
U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
mu = U.mean(axis=0)
print(f"\nsketch row mean |mu| = {np.linalg.norm(mu):.4f} "
      f"(a zero-mean stream would give 0) -> the passport is a SECOND MOMENT, "
      f"so the metric is not translation-invariant")

# ---- does the radar fire on a stream whose MEAN is unchanged but whose
#      marginal distribution changed a lot? (e.g. heavy tail) -----------------
print("\nMARGINAL shape change, mean preserved (heavy tail injected):")
for w in (0.02, 0.05, 0.1):
    Xs = X.copy()
    tail_rows = rng.random(n) < w
    Xs[tail_rows] *= 6.0
    rep = ld.check_domain_shift(Xs[:512])
    print(f"  tail frac={w:4.2f} (6x outliers): BW={rep['bw_distance']:.6f} "
          f"shifted={rep['is_shifted']}")

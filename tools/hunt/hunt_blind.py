"""Probe B (rebuilt): can a domain shift hide in the discarded spectral directions?

The passport keeps the top-32 eigenvectors of the sketched covariance
Psi_hat = U^T U / n, where U is the 256-dim RFF sketch. A feature-space shift
along u moves every sketch row roughly along Omega^T u. If Omega^T u is nearly
orthogonal to the kept 32-dim subspace, the shift should be invisible to the
radar -- even when it is huge in feature space.

Construction: pick w in R^256 orthogonal to the kept eigenvectors, then solve
Omega^T u = w (least squares) for a feature-space direction u that excites
ONLY the discarded sketch subspace.
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

tmp = tempfile.mkdtemp(prefix="blind_")
rng = np.random.default_rng(11)

# 81-dim stream, like UCI superconductivity in shape (rank 81 in sketch space)
n, d = 3000, 81
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

Psi_hat = dist.Psi_sum / dist.n_seen
w_eig, V = sla.eigh(Psi_hat)
kept = V[:, -32:]
print(f"trace(Psi_hat)={np.trace(Psi_hat):.6f}  top-32 mass={w_eig[-32:].sum():.6f}  "
      f"discarded mass={w_eig[:-32].sum():.6f}")
print(f"sketch dim k={dist.k}, feature dim d={d}, data rank in sketch space <= {d}")

# orthonormal basis of the discarded subspace (256-32 dims)
basis = sla.null_space(kept.T)          # 256 x 224, columns orthonormal
w = basis[:, 0]
w = w / np.linalg.norm(w)

# feature direction u with Omega^T u = w
# The shift Omega^T u must land in BOTH the discarded subspace and the
# reachable range of Omega^T (which is span of Omega's right singular vectors).
Omega = dist.Omega                       # (d, k)
_, _, Vt = np.linalg.svd(Omega, full_matrices=False)
P = Vt.T                                 # (k, d) basis of range(Omega^T)
M = basis.T @ P                          # project the discarded basis onto it
# take the singular direction with the largest overlap
u_sv, s_sv, vt_sv = np.linalg.svd(M, full_matrices=False)
c = vt_sv[0]
w = P @ c                                # in range(Omega^T) AND orthogonal to `kept`
w = w / np.linalg.norm(w)
print(f"overlap singular value = {s_sv[0]:.4f} (0 => no blind direction exists)")

u = np.linalg.lstsq(Omega.T, w, rcond=None)[0]
residual = np.linalg.norm(Omega.T @ u - w)
u = u / np.linalg.norm(u)
print(f"Omega^T u residual = {residual:.3e}  (0 => the shift excites ONLY discarded directions)")
print(f"kept-subspace leakage of Omega^T u = {np.linalg.norm(kept.T @ (Omega.T @ u)):.3e}")

base = ld.check_domain_shift(X[:512])
print(f"\nin-distribution:            BW={base['bw_distance']:.6f} thr={base['threshold']:.5f} "
      f"shifted={base['is_shifted']}")

# Magnitude of the blind shift: 4 sigma of the ORIGINAL data along u
sigma = X.std(axis=0)
for amp in (2.0, 4.0, 8.0, 16.0):
    delta = amp * sigma * u
    Xs = X + delta
    rep = ld.check_domain_shift(Xs[:512])
    # how far did the data actually move, per-feature and overall?
    moved = np.linalg.norm(Xs - X, axis=1).mean()
    print(f"blind shift amp={amp:5.1f}: BW={rep['bw_distance']:.6f} shifted={rep['is_shifted']}  "
          f"(mean row displacement {moved:.2f} in raw units)")

# control: an equally large shift along a random direction
for amp in (4.0, 16.0):
    ur = rng.normal(size=d)
    ur /= np.linalg.norm(ur)
    delta = amp * sigma * ur
    rep = ld.check_domain_shift((X + delta)[:512])
    moved = np.linalg.norm(X + delta - X, axis=1).mean()
    print(f"random shift amp={amp:5.1f}: BW={rep['bw_distance']:.6f} shifted={rep['is_shifted']}  "
          f"(mean row displacement {moved:.2f})")

# control: a shift along the TOP retained eigenvector direction in feature space
v = kept[:, 0]
# find u1 with Omega^T u1 ~ v
u1 = np.linalg.lstsq(Omega.T, v, rcond=None)[0]
u1 /= np.linalg.norm(u1)
for amp in (4.0, 16.0):
    delta = amp * sigma * u1
    rep = ld.check_domain_shift((X + delta)[:512])
    moved = np.linalg.norm(X + delta - X, axis=1).mean()
    print(f"visible shift amp={amp:5.1f}: BW={rep['bw_distance']:.6f} shifted={rep['is_shifted']}  "
          f"(mean row displacement {moved:.2f})")

# summary of blindness ratio
b16 = ld.check_domain_shift((X + 16.0 * sigma * u)[:512])
r16 = ld.check_domain_shift((X + 16.0 * sigma * u1)[:512])
print(f"\nBLINDNESS RATIO at amp=16: blind BW={b16['bw_distance']:.6f} vs visible BW={r16['bw_distance']:.6f} "
      f"-> blind shift is {r16['bw_distance'] / max(b16['bw_distance'], 1e-12):.1f}x less visible")
print(f"blind shift detected? {b16['is_shifted']}   visible shift detected? {r16['is_shifted']}")

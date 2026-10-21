"""Isolate WHY the stored (float16) passport disagrees with the exact one.

Variants, all on the same window:
  (a) exact      : float64 passport, float64 projector          -> ground truth
  (b) f16 pass   : float16 passport, float64 projector
  (c) f32 proj  : float64 passport, float32 projector
  (d) stored     : float16 passport, float32 projector          -> what users get
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from scipy import linalg as sla

sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402


def bw(V, lam, U_new):
    r = len(lam)
    A = np.diag(np.maximum(lam, 1e-7))
    B = V.T @ ((U_new.T @ U_new) / len(U_new)) @ V + 1e-7 * np.eye(r)
    B = 0.5 * (B + B.T)
    sA = np.diag(np.sqrt(np.diag(A)))
    inner = sA @ B @ sA
    inner = 0.5 * (inner + inner.T)
    ev = np.maximum(sla.eigvalsh(inner), 0.0)
    return float(np.sqrt(max(0.0, np.trace(A) + np.trace(B) - 2.0 * np.sum(np.sqrt(ev)))))


def embed(X, mean, std, Omega, phase, k):
    Z = np.clip((X - mean) / std, -8.0, 8.0)
    U = np.sqrt(2.0 / k) * np.cos(Z @ Omega + phase)
    return U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)


rng = np.random.default_rng(0)
n, d = 4000, 8
X = rng.normal(size=(n, d))
y = 2.0 * X[:, 0] + 0.7 * X[:, 1] * X[:, 2] + np.sin(1.3 * X[:, 3]) + 0.05 * rng.normal(size=n)
hot = rng.random(n) < 0.18
y[hot] += 8.0
X[hot] += 0.9

dist = SpectralHologramDistiller(crystal_size=120, random_state=7)
for i in range(0, n, 512):
    dist.partial_fit(X[i:i + 512], y[i:i + 512])
dist.finalize_crystal()

tmp = tempfile.mkdtemp(prefix="iso_")
path = os.path.join(tmp, "c.npz")
dist.save_hologram(path)
ld = HologramCrystalLoader(path)

Psi_hat = dist.Psi_sum / dist.n_seen
w64, V64 = sla.eigh(Psi_hat)
r = 32
lam64 = w64[-r:]
Vex = V64[:, -r:]

lam16 = ld.eigvals.astype(np.float64)          # float16 -> float64
V16 = ld.eigvecs.astype(np.float64)           # float16 -> float64

print("tr(Psi_hat)              =", np.trace(Psi_hat))
print("sum(top-32 eigvals)      =", lam64.sum())
print("lam16 == lam64 ?         =", np.array_equal(lam16, lam64))
print("max |lam16 - lam64|      =", np.abs(lam16 - lam64).max())
print("rel err of lam           =", (np.abs(lam16 - lam64) / np.abs(lam64)).max())
print("V16 orthogonality defect =", np.abs(V16.T @ V16 - np.eye(r)).max())
print("Vex orthogonality defect =", np.abs(Vex.T @ Vex - np.eye(r)).max())
print("max |V16 - Vex|          =", np.abs(V16 - Vex).max())
print("V column norms (V16)     =", np.linalg.norm(V16, axis=0)[:5])
print("V column norms (Vex)     =", np.linalg.norm(Vex, axis=0)[:5])

for B in (64, 512, 4000):
    Xin = X[:B]
    U64 = embed(Xin, dist.X_mean, dist.X_std, dist.Omega, dist.phase, dist.k)
    U32 = embed(Xin, ld.X_mean, ld.X_std, ld.Omega, ld.phase, dist.k)

    a = bw(Vex, lam64, U64)
    b = bw(V16, lam64, U64)
    c = bw(Vex, lam64, U32)
    d_ = bw(V16, lam16, U32)
    print(f"\nB={B}")
    print(f"  (a) exact f64 pass + f64 proj : {a:.8f}")
    print(f"  (b) f16 V   + f64 lam + f64 proj: {b:.8f}")
    print(f"  (c) f64 V   + f64 lam + f32 proj: {c:.8f}")
    print(f"  (d) stored f16 V + f16 lam + f32 proj: {d_:.8f}")
    print(f"  |U64-U32| max = {np.abs(U64 - U32).max():.3e}")
    # same U, only the passport changes -> isolates the passport alone
    print(f"  passport-only effect (a->b, same U): {abs(a-b):.3e}  "
          f"(lam-only: {abs(bw(Vex,lam16,U64)-a):.3e}, V-only: {abs(bw(V16,lam64,U64)-a):.3e})")

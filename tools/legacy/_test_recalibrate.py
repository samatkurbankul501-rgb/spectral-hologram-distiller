# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import os, sys, tempfile
import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(REPO_PATH))
from shd import SpectralHologramDistiller, HologramCrystalLoader
from bench_seeds import load_superconduct

X, y, crit, title = load_superconduct()
A = y < 40.0
B = y >= 40.0
X_A, y_A = X[A], y[A]
X_B, y_B = X[B], y[B]
X_tr_A, X_te_A, y_tr_A, y_te_A = train_test_split(X_A, y_A, test_size=0.20, random_state=42)
X_tr_B, X_te_B, y_tr_B, y_te_B = train_test_split(X_B, y_B, test_size=0.20, random_state=42)

d = SpectralHologramDistiller(crystal_size=200, high_band_ratio=0.26, sketch_dim=256,
                              tail_quantile=84.0, phase_kappa=3.2, random_state=0)
for i in range(0, len(X_tr_A), 2500):
    d.partial_fit(X_tr_A[i:i+2500], y_tr_A[i:i+2500])
d.finalize_crystal()
path = os.path.join(tempfile.mkdtemp(prefix="shd_recal_"), "c.npz")
d.save_hologram(path, include_projector=True)

ld = HologramCrystalLoader(path)
print("pre B:", ld.check_domain_shift(X_te_B)["bw_distance"])
print("pre A:", ld.check_domain_shift(X_te_A)["bw_distance"])
rep = ld.heal_domain_shift(X_tr_B, y_tr_B, n_inject=45, max_crystal_size=220, random_state=0)
print("heal blend: after B =", ld.check_domain_shift(X_te_B)["bw_distance"], "A =", ld.check_domain_shift(X_te_A)["bw_distance"])

# Recalibrate the passport to the heal window.
_, U = ld._prepare_stream(X_tr_B)
Psi_new = (U.T @ U) / len(U)
eigvals, eigvecs = linalg.eigh(Psi_new)
rank = len(ld.eigvals)
ld.eigvals = eigvals[-rank:]
ld.eigvecs = eigvecs[:, -rank:]
ld.Psi_passport = (ld.eigvecs * ld.eigvals) @ ld.eigvecs.T
print("recalibrated: B =", ld.check_domain_shift(X_te_B)["bw_distance"], "A =", ld.check_domain_shift(X_te_A)["bw_distance"])
print("threshold B (1446 rows):", ld.check_domain_shift(X_te_B)["threshold"])

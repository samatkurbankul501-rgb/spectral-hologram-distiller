"""Exact reproduction of the published reactor arm: same split, same order."""
import sys
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from sklearn.model_selection import train_test_split
from shd import SpectralHologramDistiller


def load_reactor(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    crit = float(np.percentile(y, 97.0))
    return X, y, crit


X, y, crit = load_reactor()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
crit_tr = float(np.percentile(y_tr, 97.0))
print("train=%d crit_thr=%.4f critical share=%.2f%% expected=%.1f"
      % (len(X_tr), crit_tr, 100 * (y_tr >= crit_tr).mean(), 300 * (y_tr >= crit_tr).mean()))

caught = []
for seed in range(10):
    d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                                  tail_quantile=84.0, phase_kappa=3.2, random_state=seed)
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i:i+2500], y_tr[i:i+2500])
    d.finalize_crystal()
    caught.append(int(np.sum(d.crystal_Y >= crit_tr)))
print("critical medoids per seed:", caught)
print("median = %.1f  (published: 19.5)  IQR = %.1f"
      % (np.median(caught), np.percentile(caught, 75) - np.percentile(caught, 25)))

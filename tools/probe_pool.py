"""Probe the real shapes inside _pool_predict / fit_residual_stats on the smoke distiller."""
import sys

import numpy as np

sys.path.insert(0, ".")
from shd import SpectralHologramDistiller  # noqa: E402


def make_stream(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    return X, y


def main():
    X, y = make_stream()
    d = SpectralHologramDistiller(
        crystal_size=80, high_band_ratio=0.26, sketch_dim=64,
        tail_quantile=84.0, random_state=3,
    )
    B = 1500
    for i in range(0, len(X), B):
        d.partial_fit(X[i:i + B], y[i:i + B])
    d.finalize_crystal()

    print("Phi_sum shape:", d.Phi_sum.shape, "y_dim =", d.Phi_sum.shape[1])
    # Reproduce _pool_predict shapes step by step
    from scipy import linalg
    w_prob = d.gamma_weights / float(np.sum(d.gamma_weights))
    G = (d.crystal_U.T * w_prob) @ d.crystal_U
    A = G + d.fista_l2_reg * np.eye(G.shape[0])
    Bm = (d.crystal_U.T * w_prob) @ d.crystal_Y[:, None]
    W = linalg.cho_solve(linalg.cho_factor(A, lower=True), Bm).reshape(G.shape[0])
    print("G:", G.shape, "B:", Bm.shape, "W:", W.shape)
    X_pool = X
    fX = d._pool_predict(X_pool).reshape(-1)
    print("fX shape:", fX.shape, "y_pool shape:", y.shape)
    resid = y - fX
    print("resid shape:", resid.shape, "OK" if resid.shape == y.shape else "MISMATCH")


if __name__ == "__main__":
    main()

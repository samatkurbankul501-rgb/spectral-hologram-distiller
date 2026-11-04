"""Kernel Thinning (KT) and Supervised Kernel Thinning (SKT) baselines for SHD.

This module is a faithful pure-numpy port of the X-based (store_K=False) code
path of the OFFICIAL Microsoft `goodpoints` implementation of kernel thinning:

  Raaz Dwivedi and Lester Mackey. "Kernel Thinning."
  arXiv:2105.05842; JMLR 25(152), 2024.

  Raaz Dwivedi and Lester Mackey. "Generalized Kernel Thinning."
  arXiv:2110.01593; ICLR 2022.

Ported from https://github.com/microsoft/goodpoints (goodpoints/kt.py,
MIT License, Copyright (c) Microsoft Corporation) so that the SHD benchmark
stack keeps its zero-dependency (numpy/scipy/sklearn) philosophy — the
official package's Cython extension does not link on Windows (it requests the
Unix math library `m.lib`). Only the pure-numpy functions are ported:
`split_X`, `best_X`, `squared_emp_rel_mmd_X`, `refine_X`,
`kernel_matrix_row_mean` (renamed `_kernel_row_means`). The Cython-accelerated
K-matrix paths (`split_K`/`refine_K`) are intentionally not ported.

SKT follows "Supervised Kernel Thinning" (Gong, Choi & Dwivedi, NeurIPS 2024,
arXiv:2410.13749) in its canonical form: run KT on the PRODUCT kernel
k((x,y),(x',y')) = k_X(x,x') * k_Y(y,y'). With Gaussian factors this equals a
single Gaussian kernel on the joint vector [x, alpha * y_scaled] with
alpha = sqrt(t_X / t_Y), which is how it is implemented here.

Bandwidths use the median heuristic (median of squared pairwise distances,
estimated on a subsample of at most 1,000 points), documented and identical
for every arm.
"""

from __future__ import annotations

import numpy as np
from numpy import newaxis
from numpy.random import default_rng
from scipy.spatial.distance import cdist

__all__ = ["median_bandwidth_sq", "make_gauss_kernels", "kt_thin", "kt_skt_joint"]


# ---------------------------------------------------------------- kernels ----

def median_bandwidth_sq(X, cap=1000, seed=0):
    """Median of squared pairwise distances (median heuristic), t > 0.

    Accepts (n, d) points or a flat (n,) vector (treated as n 1-D points).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = len(X)
    if n > cap:
        idx = default_rng(seed).choice(n, size=cap, replace=False)
        X = X[idx]
    d2 = cdist(X, X, metric="sqeuclidean")
    iu = np.triu_indices(len(X), k=1)
    med = float(np.median(d2[iu]))
    return med if med > 0 else 1.0


def make_gauss_kernels(var):
    """Return (swap_kernel, split_kernel) with the goodpoints kernel contract.

    kernel(y, X) returns a 1-D array of length len(X): entry j = k(y_row, X_j)
    when y is a single row (broadcast), and entry j = k(y_j, X_j) (paired
    evaluation) when y has the same number of rows as X — so kernel(X, X) is
    the kernel DIAGONAL, exactly as expected by goodpoints' refine_X.
    Gaussian form: exp(-||x - y||^2 / (2 * var)), var = sigma^2.

    Following the official numpy experiment configuration
    (examples/kt/run_kt_experiment.ipynb): the swap kernel is the Gaussian
    with bandwidth `var` and the split kernel is the Gaussian with
    bandwidth `var / 2` (their "gauss_rt" convention).
    """
    def gauss(y, X):
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if y.shape[0] == 1:
            d2 = np.sum((X - y[0]) ** 2, axis=1)
            return np.exp(-0.5 * d2 / var)
        # paired evaluation (diagonal when y is X itself)
        d2 = np.sum((X - y) ** 2, axis=1)
        return np.exp(-0.5 * d2 / var)

    def gauss_half(y, X):
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        if y.shape[0] == 1:
            d2 = np.sum((X - y[0]) ** 2, axis=1)
            return np.exp(-d2 / var)  # exp(-0.5*d2/(var/2))
        d2 = np.sum((X - y) ** 2, axis=1)
        return np.exp(-d2 / var)

    return gauss, gauss_half


# ------------------------------------------------- ported from goodpoints ----

def _largest_power_of_two(n):
    """Returns the largest j such that 2**j divides n."""
    return (n & (~(n - 1))).bit_length() - 1


def _split_X(X, m, kernel, delta=0.5, seed=None, verbose=False):
    """Returns 2^m KT-SPLIT coresets of size floor(n/2^m) as a 2D array.

    Faithful port of goodpoints.kt.split_X (pure numpy; O(nd) memory).
    """
    if m == 0:
        return np.arange(X.shape[0], dtype=int)[newaxis, :]

    verbose = verbose and (m >= 7)

    def k(ii, jj):
        return kernel(X[ii], X[jj])

    rng = default_rng(seed)
    n, _ = X.shape

    coresets = dict()
    KC = dict()
    sig_sqd = dict()
    log_multiplier = 2 * np.log(2 * n * m / delta)

    for j in range(m + 1):
        num_coresets = int(2 ** j)
        num_points_in_coreset = n // num_coresets
        coresets[j] = np.full((num_coresets, num_points_in_coreset), -1, dtype=int)
        KC[j] = np.empty((num_coresets, num_points_in_coreset))
        sig_sqd[j] = np.zeros(num_coresets)

    diagK = np.empty(n)

    for i in range(n):
        coreset = coresets[0][0]
        coreset[i] = i
        i_array = coreset[i, newaxis]
        ki = k(i_array, coreset[:(i + 1)])
        KC[0][0, i] = np.sum(ki[:i])
        diagK[i] = ki[i]

        for j in range(min(m, _largest_power_of_two(i + 1))):
            parent_coresets = coresets[j]
            child_coresets = coresets[j + 1]
            parent_KC = KC[j]
            child_KC = KC[j + 1]
            num_parent_coresets = parent_coresets.shape[0]
            j_log_multiplier = log_multiplier - j * TWO_LOG_2
            for j2 in range(num_parent_coresets):
                parent_coreset = parent_coresets[j2]
                parent_idx = (i + 1) // num_parent_coresets
                point1, point2 = parent_coreset[parent_idx - 2], parent_coreset[parent_idx - 1]
                point1_arr, point2_arr = point1[newaxis], point2[newaxis]
                K12 = np.asarray(k(point1_arr, point2_arr)).item()

                b_sqd = float(diagK[point2] + diagK[point1] - 2 * K12)
                thresh = max(np.sqrt(sig_sqd[j][j2] * b_sqd * j_log_multiplier), b_sqd)
                if sig_sqd[j][j2] == 0:
                    sig_sqd[j][j2] = b_sqd
                elif thresh != 0:
                    sig_sqd_update = 0.5 + (b_sqd / (2 * thresh) - 1) * sig_sqd[j][j2] / thresh
                    if sig_sqd_update > 0:
                        sig_sqd[j][j2] += 2 * b_sqd * sig_sqd_update
                if thresh == 0:
                    thresh = 1.0

                if parent_idx > 2:
                    alpha = parent_KC[j2, parent_idx - 2] - parent_KC[j2, parent_idx - 1] + K12
                else:
                    alpha = 0.0
                left_child_coreset = child_coresets[2 * j2]
                right_child_coreset = child_coresets[2 * j2 + 1]
                child_idx = (parent_idx // 2) - 1
                if child_idx > 0:
                    child_points = left_child_coreset[:child_idx]
                    point1_kernel_sum = np.sum(k(point1_arr, child_points))
                    point2_kernel_sum = np.sum(k(point2_arr, child_points))
                    alpha -= 2 * (point1_kernel_sum - point2_kernel_sum)
                else:
                    point1_kernel_sum = 0
                    point2_kernel_sum = 0
                prob_point2 = 0.5 * (1 - alpha / thresh)
                if rng.random() <= prob_point2:
                    left_child_coreset[child_idx] = point2
                    right_child_coreset[child_idx] = point1
                    child_KC[2 * j2, child_idx] = point2_kernel_sum
                    child_KC[2 * j2 + 1, child_idx] = point1_kernel_sum
                else:
                    left_child_coreset[child_idx] = point1
                    right_child_coreset[child_idx] = point2
                    child_KC[2 * j2, child_idx] = point1_kernel_sum
                    child_KC[2 * j2 + 1, child_idx] = point2_kernel_sum

    return coresets[m]


TWO_LOG_2 = 2 * np.log(2)


def _kernel_row_means(X, kernel):
    """Mean of each kernel-matrix row (goodpoints.kt.kernel_matrix_row_mean)."""
    n = X.shape[0]
    meanK = np.empty(n)
    for ii in range(n):
        meanK[ii] = np.mean(kernel(X[ii, newaxis], X))
    return meanK


def _squared_emp_rel_mmd_X(X, coreset, kernel, meanK=None):
    """RelMMD^2 = mean(K[coreset][:,coreset]) - 2*mean(K[coreset, :]).
    (goodpoints.kt.squared_emp_rel_mmd_X)"""
    k_core_core = 0.0
    k_core_all = 0.0
    if meanK is None:
        for ii in coreset:
            kii = kernel(X[ii, newaxis], X)
            k_core_core += np.mean(kii[coreset])
            k_core_all += np.mean(kii)
    else:
        k_core_all = np.sum(meanK[coreset])
        for ii in coreset:
            k_core_core += np.mean(kernel(X[ii, newaxis], X[coreset]))
    coreset_size = len(coreset)
    return (k_core_core - 2 * k_core_all) / coreset_size


def _best_X(X, coresets, kernel, meanK=None):
    """Selects the candidate coreset with smallest MMD to all points in X,
    after comparing with a baseline standard-thinning coreset.
    (goodpoints.kt.best_X)"""
    n = X.shape[0]
    coreset_size = coresets.shape[1]
    if coreset_size < 1:
        # floor(n / 2**m) == 0: the standard-thinning stride below would divide
        # by zero (upstream goodpoints fails here too, just less clearly).
        raise ValueError(
            f"kernel thinning got n={n} points for coreset size {coreset_size} "
            "(floor(n / 2**m) == 0): need n >= 2**m -- reduce m or pass more points"
        )
    best_coreset = np.array(range(n - 1, -1, -(n // coreset_size)))[::-1]
    best_rel_mmd2 = _squared_emp_rel_mmd_X(X, best_coreset, kernel, meanK=meanK)
    for coreset in coresets:
        rel_mmd2 = _squared_emp_rel_mmd_X(X, coreset, kernel, meanK=meanK)
        if rel_mmd2 < best_rel_mmd2:
            best_rel_mmd2 = rel_mmd2
            best_coreset = coreset
    return best_coreset


def _refine_X(X, coreset, kernel, meanK=None, unique=False):
    """Point-by-point swap refinement minimizing MMD to all input points.
    (goodpoints.kt.refine_X)"""
    n = X.shape[0]
    coreset = np.copy(coreset)
    coreset_size = len(coreset)
    two_over_coreset_size = 2 / coreset_size

    coreset_indicator = np.zeros(n, dtype=bool)
    coreset_indicator[coreset] = True

    if meanK is None:
        sufficient_stat = np.empty(n)
        for ii in range(n):
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                kii = kernel(X[ii, newaxis], X)
                sufficient_stat[ii] = 2 * (np.mean(kii[coreset]) - np.mean(kii)) + kii[ii] / coreset_size
    else:
        sufficient_stat = kernel(X, X) / coreset_size - 2 * meanK
        for ii in range(n):
            if unique and coreset_indicator[ii]:
                sufficient_stat[ii] = np.inf
            else:
                kiicore = kernel(X[ii, newaxis], X[coreset])
                sufficient_stat[ii] += 2 * np.mean(kiicore)

    for coreset_idx in range(coreset_size):
        if unique:
            cidx = coreset[coreset_idx]
            if meanK is None:
                kcidx = kernel(X[cidx, newaxis], X)
                sufficient_stat[cidx] = 2 * (np.mean(kcidx[coreset]) - np.mean(kcidx)) + kcidx[cidx] / coreset_size
            else:
                kcidxcore = kernel(X[cidx, newaxis], X[coreset])
                k_diag = float(np.asarray(kernel(X[cidx, newaxis], X[cidx, newaxis])).reshape(-1)[0])
                sufficient_stat[cidx] = k_diag / coreset_size - 2 * meanK[cidx]
                sufficient_stat[cidx] += 2 * np.mean(kcidxcore)

        sufficient_stat -= kernel(X[coreset[coreset_idx], newaxis], X) * two_over_coreset_size
        best_point = np.argmin(sufficient_stat)
        coreset[coreset_idx] = best_point
        sufficient_stat += kernel(X[best_point, newaxis], X) * two_over_coreset_size
        if unique:
            sufficient_stat[best_point] = np.inf
    return coreset


def kt_thin(X, m, var, delta=0.5, seed=None, unique=False):
    """Full KT pipeline on points X with Gaussian bandwidth ``var`` (= sigma^2).

    Parameters mirror the official ``goodpoints.kt.thin`` signature:
    ``unique=False`` by default (the author's default), and ``m == 0`` returns
    all row indices. Returns coreset indices of size floor(n / 2^m) into X.
    Split kernel: Gaussian with var/2; swap kernel: Gaussian with var
    (the configuration of the official numpy experiment notebook).

    ``seed=None`` (the default) means NON-DETERMINISTIC output: the split
    coin flips come from an unseeded generator; pass an explicit ``seed``
    for reproducible coresets.
    """
    X = np.asarray(X, dtype=np.float64)
    if m == 0:
        return np.arange(X.shape[0], dtype=int)
    gauss, gauss_half = make_gauss_kernels(var)
    meanK = _kernel_row_means(X, gauss)
    coresets = _split_X(X, m, gauss_half, delta=delta, seed=seed)
    coreset = _best_X(X, coresets, gauss, meanK=meanK)
    return _refine_X(X, coreset, gauss, meanK=meanK, unique=unique)


def kt_skt_joint(Z, y, t_x, t_y):
    """Joint vectors [Z, alpha * y_std] realizing the SKT product kernel.

    exp(-||z-z'||^2/t_x) * exp(-(y-y')^2/t_y) == exp(-||w-w'||^2/t_x)
    with w = [z, sqrt(t_x/t_y) * y]. Returns (W, t) for kt_thin.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    alpha = float(np.sqrt(t_x / t_y))
    W = np.column_stack([Z, alpha * y])
    return W, float(t_x)

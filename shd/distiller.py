"""Spectral Hologram Distiller (SHD) v3.7 — streaming dataset distillation.

The core pipeline, in plain terms:

1. **RFF sketch (Bochner).** Random Fourier features ``U = cos(Z @ Omega + phase)``
   (256 harmonics, row-normalised). Two accumulators ``Psi_sum = U^T U`` and
   ``Phi_sum = U^T y`` are maintained online, so memory is O(k^2 + d), not O(n).
2. **Residual trigger.** A ridge predictor on the accumulated sketch flags rows in
   the top ``tail_quantile`` percentile of ``residual_ratio + 2 * y_z^2`` as
   "phase-peak" candidates (rare / critical regimes).
3. **Phase-manifold Voronoi lattice.** A single non-overlapping Voronoi tessellation
   is built on the augmented manifold ``[Z || kappa * y_norm]``, so rare regimes
   (e.g. high-Tc superconductors) occupy their own cells and cannot be absorbed
   by the low-temperature background. Cell representatives are *real* medoids,
   never synthetic centroids.
4. **Spectral reweighting.** Weights ``gamma*`` minimise the Frobenius distance
   between ``sum_i g_i u_i u_i^T`` and the streamed ``Psi_hat``. This is
   kernel herding solved with FISTA (projected gradient with Nesterov momentum);
   the Voronoi cell mass acts as the importance-weighting prior, softened by
   root tempering ``p^0.55``. (In earlier notes this was called "Hadamard-Gram
   quadrature" and "Radon-Nikodym tempering" — same math, honest names here.)
5. **16-bit spectral passport + OOD radar.** The top-32 eigenpairs of ``Psi_hat``
   are stored in float16. A Bures-Wasserstein distance on that 32-dim subspace
   detects domain shift in milliseconds without access to the original dataset.
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

__all__ = ["SpectralHologramDistiller", "HologramCrystalLoader", "BW_ALERT_THRESHOLD"]

#: Default Bures-Wasserstein alert threshold for the domain-shift radar.
BW_ALERT_THRESHOLD = 0.085


class SpectralHologramDistiller:
    """Streaming distiller, SHD v3.7 (phase-manifold synthesis).

    Port of the reference Colab implementation (cell 9 of ``Untitled22.ipynb``),
    kept numerically identical to the published benchmark numbers.

    Parameters
    ----------
    crystal_size : int
        Number of medoids M in the final crystal (default 300).
    high_band_ratio : float
        Fraction of cells reserved for the phase-peak band (default 0.26).
    sketch_dim : int
        Number of random Fourier harmonics k (default 256).
    beta_low : float
        Ridge regularisation of the residual-trigger predictor (default 0.25).
    tail_quantile : float
        Percentile of the phase score above which a row is "phase-peak" (default 84).
    phase_kappa : float
        Weight of the ``y``-axis on the phase manifold ``[Z || kappa * y_norm]``
        (default 3.2).
    random_state : int
        Seed for the projector and k-means++ seeding.
    max_init_buffer : int
        Hard cap on the warm-up buffer size before building the Voronoi lattice
        (default 8192). Prevents O(N^2) memory spikes in ``cdist`` when
        ``crystal_size`` is large, keeping initialization strictly O(1) in RAM.
    include_deltas : bool
        When ``True``, augments the internal RFF sketch and Voronoi phase
        manifold with streaming first differences ``dX_t = X_t - X_{t-1}``
        while storing only the last row of the previous batch (O(d) memory).
        Makes the crystal sensitive to rapid rate-of-change transients in
        industrial sensor telemetry while keeping ``crystal_X`` in the original
        ``d``-dimensional feature space.
    """

    def __init__(
        self,
        crystal_size: int = 300,
        high_band_ratio: float = 0.26,
        sketch_dim: int = 256,
        beta_low: float = 0.25,
        tail_quantile: float = 84.0,
        phase_kappa: float = 3.2,
        random_state: int = 42,
        max_init_buffer: int = 8192,
        include_deltas: bool = False,
    ):
        self.M = crystal_size
        self.M_high = int(round(crystal_size * high_band_ratio))
        self.M_low = crystal_size - self.M_high
        self.k = sketch_dim
        self.beta_low = beta_low
        self.tail_q = tail_quantile
        self.kappa = phase_kappa
        self.max_init_buffer = max(int(max_init_buffer), self.M * 2)
        self.include_deltas = bool(include_deltas)
        self.rng = np.random.default_rng(random_state)

        self.n_seen = 0
        self.Psi_sum = np.zeros((self.k, self.k), dtype=np.float64)
        self.Phi_sum = None
        self._proj_initialized = False
        self.unsupervised_mode = False
        self._prev_X_row = None

        self.init_Z, self.init_X, self.init_Y, self.init_U, self.init_score = [], [], [], [], []
        self.lattice_ready = False

        self.centers_ZY = None
        self.med_X = None
        self.med_Y = None
        self.med_U = None
        self.counts = None

        self.crystal_X = None
        self.crystal_Y = None
        self.crystal_U = None
        self.gamma_weights = None

    # --- internals -----------------------------------------------------------

    def _init_projector(self, X_batch: np.ndarray, y_batch: np.ndarray):
        d = X_batch.shape[1]
        y_dim = y_batch.shape[1]
        self.Phi_sum = np.zeros((self.k, y_dim), dtype=np.float64)

        self.X_mean = np.mean(X_batch, axis=0)
        self.X_std = np.std(X_batch, axis=0) + 1e-8
        self.y_mean = float(np.mean(y_batch))
        self.y_std = float(np.std(y_batch)) + 1e-8

        scale_d = 1.0 / np.sqrt(max(d, 1))
        k_half = self.k // 2
        omega_low = self.rng.normal(0, 0.75 * scale_d, size=(d, k_half))
        omega_high = self.rng.normal(0, 2.10 * scale_d, size=(d, self.k - k_half))
        self.Omega = np.hstack([omega_low, omega_high])
        self.phase = self.rng.uniform(0, 2 * np.pi, size=(1, self.k))
        self._proj_initialized = True

    def _augment_deltas(self, X_batch: np.ndarray, update_state: bool = False) -> np.ndarray:
        if not self.include_deltas:
            return X_batch
        dX = np.zeros_like(X_batch, dtype=np.float64)
        if len(X_batch) > 1:
            dX[1:] = X_batch[1:] - X_batch[:-1]
        if self._prev_X_row is not None and len(X_batch) > 0:
            dX[0] = X_batch[0] - self._prev_X_row
        if update_state and len(X_batch) > 0:
            self._prev_X_row = X_batch[-1].copy()
        return np.hstack([X_batch, dX])

    def _normalize_X(self, X_batch: np.ndarray, update_state: bool = False) -> np.ndarray:
        X_eff = self._augment_deltas(X_batch, update_state=update_state)
        return np.clip((X_eff - self.X_mean) / self.X_std, -8.0, 8.0)

    def _make_phase_coords(self, Z_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """Coordinates on the phase manifold (Z || kappa * y_norm)."""
        y_scaled = self.kappa * ((y_batch - self.y_mean) / self.y_std)
        return np.hstack([Z_batch, y_scaled])

    def _spectral_transform(self, X_batch: np.ndarray, update_state: bool = False) -> tuple:
        Z = self._normalize_X(X_batch, update_state=update_state)
        U = np.sqrt(2.0 / self.k) * np.cos(Z @ self.Omega + self.phase)
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        return Z, U / norms

    def _seed_kmeans_pp(self, ZY_cand: np.ndarray, n_select: int) -> np.ndarray:
        n_total = len(ZY_cand)
        n_select = min(n_select, n_total)
        first_idx = int(self.rng.integers(0, n_total))
        selected = [first_idx]
        min_sq_dists = np.sum((ZY_cand - ZY_cand[first_idx]) ** 2, axis=1)

        for _ in range(1, n_select):
            probs = min_sq_dists / (np.sum(min_sq_dists) + 1e-12)
            next_idx = int(self.rng.choice(n_total, p=probs))
            selected.append(next_idx)
            new_dists = np.sum((ZY_cand - ZY_cand[next_idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_dists)

        return ZY_cand[selected].copy()

    def _build_phase_manifold_lattice(self):
        Z_all = np.vstack(self.init_Z)
        X_all = np.vstack(self.init_X)
        Y_all = np.vstack(self.init_Y)
        U_all = np.vstack(self.init_U)
        sc_all = np.concatenate(self.init_score)

        ZY_all = self._make_phase_coords(Z_all, Y_all)

        thr = np.percentile(sc_all, self.tail_q)
        is_high = sc_all >= thr

        # 1. Seed centers in phase space (Z, kappa*y): phase-peak centers are
        #    separated along the y-axis and cannot be absorbed by the background.
        c_low = self._seed_kmeans_pp(ZY_all[~is_high], self.M_low)
        c_high = self._seed_kmeans_pp(ZY_all[is_high], self.M_high)
        centers_ZY = np.vstack([c_low, c_high])
        M_total = len(centers_ZY)

        # 2. Lloyd relaxation on the phase manifold.
        for _ in range(4):
            d_mat = cdist(ZY_all, centers_ZY, metric="sqeuclidean")
            labels = np.argmin(d_mat, axis=1)
            for c_i in range(M_total):
                mask = labels == c_i
                if np.any(mask):
                    centers_ZY[c_i] = np.mean(ZY_all[mask], axis=0)

        # 3. Snap each phase center to the nearest *real* data point (medoid).
        d_mat = cdist(ZY_all, centers_ZY, metric="sqeuclidean")
        labels = np.argmin(d_mat, axis=1)
        counts = np.zeros(M_total, dtype=np.float64)
        med_X = np.zeros((M_total, X_all.shape[1]), dtype=np.float64)
        med_Z = np.zeros((M_total, Z_all.shape[1]), dtype=np.float64)
        med_Y = np.zeros((M_total, Y_all.shape[1]), dtype=np.float64)
        med_U = np.zeros((M_total, U_all.shape[1]), dtype=np.float64)

        for c_i in range(M_total):
            mask = labels == c_i
            if np.any(mask):
                counts[c_i] = float(np.sum(mask))
                d_local = np.sum((ZY_all[mask] - centers_ZY[c_i]) ** 2, axis=1)
                best_local = int(np.argmin(d_local))
                idx_global = np.where(mask)[0][best_local]
            else:
                counts[c_i] = 1.0
                idx_global = int(np.argmin(d_mat[:, c_i]))

            med_X[c_i] = X_all[idx_global]
            med_Z[c_i] = Z_all[idx_global]
            med_Y[c_i] = Y_all[idx_global]
            med_U[c_i] = U_all[idx_global]

        self.centers_ZY = centers_ZY
        self.med_X = med_X
        self.med_Z = med_Z
        self.med_Y = med_Y
        self.med_U = med_U
        self.counts = counts
        self.init_Z, self.init_X, self.init_Y, self.init_U, self.init_score = [], [], [], [], []
        self.lattice_ready = True

    # --- public API -----------------------------------------------------------

    def partial_fit(self, X_batch: np.ndarray, y_batch: np.ndarray | None = None):
        """Absorb one streaming batch.

        Supports both supervised streaming (``y_batch`` provided, shape ``(B,)``
        or ``(B, 1)``) and label-free / unsupervised sensor streaming
        (``y_batch=None``). When ``y_batch`` is ``None``, the distiller computes
        each sample's Bochner spectral leverage score
        ``s_i = u_i^T (Psi_hat + beta_low I)^(-1) u_i`` via Cholesky factorization
        and uses it as a synthetic novelty response on the phase manifold
        ``[Z || kappa * s_norm]``, preserving rare / anomalous sensor states
        without labels.
        """
        B = X_batch.shape[0]

        if y_batch is None:
            self.unsupervised_mode = True
            if not self._proj_initialized:
                X_eff0 = self._augment_deltas(X_batch, update_state=False)
                d = X_eff0.shape[1]
                self.Phi_sum = np.zeros((self.k, 1), dtype=np.float64)
                self.X_mean = np.mean(X_eff0, axis=0)
                self.X_std = np.std(X_eff0, axis=0) + 1e-8
                scale_d = 1.0 / np.sqrt(max(d, 1))
                k_half = self.k // 2
                omega_low = self.rng.normal(0, 0.75 * scale_d, size=(d, k_half))
                omega_high = self.rng.normal(0, 2.10 * scale_d, size=(d, self.k - k_half))
                self.Omega = np.hstack([omega_low, omega_high])
                self.phase = self.rng.uniform(0, 2 * np.pi, size=(1, self.k))
                Z_B, U_B = self._spectral_transform(X_batch, update_state=True)
                self.Psi_sum += U_B.T @ U_B
                self.n_seen += B
                Psi_hat = self.Psi_sum / self.n_seen
                I_k = np.eye(self.k, dtype=np.float64)
                L_low = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
                V_B = linalg.cho_solve(L_low, U_B.T).T
                lev = np.sum(U_B * V_B, axis=1)
                y_batch = lev[:, None]
                self.y_mean = float(np.mean(y_batch))
                self.y_std = float(np.std(y_batch)) + 1e-8
                self.Phi_sum += U_B.T @ y_batch
                self._proj_initialized = True
            else:
                Z_B, U_B = self._spectral_transform(X_batch, update_state=True)
                self.Psi_sum += U_B.T @ U_B
                self.n_seen += B
                Psi_hat = self.Psi_sum / self.n_seen
                I_k = np.eye(self.k, dtype=np.float64)
                L_low = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
                V_B = linalg.cho_solve(L_low, U_B.T).T
                lev = np.sum(U_B * V_B, axis=1)
                y_batch = lev[:, None]
                self.Phi_sum += U_B.T @ y_batch
            Phi_hat = self.Phi_sum / self.n_seen
            W_pred = linalg.cho_solve(L_low, Phi_hat)
        else:
            if y_batch.ndim == 1:
                y_batch = y_batch[:, None]

            if not self._proj_initialized:
                X_eff0 = self._augment_deltas(X_batch, update_state=False)
                self._init_projector(X_eff0, y_batch)

            Z_B, U_B = self._spectral_transform(X_batch, update_state=True)

            self.Psi_sum += U_B.T @ U_B
            self.Phi_sum += U_B.T @ y_batch
            self.n_seen += B

            Psi_hat = self.Psi_sum / self.n_seen
            Phi_hat = self.Phi_sum / self.n_seen

            I_k = np.eye(self.k, dtype=np.float64)
            L_low = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
            W_pred = linalg.cho_solve(L_low, Phi_hat)

        residual = np.sum((y_batch - U_B @ W_pred) ** 2, axis=1)
        res_ratio = residual / (np.median(residual) + 1e-8)
        y_z = np.maximum(0.0, (y_batch.squeeze() - self.y_mean) / self.y_std)
        phase_score = res_ratio + 2.0 * (y_z ** 2)

        if not self.lattice_ready:
            self.init_Z.append(Z_B)
            self.init_X.append(X_batch)
            self.init_Y.append(y_batch)
            self.init_U.append(U_B)
            self.init_score.append(phase_score)
            target_buf = min(self.max_init_buffer, max(4500, self.M * 12))
            if sum(len(a) for a in self.init_Z) >= target_buf:
                self._build_phase_manifold_lattice()
            return

        # Online update of the phase-manifold lattice.
        ZY_B = self._make_phase_coords(Z_B, y_batch)
        dists = cdist(ZY_B, self.centers_ZY, metric="sqeuclidean")
        assign = np.argmin(dists, axis=1)
        med_ZY = self._make_phase_coords(self.med_Z, self.med_Y)

        for c_id in np.unique(assign):
            mask = assign == c_id
            n_new = int(np.sum(mask))
            old_count = self.counts[c_id]
            new_count = old_count + n_new
            self.counts[c_id] = new_count

            lr = min(0.25, n_new / new_count)
            self.centers_ZY[c_id] = (1.0 - lr) * self.centers_ZY[c_id] + lr * np.mean(ZY_B[mask], axis=0)

            d_cand = np.sum((ZY_B[mask] - self.centers_ZY[c_id]) ** 2, axis=1)
            best_local = int(np.argmin(d_cand))
            curr_dist = float(np.sum((med_ZY[c_id] - self.centers_ZY[c_id]) ** 2))

            if float(d_cand[best_local]) <= curr_dist:
                idx_global = np.where(mask)[0][best_local]
                self.med_X[c_id] = X_batch[idx_global]
                self.med_Z[c_id] = Z_B[idx_global]
                self.med_Y[c_id] = y_batch[idx_global]
                self.med_U[c_id] = U_B[idx_global]

    def finalize_crystal(self, fista_steps: int = 250) -> float:
        """Build the crystal and fit herding weights; returns relative spectral error."""
        if not self.lattice_ready:
            self._build_phase_manifold_lattice()

        order = np.argsort(self.counts)[::-1]
        self.crystal_X = self.med_X[order].copy()
        self.crystal_Y = self.med_Y[order].squeeze().copy()
        self.crystal_U = self.med_U[order].copy()
        sorted_counts = self.counts[order].copy()
        M_curr = len(self.crystal_U)

        voronoi_prob = sorted_counts / np.sum(sorted_counts)

        Psi_hat = self.Psi_sum / self.n_seen
        G_U = self.crystal_U @ self.crystal_U.T
        K = (G_U ** 2) + 0.20 * np.eye(M_curr)
        b = np.sum((self.crystal_U @ Psi_hat) * self.crystal_U, axis=1) + 0.20 * voronoi_prob

        L_lip = float(linalg.norm(K, ord=2)) + 1e-6
        gamma = voronoi_prob.copy()
        z = gamma.copy()
        t_step = 1.0

        for _ in range(fista_steps):
            grad = K @ z - b
            gamma_next = np.maximum(1e-5, z - grad / L_lip)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_step ** 2))
            z = gamma_next + ((t_step - 1.0) / t_next) * (gamma_next - gamma)
            gamma, t_step = gamma_next, t_next

        gamma_prob = gamma / np.sum(gamma)
        Psi_crystal = (self.crystal_U.T * gamma_prob) @ self.crystal_U
        rel_error = linalg.norm(Psi_crystal - Psi_hat, ord="fro") / linalg.norm(Psi_hat, ord="fro")

        # Root tempering of the Voronoi measure (p^0.55): keeps the global RMSE
        # balanced while preserving the amplitude of the phase-peak band.
        tempered_voronoi = (voronoi_prob ** 0.55)
        tempered_voronoi /= np.sum(tempered_voronoi)
        w_combined = (0.75 * tempered_voronoi + 0.25 * gamma_prob) * M_curr
        self.gamma_weights = w_combined / np.mean(w_combined)
        return rel_error

    def save_hologram(
        self,
        filepath: str = "hologram.npz",
        rank_passport: int = 32,
        include_projector: bool = True,
    ) -> float:
        """Save the crystal to a compressed .hologram.npz; returns file size in KB.

        With ``include_projector=True`` the RFF projector (Omega, phase) is stored
        as well, so :class:`HologramCrystalLoader` can run the OOD radar without
        any reference to the original distiller object. The published artifact
        ``superconduct_v37.hologram.npz`` predates this field; the loader handles
        both formats.
        """
        Psi_hat = self.Psi_sum / self.n_seen
        eigvals, eigvecs = linalg.eigh(Psi_hat)
        payload = dict(
            crystal_X=self.crystal_X.astype(np.float32),
            crystal_Y=self.crystal_Y.astype(np.float32),
            gamma_weights=self.gamma_weights.astype(np.float32),
            passport_eigvals=eigvals[-rank_passport:].astype(np.float16),
            passport_eigvecs=eigvecs[:, -rank_passport:].astype(np.float16),
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64),
        )
        if include_projector:
            payload["Omega"] = self.Omega.astype(np.float32)
            payload["phase"] = self.phase.astype(np.float32)
        np.savez_compressed(filepath, **payload)
        return os.path.getsize(filepath) / 1024.0


class HologramCrystalLoader:
    """Client SDK for ``.hologram.npz`` crystals.

    1. Instant load of (X, y, sample_weight) for any ML/DL model.
    2. Anytime mode: training slices of the first ``k`` medoids.
    3. Domain-shift radar: Bures-Wasserstein OOD detector on the 16-bit
       spectral passport, in milliseconds, without the original dataset.

    The radar is self-contained when the artifact stores the RFF projector
    (``Omega``/``phase``, written by ``save_hologram(include_projector=True)``).
    For older artifacts pass a live ``SpectralHologramDistiller`` (or any object
    with ``_spectral_transform``) as ``distiller_ref`` to
    :meth:`check_domain_shift`.
    """

    def __init__(self, filepath: str = "hologram.npz"):
        data = np.load(filepath, allow_pickle=False)
        self.filepath = filepath
        self.X = data["crystal_X"].astype(np.float64)
        self.y = data["crystal_Y"].astype(np.float64)
        self.weights = data["gamma_weights"].astype(np.float64)
        self.eigvals = data["passport_eigvals"].astype(np.float64)
        self.eigvecs = data["passport_eigvecs"].astype(np.float64)
        self.X_mean = data["X_mean"].astype(np.float64)
        self.X_std = data["X_std"].astype(np.float64)
        self.n_seen = int(data["n_seen"][0])

        self.k_dim = self.eigvecs.shape[0]
        if "Omega" in data.files:
            self.Omega = data["Omega"].astype(np.float64)
            self.phase = data["phase"].astype(np.float64)
        else:
            self.Omega = None
            self.phase = None

        # Reconstruct the passport spectral operator from the top-32 eigenpairs.
        self.Psi_passport = (self.eigvecs * self.eigvals) @ self.eigvecs.T

    def get_training_data(self, budget_points: int | None = None):
        """Return (X, y, sample_weight) for the chosen medoid budget (anytime mode)."""
        if budget_points is None or budget_points >= len(self.X):
            return self.X, self.y, self.weights
        w_sub = self.weights[:budget_points].copy()
        w_sub /= np.mean(w_sub)
        return self.X[:budget_points], self.y[:budget_points], w_sub

    def get_phase_peak_slice(
        self,
        high_band_ratio: float = 0.26,
        y_threshold: float | None = None,
    ):
        """Return ``(X_peak, y_peak, w_peak)`` for rare / discontinuity regimes.

        Intended for warm-starting Bayesian Optimization (e.g. NEON Lab GP+EI)
        around phase transitions without fitting a cubic O(N^3) GP on the full
        telemetry stream. If ``y_threshold`` is given, selects medoids with
        ``y >= y_threshold``; otherwise selects the top ``high_band_ratio``
        fraction of medoids by target value.
        """
        if y_threshold is not None:
            mask = self.y >= float(y_threshold)
            if not np.any(mask):
                mask = np.argsort(self.y)[-max(1, int(round(len(self.y) * high_band_ratio))) :]
        else:
            k_top = max(1, int(round(len(self.y) * high_band_ratio)))
            mask = np.argsort(self.y)[-k_top:]
        X_p = self.X[mask].copy()
        y_p = self.y[mask].copy()
        w_p = self.weights[mask].copy()
        w_p /= np.mean(w_p)
        return X_p, y_p, w_p

    def _normalize_stream(self, X_new_stream: np.ndarray) -> np.ndarray:
        if len(self.X_mean) == 2 * X_new_stream.shape[1]:
            dX = np.zeros_like(X_new_stream, dtype=np.float64)
            if len(X_new_stream) > 1:
                dX[1:] = X_new_stream[1:] - X_new_stream[:-1]
            X_eff = np.hstack([X_new_stream, dX])
        else:
            X_eff = X_new_stream
        return np.clip((X_eff - self.X_mean) / self.X_std, -8.0, 8.0)

    def _embed(self, X_new_stream: np.ndarray, distiller_ref) -> np.ndarray:
        if self.Omega is not None:
            Z = self._normalize_stream(X_new_stream)
            U = np.sqrt(2.0 / self.k_dim) * np.cos(Z @ self.Omega + self.phase)
            norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            return U / norms
        if distiller_ref is None:
            raise ValueError(
                "This artifact has no stored projector (Omega/phase). Pass a "
                "distiller_ref with _spectral_transform, or re-save the crystal "
                "with save_hologram(include_projector=True)."
            )
        t = distiller_ref._spectral_transform(X_new_stream)
        return t[1] if isinstance(t, tuple) else t

    def _calibrated_threshold(self, n_batch: int, adaptive_threshold: bool = True) -> float:
        if not adaptive_threshold:
            return float(BW_ALERT_THRESHOLD)
        # Finite-sample covariance fluctuation scales as 1 / sqrt(B). For batches
        # smaller than 1024 rows, widen the threshold smoothly so small sensor
        # windows do not trigger false alarms from sampling noise alone. The
        # constant 1024 was calibrated on the UCI Superconductivity heal
        # benchmark: in-distribution windows of B=64 stay below the widened
        # threshold while true OOD windows (BW ~0.44+) are still detected.
        b_eff = max(int(n_batch), 16)
        sample_scale = max(1.0, float(np.sqrt(1024.0 / b_eff)))
        return float(BW_ALERT_THRESHOLD * sample_scale)

    def check_domain_shift(
        self,
        X_new_stream: np.ndarray,
        distiller_ref=None,
        adaptive_threshold: bool = True,
    ) -> dict:
        """Compare a new data stream against the spectral passport.

        Projects the new stream's RFF spectrum onto the 32-dim eigen-subspace of
        the passport and computes a Bures-Wasserstein distance (32x32 matrices
        instead of 256x256). Returns ``bw_distance``, ``threshold``,
        ``latency_ms`` and ``is_shifted``.
        """
        t0 = time.perf_counter()
        U_new = self._embed(X_new_stream, distiller_ref)
        Psi_new = (U_new.T @ U_new) / len(U_new)

        V = self.eigvecs  # (k, 32)
        A_sub = np.diag(np.maximum(self.eigvals, 1e-7))
        B_sub = V.T @ Psi_new @ V + 1e-7 * np.eye(len(self.eigvals))
        B_sub = 0.5 * (B_sub + B_sub.T)

        sqrt_A = np.diag(np.sqrt(np.diag(A_sub)))
        inner = sqrt_A @ B_sub @ sqrt_A
        inner = 0.5 * (inner + inner.T)
        eig_inner = np.maximum(linalg.eigvalsh(inner), 0.0)

        bw_dist = float(np.sqrt(max(0.0, np.trace(A_sub) + np.trace(B_sub) - 2.0 * np.sum(np.sqrt(eig_inner)))))
        thr = self._calibrated_threshold(len(X_new_stream), adaptive_threshold=adaptive_threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "bw_distance": bw_dist,
            "threshold": thr,
            "latency_ms": latency_ms,
            "is_shifted": bw_dist > thr,
        }

    def heal_domain_shift(
        self,
        X_new_stream: np.ndarray,
        y_new_stream: np.ndarray | None = None,
        n_inject: int = 35,
        max_crystal_size: int | None = None,
        distiller_ref=None,
        force: bool = False,
        random_state: int = 42,
    ) -> dict:
        """Surgical self-healing: inject new Voronoi medoids when OOD shift is detected.

        Checks the Bures-Wasserstein radar on ``X_new_stream``. If a shift is
        detected (or ``force=True``), selects ``n_inject`` real medoids via
        k-means++ + Lloyd relaxation in normalized space, appends them to the
        crystal without touching existing medoids (zero catastrophic forgetting),
        and updates the 32-mode spectral passport in place. If
        ``max_crystal_size`` is set and exceeded, prunes only low-weight
        background medoids while protecting the phase-peak band and newly
        injected medoids.
        """
        t0 = time.perf_counter()
        shift = self.check_domain_shift(X_new_stream, distiller_ref=distiller_ref)
        if not shift["is_shifted"] and not force:
            return {
                "healed": False,
                "n_injected": 0,
                "n_pruned": 0,
                "bw_before": shift["bw_distance"],
                "bw_after": shift["bw_distance"],
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        rng = np.random.default_rng(random_state)
        Z_new = self._normalize_stream(X_new_stream)
        U_new = self._embed(X_new_stream, distiller_ref)
        n_total = len(Z_new)
        n_sel = min(int(n_inject), n_total)

        # k-means++ seeding in Z-space
        first_idx = int(rng.integers(0, n_total))
        selected = [first_idx]
        min_sq = np.sum((Z_new - Z_new[first_idx]) ** 2, axis=1)
        for _ in range(1, n_sel):
            probs = min_sq / (np.sum(min_sq) + 1e-12)
            nxt = int(rng.choice(n_total, p=probs))
            selected.append(nxt)
            min_sq = np.minimum(min_sq, np.sum((Z_new - Z_new[nxt]) ** 2, axis=1))
        centers = Z_new[selected].copy()

        # 4 steps of Lloyd relaxation + medoid snapping to real observations
        for _ in range(4):
            d_mat = cdist(Z_new, centers, metric="sqeuclidean")
            labels = np.argmin(d_mat, axis=1)
            for c_i in range(n_sel):
                mask = labels == c_i
                if np.any(mask):
                    centers[c_i] = np.mean(Z_new[mask], axis=0)

        d_final = cdist(centers, Z_new, metric="sqeuclidean")
        med_idx = np.argmin(d_final, axis=1)
        counts = np.bincount(labels, minlength=n_sel).astype(np.float64) + 1.0
        w_patch = counts / np.mean(counts)

        X_patch = X_new_stream[med_idx].astype(np.float64).copy()
        if y_new_stream is not None:
            y_patch = np.asarray(y_new_stream, dtype=np.float64).squeeze()[med_idx].copy()
        else:
            # Unsupervised fallback: spectral novelty score against current passport
            proj = U_new[med_idx] @ self.eigvecs
            y_patch = np.sum(U_new[med_idx] ** 2, axis=1) - np.sum(proj ** 2, axis=1)

        n_old = len(self.X)
        self.X = np.vstack([self.X, X_patch])
        self.y = np.concatenate([self.y, np.atleast_1d(y_patch)])
        w_all = np.concatenate([self.weights, w_patch])
        self.weights = w_all / np.mean(w_all)

        n_pruned = 0
        if max_crystal_size is not None and len(self.X) > int(max_crystal_size):
            cap = max(int(max_crystal_size), n_sel + 4)
            excess = len(self.X) - cap
            if excess > 0:
                # Protect top 26% phase-peak medoids among old medoids + all new medoids
                n_protect_old = max(1, int(round(0.26 * n_old)))
                peak_old_idx = np.argsort(self.y[:n_old])[-n_protect_old:]
                protected = np.zeros(len(self.X), dtype=bool)
                protected[peak_old_idx] = True
                protected[n_old:] = True
                prunable_idx = np.where(~protected)[0]
                if len(prunable_idx) > 0:
                    n_drop = min(excess, len(prunable_idx))
                    drop_order = prunable_idx[np.argsort(self.weights[prunable_idx])[:n_drop]]
                    keep_mask = np.ones(len(self.X), dtype=bool)
                    keep_mask[drop_order] = False
                    # Reassign dropped weights to nearest kept medoids
                    X_drop = self.X[drop_order]
                    w_drop = self.weights[drop_order]
                    self.X = self.X[keep_mask]
                    self.y = self.y[keep_mask]
                    self.weights = self.weights[keep_mask]
                    nearest = np.argmin(cdist(X_drop, self.X, metric="sqeuclidean"), axis=1)
                    np.add.at(self.weights, nearest, w_drop)
                    self.weights /= np.mean(self.weights)
                    n_pruned = int(n_drop)

        # Blend the spectral passport with the new regime so subsequent radar
        # checks recognize the healed domain as in-distribution.
        Psi_new = (U_new.T @ U_new) / max(len(U_new), 1)
        alpha_blend = min(0.50, max(0.15, len(U_new) / (self.n_seen + len(U_new))))
        Psi_blend = (1.0 - alpha_blend) * self.Psi_passport + alpha_blend * Psi_new
        Psi_blend = 0.5 * (Psi_blend + Psi_blend.T)
        rank_p = len(self.eigvals)
        eigvals, eigvecs = linalg.eigh(Psi_blend)
        self.eigvals = eigvals[-rank_p:].astype(np.float64)
        self.eigvecs = eigvecs[:, -rank_p:].astype(np.float64)
        self.Psi_passport = (self.eigvecs * self.eigvals) @ self.eigvecs.T
        self.n_seen += len(X_new_stream)

        after = self.check_domain_shift(X_new_stream, distiller_ref=distiller_ref)
        return {
            "healed": True,
            "n_injected": int(n_sel),
            "n_pruned": int(n_pruned),
            "bw_before": shift["bw_distance"],
            "bw_after": after["bw_distance"],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def save(self, filepath: str | None = None) -> float:
        """Persist the (possibly self-healed) crystal back to ``.hologram.npz``.

        Returns the saved file size in KB.
        """
        out_path = filepath or self.filepath
        payload = dict(
            crystal_X=self.X.astype(np.float32),
            crystal_Y=self.y.astype(np.float32),
            gamma_weights=self.weights.astype(np.float32),
            passport_eigvals=self.eigvals.astype(np.float16),
            passport_eigvecs=self.eigvecs.astype(np.float16),
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64),
        )
        if self.Omega is not None and self.phase is not None:
            payload["Omega"] = self.Omega.astype(np.float32)
            payload["phase"] = self.phase.astype(np.float32)
        np.savez_compressed(out_path, **payload)
        return os.path.getsize(out_path) / 1024.0

    def summary(self) -> dict:
        """Return a JSON-serializable metadata summary of the loaded crystal."""
        size_kb = os.path.getsize(self.filepath) / 1024.0 if os.path.exists(self.filepath) else 0.0
        return {
            "filepath": self.filepath,
            "file_size_kb": round(float(size_kb), 2),
            "n_medoids": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "n_seen": int(self.n_seen),
            "compression_ratio": round(float(self.n_seen) / max(len(self.X), 1), 2),
            "sketch_dim": int(self.k_dim),
            "passport_rank": int(len(self.eigvals)),
            "has_projector": bool(self.Omega is not None),
            "y_min": float(np.min(self.y)),
            "y_median": float(np.median(self.y)),
            "y_max": float(np.max(self.y)),
        }



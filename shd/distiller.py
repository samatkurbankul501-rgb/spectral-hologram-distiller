"""Spectral Hologram Distiller (SHD) — streaming dataset distillation.

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

Threading: ``HologramCrystalLoader`` is thread-safe for the read-only
paths (``check_domain_shift``, ``save``, ``get_training_data``, ``summary``)
via a per-loader re-entrant lock (v3.14.4). Mutating paths
(``heal_domain_shift``, ``calibrate_threshold*``) acquire the same lock;
concurrent heals serialize rather than race. The distiller
(``SpectralHologramDistiller``) mutates streaming state (``_prev_X_row``)
and its accumulators on every batch and remains single-threaded by
contract — one object per thread; single-threaded use is the supported
deployment.
"""

from __future__ import annotations

import os
import threading
import time
import warnings

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

__all__ = ["SpectralHologramDistiller", "HologramCrystalLoader", "BW_ALERT_THRESHOLD", "SHD_VERSION"]

#: Default Bures-Wasserstein alert threshold for the domain-shift radar.
BW_ALERT_THRESHOLD = 0.085

#: Package version, single source of truth (re-exported as ``shd.__version__``).
#: Recorded into every .hologram.npz at save time so a crystal can always tell
#: which build distilled it.
SHD_VERSION = "3.15.2"


def _calibration_fields(prefix, calibrated, window, n_windows, fa_rate):
    """Return the artifact fields for one radar calibration group.

    A group is written only when it is COMPLETE. A half-populated group
    (calibrated set but window / n_windows / fa_rate None) used to crash
    ``save()`` with ``int(None)`` -- and save() is the persist path, so that
    crash loses the result of a heal. An incomplete group is dropped
    instead: the radar falls back to its default threshold.
    """
    if calibrated is None:
        return {}
    if window is None or n_windows is None or fa_rate is None:
        import warnings
        warnings.warn(
            "dropping the incomplete " + prefix + " calibration from the "
            "artifact: the radar falls back to its default threshold",
            UserWarning, stacklevel=2,
        )
        return {}
    return {
        prefix + "_calibrated": np.array([float(calibrated)], dtype=np.float64),
        prefix + "_window": np.array([int(window)], dtype=np.int64),
        prefix + "_n_windows": np.array([int(n_windows)], dtype=np.int64),
        prefix + "_fa_rate": np.array([float(fa_rate)], dtype=np.float64),
    }


def npz_output_path(path: str) -> str:
    """Return the path numpy will actually write for a crystal artifact.

    ``np.savez_compressed`` appends ``.npz`` when the name lacks it, so
    ``save_hologram("out.hologram")`` writes ``out.hologram.npz`` while the
    follow-up ``getsize`` looks for ``out.hologram`` and raises
    FileNotFoundError -- with a stray file left on disk. v3.15.1 gave the
    library the same normalisation the CLI already had.
    """
    p = str(path)
    if not p.lower().endswith(".npz"):
        p = p + ".npz"
    return p




class SpectralHologramDistiller:
    """Streaming distiller (phase-manifold synthesis).

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
    tail : str
        Which tail of the target distribution the phase-peak trigger protects
        (default ``"high"``). ``"high"`` flags rows with an unusually large
        ``y`` (e.g. high-Tc superconductors), ``"low"`` flags rows with an
        unusually small ``y`` (e.g. pressure/voltage sags in industrial
        telemetry), ``"both"`` flags either extreme. In label-free mode the
        synthetic novelty coordinate is high spectral leverage by
        construction, so ``"high"`` is always used there regardless of this
        option.
    adaptive_y_stats : bool
        When ``True``, ``y_mean`` / ``y_std`` keep adapting after the Voronoi
        lattice is built via an exponential moving average (rate
        ``min(0.1, B / n_seen)``), so the ``kappa * y`` phase coordinate does
        not go stale on long streams with target drift. Default ``False``
        preserves the frozen first-batch statistics of the published
        benchmarks.
    omega_low_scale : float
        Std of the low-frequency half of the RFF projection matrix, in units
        of ``1 / sqrt(d)`` (default 0.75).
    omega_high_scale : float
        Std of the high-frequency half of the RFF projection matrix, in units
        of ``1 / sqrt(d)`` (default 2.10).
    center_lr : float
        Upper bound of the online lattice centroid learning rate
        (default 0.25).
    fista_l2_reg : float
        Diagonal ridge added to the herding Gram matrix and the matching
        linear term (default 0.20).
    voronoi_temper : float
        Exponent of the Voronoi-measure tempering ``p ** voronoi_temper``
        (default 0.55). 0.0 disables tempering (uniform relative volumes);
        1.0 uses raw cell volumes.
    herding_mix : float
        Blend of tempered Voronoi weights vs FISTA herding weights in the
        final coreset sample weights (default 0.75). 1.0 = pure tempered
        Voronoi, 0.0 = pure herding.
    use_residual : bool
        Whether the phase trigger adds the ridge prediction residual to the
        target-extremity term (default True). False isolates a y-only trigger.
    critical_threshold : float or None
        If set (supervised mode), the phase-peak band is defined by
        ``y >= critical_threshold`` instead of the phase-score percentile.
        ``auto_band_ratio=True`` then allocates cells in proportion to the
        observed critical fraction (clipped to 10%-30%); this is the
        critical-aware mode for domains with a known alarm threshold.
    lattice_rebuild_interval : int
        How often (in absorbed rows) the Voronoi lattice is REBUILT from the
        retained reservoir instead of only being nudged online.

        ``0`` (the published default) NEVER rebuilds. The lattice is built
        once from the warm-up window (``min(max_init_buffer, max(4500,
        12 * crystal_size))`` rows) and afterwards only nudged online; the
        rebuild reservoir is not collected at all. A rare regime that
        appears LATER in the stream therefore reaches the crystal with
        almost no dedicated cells: measured on a 60k-row synthetic stream
        whose last 10% is a shifted regime, the default kept 1 of 300
        critical medoids for that regime. The default is kept bit-for-bit
        because every published artifact was distilled with it; an earlier
        docstring wrongly promised a one-shot post-warm-up rebuild at 0 --
        the code has gated both the reservoir and the rebuild on ``> 0``
        since the mechanism was introduced (v3.12.0).

        ``> 0`` rebuilds the lattice every ``interval`` absorbed rows from a
        bounded reservoir (evicting background stratum + never-evicting peak
        stratum), so late-arriving regimes get their own cells: the same
        60k-row probe kept 126 of 300 critical medoids at interval=3000.
        Use it for long streams with drift or late regimes (the regression
        suite uses 1500). Each rebuild is O(reservoir * M) and is amortised
        over the stream; the reservoir stays bounded by ``max_init_buffer``
        rows, so memory remains O(k^2 + d + reservoir).
    peak_z_threshold : float
        |y| in standard deviations above which a streamed row is retained in
        the reservoir's *peak* stratum (which never evicts) instead of the
        evicting *background* stratum. Default 2.0.
    passport_reference : str
        Which population the spectral passport describes.

        ``"stream"`` (default, v3.11.1) builds the passport over every row.
        This is the reference the published artifacts were distilled with, so
        it keeps them reproducible and keeps the radar's alert semantics
        identical to the documented ones.

        ``"background"`` (opt-in, experimental) builds the passport from the
        BACKGROUND rows only -- the rows that did NOT trigger the phase-peak
        score. The radar then answers "is this window the background
        regime?", which is the question an early-warning system actually asks:
        measured on a 20k-row stream, a shifted regime occupying 10% of the
        TRAINING data made the whole-stream passport flag 100% of the normal
        windows as shifted (BW 0.158 vs threshold 0.120), because the
        reference itself absorbed the shift. It is NOT the default because it
        changes the radar's sensitivity, not only its reference: on the
        shipped regression suite it moved the golden shift distance
        (0.50448 -> 0.52960) and stopped the heal from firing on a shifted
        window. Enable it per-run (``--passport-reference background``) and
        re-benchmark before trusting it.
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
        tail: str = "high",
        adaptive_y_stats: bool = False,
        omega_low_scale: float = 0.75,
        omega_high_scale: float = 2.10,
        center_lr: float = 0.25,
        fista_l2_reg: float = 0.20,
        voronoi_temper: float = 0.55,
        herding_mix: float = 0.75,
        residual_weight: float = 1.0,
        rff_mode: str = "dual",
        on_nonfinite: str = "raise",
        use_residual: bool = True,
        critical_threshold: float | None = None,
        auto_band_ratio: bool = False,
        lattice_rebuild_interval: int = 0,
        peak_z_threshold: float = 2.0,
        passport_reference: str = "stream",
        global_guard: bool = False,
        global_guard_n: int = 150,
        global_guard_weight: float = 0.35,
    ):
        if tail not in ("high", "low", "both"):
            raise ValueError(f"tail must be 'high', 'low' or 'both', got {tail!r}")
        if on_nonfinite not in ("raise", "drop"):
            raise ValueError(f"on_nonfinite must be 'raise' or 'drop', got {on_nonfinite!r}")
        # Lattice-geometry sanity: reject parameter sets that cannot produce a
        # usable Voronoi lattice before any streaming state is allocated.
        if int(crystal_size) < 1:
            raise ValueError(f"crystal_size must be >= 1, got {crystal_size}")
        if int(sketch_dim) < 2:
            raise ValueError(f"sketch_dim must be >= 2, got {sketch_dim}")
        if not 0.0 <= float(high_band_ratio) <= 1.0:
            raise ValueError(
                f"high_band_ratio must be within [0, 1], got {high_band_ratio}"
            )
        self.on_nonfinite = on_nonfinite
        self.d_in = None  # fixed by the first non-empty batch
        self.M = crystal_size
        self.M_high = int(round(crystal_size * high_band_ratio))
        # Raw ratio kept for artifact persistence: heal_domain_shift protects
        # this fraction of the old crystal's phase-peak band when pruning.
        self.high_band_ratio = float(high_band_ratio)
        self.M_low = crystal_size - self.M_high
        self.k = sketch_dim
        self.beta_low = beta_low
        self.tail_q = tail_quantile
        self.kappa = phase_kappa
        self.max_init_buffer = max(int(max_init_buffer), self.M * 2)
        self.include_deltas = bool(include_deltas)
        self.tail = tail
        self.adaptive_y_stats = bool(adaptive_y_stats)
        self.omega_low_scale = float(omega_low_scale)
        self.omega_high_scale = float(omega_high_scale)
        self.center_lr = float(center_lr)
        self.fista_l2_reg = float(fista_l2_reg)
        self.voronoi_temper = float(voronoi_temper)
        self.herding_mix = float(herding_mix)
        self.residual_weight = float(residual_weight)
        if self.residual_weight < 0.0:
            raise ValueError(f"residual_weight must be >= 0, got {residual_weight}")
        if rff_mode not in ("dual", "single_low", "single_high"):
            raise ValueError(f"rff_mode must be 'dual'|'single_low'|'single_high', got {rff_mode!r}")
        self.rff_mode = rff_mode
        # rff_mode collapses the two scales into one; the dual default is
        # bit-for-bit the pre-v3.14.4 behaviour (0.75 + 2.10).
        # The active scale is recorded; _init_projector (and the inline
        # reservoir-only init) materialise all k channels from it, so a
        # "single" mode is NOT a dual stack with both halves at one scale.
        # omega_low/high_scale keep their construction-time values so
        # external inspection shows the original request; effective
        # collapse happens in _rff_scale.
        if self.rff_mode == "single_low":
            self._rff_scale = float(omega_low_scale)
        elif self.rff_mode == "single_high":
            self._rff_scale = float(omega_high_scale)
        else:
            self._rff_scale = None
        self.use_residual = bool(use_residual)
        self.critical_threshold = None if critical_threshold is None else float(critical_threshold)
        if self.critical_threshold is not None and tail == "low":
            # The critical band is selected by y >= critical_threshold (the
            # UPPER side of the target), while tail="low" makes the loader
            # serve the LOWEST medoids as the phase-peak band. The two
            # contradict each other: the crystal is built from the high side
            # and read back from the low side (measured: band built from
            # y >= 1.0, slice returned y in [-2.31, -0.69]).
            raise ValueError(
                "critical_threshold selects the UPPER target side (y >= "
                "threshold) but tail='low' serves the LOWEST medoids; use "
                "tail='high' (or 'both') together with critical_threshold, or "
                "drop critical_threshold"
            )
        self.auto_band_ratio = bool(auto_band_ratio)
        self.lattice_rebuild_interval = int(lattice_rebuild_interval)
        if self.lattice_rebuild_interval < 0:
            raise ValueError(
                f"lattice_rebuild_interval must be >= 0, got {lattice_rebuild_interval}"
            )
        self.peak_z_threshold = float(peak_z_threshold)
        if passport_reference not in ("background", "stream"):
            raise ValueError(
                f"passport_reference must be 'background' or 'stream', got {passport_reference!r}"
            )
        self.passport_reference = passport_reference
        self.global_guard = bool(global_guard)
        self.global_guard_n = int(global_guard_n)
        self.global_guard_weight = float(global_guard_weight)
        if self.global_guard and self.global_guard_n > self.M:
            # v3.15.0: the guard swaps the lowest-weight rows for K-Means
            # global medoids. The published default (n_global=150) assumes a
            # 300-row crystal; on a smaller crystal it raised at save time
            # ("n_global=150 exceeds crystal size 60"), which made the
            # global_guard presets unusable below 150 rows. Clamp to half the
            # crystal -- the documented intent (reserve half the budget for
            # global coverage) -- and say so out loud.
            clamped = max(1, self.M // 2)
            warnings.warn(
                f"global_guard_n={self.global_guard_n} exceeds crystal size "
                f"{self.M}; clamping to {clamped} (half the crystal). "
                "Pass global_guard_n explicitly to override.",
                UserWarning, stacklevel=2,
            )
            self.global_guard_n = clamped
        self.rng = np.random.default_rng(random_state)

        self.n_seen = 0
        self.Psi_sum = np.zeros((self.k, self.k), dtype=np.float64)
        # Background-only accumulator: second moment of the sketch rows that
        # did NOT trigger the phase-peak score. Used for the passport when
        # passport_reference="background" (see the class docstring).
        self.Psi_bg_sum = np.zeros((self.k, self.k), dtype=np.float64)
        self.n_bg = 0
        self.Phi_sum = None
        self._proj_initialized = False
        self.unsupervised_mode = False
        self._mode = None  # "supervised" | "label_free", locked by the first non-empty batch
        self._prev_X_row = None

        self.init_Z, self.init_X, self.init_Y, self.init_U, self.init_score = [], [], [], [], []
        self.lattice_ready = False

        # Reservoir of streamed rows used to REBUILD the lattice once the stream
        # has grown past the warm-up window (see lattice_rebuild_interval).
        # Bounded by max_init_buffer rows so memory stays O(k^2 + d + reservoir).
        self.res_peak_Z, self.res_peak_X, self.res_peak_Y = [], [], []
        self.res_peak_U, self.res_peak_score = [], []
        self.res_bg_Z, self.res_bg_X, self.res_bg_Y = [], [], []
        self.res_bg_U, self.res_bg_score = [], []
        self._res_peak_rows = 0
        # Total peak-stratum rows ever OFFERED to the reservoir (v3.12.0):
        # algorithm-R replacement needs the offer count, not the retained
        # count, so late peaks get a fair chance against early ones.
        self._res_peak_seen = 0
        self._res_bg_rows = 0
        self._reservoir_rows = 0
        self._rows_since_rebuild = 0

        # v3.15.1: donor pool for the global guard. The lattice reservoir
        # above is only filled when ``lattice_rebuild_interval > 0`` and it
        # stores the RFF sketch (k floats/row) -- far too heavy to keep
        # enabled by default. The guard needs a different thing anyway: a
        # bounded, UNIFORM sample of the RAW feature rows of the WHOLE
        # stream, from which it fits its own K-Means donors. So it gets its
        # own reservoir:
        #   * raw X only (no Z/U/score) -- the guard re-normalises the rows
        #     itself through the frozen projector statistics;
        #   * its own RNG stream (``self._donor_rng``) so consuming it does
        #     NOT perturb ``self.rng``: every published artifact stays
        #     bit-for-bit identical (the guard is opt-in and off by default,
        #     so the default path draws exactly as before);
        #   * Vitter's algorithm R, so every row of the stream has the same
        #     chance of being a donor -- a late rare regime is represented
        #     exactly like an early one.
        self._donor_X = []
        self._donor_y = []
        # Stream index of each retained donor row: lets a caller (and
        # the test-suite) CHECK that the pool spans the whole stream
        # instead of trusting that it does.
        self._donor_idx = []
        self._donor_rows = 0
        self._donor_seen = 0
        self._donor_cap = int(self.max_init_buffer)
        self._donor_rng = np.random.default_rng(
            None if random_state is None else int(random_state) + 9973
        )

        self.centers_ZY = None
        self.med_X = None
        self.med_Y = None
        self.med_U = None
        self.counts = None

        self.crystal_X = None
        self.crystal_Y = None
        self.crystal_U = None
        self.gamma_weights = None

        # v3.13.0: robust residual statistics for the global-donor guard's
        # residual-aware donor scores. Fitted by fit_residual_stats() / lazily
        # inside build_global_guard(); None until then.
        self._resid_mean = None
        self._resid_std = None

        # v3.15.1: guard bookkeeping. ``_guard_applied`` is set only
        # after the crystal has actually been guarded, so a second
        # save_hologram() does not silently re-run the guard (it would
        # replace previously-guarded rows with fresh donors).
        self._guard_applied = False
        self._guard_report = None
        # v3.15.1: median-vs-mean skew of the raw targets, captured when
        # the lattice is built (see _build_phase_manifold_lattice). None
        # means "not measurable" -- no hint is emitted in that case.
        self._warm_y_skew = None

        # v3.13.1: reuse ONE Cholesky factorisation per batch. Every batch
        # changes Psi_sum, so the factor is rebuilt lazily at most once per
        # batch and shared by the leverage scores and the ridge head
        # (previously partial_fit factorised Psi_hat twice per batch:
        # O(B k^3) -> O(B k^2) for the whole-stream fit).
        self._chol_cache = None  # (n_seen_at_factorisation, cho_factor)

    def _chol_psi(self, Psi_hat: np.ndarray):
        """Cholesky of ``Psi_hat + beta_low * I`` with a per-batch cache.

        Keyed on ``n_seen``: the accumulators only change inside
        :meth:`partial_fit`, so a factorisation computed at row count N stays
        valid for every solve at the same N (the leverage pass and the ridge
        head read exactly the same ``Psi_hat``). External state mutations
        (direct ``Psi_sum`` writes, restore-from-state) must call
        :meth:`_invalidate_chol` or reset ``n_seen``.
        """
        if self._chol_cache is not None and self._chol_cache[0] == self.n_seen:
            return self._chol_cache[1]
        I_k = np.eye(self.k, dtype=np.float64)
        factor = linalg.cho_factor(Psi_hat + self.beta_low * I_k, lower=True)
        self._chol_cache = (int(self.n_seen), factor)
        return factor

    def _invalidate_chol(self) -> None:
        """Drop the cached Cholesky factor (call after touching Psi_sum directly)."""
        self._chol_cache = None

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
        if self._rff_scale is not None:
            self.Omega = self.rng.normal(0, self._rff_scale * scale_d, size=(d, self.k))
        else:
            k_half = self.k // 2
            omega_low = self.rng.normal(0, self.omega_low_scale * scale_d, size=(d, k_half))
            omega_high = self.rng.normal(0, self.omega_high_scale * scale_d, size=(d, self.k - k_half))
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
        n_select = int(n_select)
        if n_select <= 0:
            # Zero-budget band (e.g. high_band_ratio=0 or 1): seeding nothing
            # here is the contract, otherwise one stray centre is returned and
            # the lattice grows past the requested crystal_size.
            return ZY_cand[:0].copy()
        n_total = len(ZY_cand)
        if n_total == 0:
            # Degenerate split (every row in the same trigger class): there is
            # nothing to seed here; the caller rebalances the centre budget.
            return ZY_cand[:0].copy()
        n_select = min(n_select, n_total)
        first_idx = int(self.rng.integers(0, n_total))
        selected = [first_idx]
        min_sq_dists = np.sum((ZY_cand - ZY_cand[first_idx]) ** 2, axis=1)

        for _ in range(1, n_select):
            total = float(np.sum(min_sq_dists))
            if not np.isfinite(total) or total <= 0:
                # All candidates coincide with a chosen centre (degenerate or
                # constant data): fall back to uniform sampling.
                next_idx = int(self.rng.integers(0, n_total))
            else:
                probs = min_sq_dists / total
                next_idx = int(self.rng.choice(n_total, p=probs))
            selected.append(next_idx)
            new_dists = np.sum((ZY_cand - ZY_cand[next_idx]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, new_dists)

        return ZY_cand[selected].copy()

    def _build_phase_manifold_lattice(self):
        # v3.15.1 (Banner fix 4): capture the target skew HERE, while
        # init_Y still holds the RAW warm-up targets. The lattice build
        # empties init_Y at the end of this method, and it usually runs
        # inside partial_fit (not finalize_crystal), so this is the only
        # point where the raw distribution is still available.
        self._warm_y_skew = None
        if self.init_Y:
            try:
                _yv = np.concatenate([np.asarray(iy).ravel() for iy in self.init_Y])
                if _yv.size >= 8:
                    _sd = float(np.std(_yv))
                    if _sd > 1e-12:
                        _med = float(np.median(_yv))
                        _mean = float(np.mean(_yv))
                        self._warm_y_skew = (_med - _mean) / _sd
            except Exception:
                self._warm_y_skew = None
        Z_all = np.vstack(self.init_Z)
        X_all = np.vstack(self.init_X)
        Y_all = np.vstack(self.init_Y)
        U_all = np.vstack(self.init_U)
        sc_all = np.concatenate(self.init_score)

        ZY_all = self._make_phase_coords(Z_all, Y_all)

        thr = np.percentile(sc_all, self.tail_q)
        is_high = sc_all >= thr
        if self.critical_threshold is not None and self._mode != "label_free":
            y_flat = Y_all.reshape(-1)
            is_high = y_flat >= self.critical_threshold
            frac = float(np.mean(is_high))
            if self.auto_band_ratio:
                ratio = float(np.clip(2.0 * frac, 0.10, 0.30)) if frac > 0 else 0.0
                m_high = int(round(self.M * ratio))
            else:
                m_high = self.M_high
            m_high = max(0, min(self.M, m_high))
            m_low = self.M - m_high
        else:
            m_low, m_high = self.M_low, self.M_high

        # 1. Seed centers in phase space (Z, kappa*y): phase-peak centers are
        #    separated along the y-axis and cannot be absorbed by the background.
        #    If one class is empty (degenerate trigger, e.g. constant targets),
        #    its centre budget is reallocated to the non-empty class instead of
        #    crashing inside k-means++ seeding.
        ZY_low = ZY_all[~is_high]
        ZY_high = ZY_all[is_high]
        if len(ZY_low) == 0:
            m_low, m_high = 0, m_low + m_high
        elif len(ZY_high) == 0:
            m_low, m_high = m_low + m_high, 0
        c_low = self._seed_kmeans_pp(ZY_low, m_low)
        c_high = self._seed_kmeans_pp(ZY_high, m_high)
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

    def _rebuild_lattice_from_reservoir(self):
        """Re-seed the lattice from the retained reservoir.

        The reservoir holds a bounded sample of the whole stream (not just the
        warm-up window), so a rare regime that appeared late gets its own
        Voronoi cells. The reservoir's phase scores are re-used as-is: they
        were computed against the same trigger definition, so the percentile
        split stays meaningful.
        """
        if not self.res_bg_X and not self.res_peak_X:
            return False
        parts_Z, parts_X, parts_Y, parts_U, parts_score = [], [], [], [], []
        for store in ("peak", "bg"):
            Zs = getattr(self, f"res_{store}_Z")
            Xs = getattr(self, f"res_{store}_X")
            if not Xs:
                continue
            parts_Z.append(np.vstack(Zs))
            parts_X.append(np.vstack(Xs))
            parts_Y.append(np.vstack(getattr(self, f"res_{store}_Y")))
            parts_U.append(np.vstack(getattr(self, f"res_{store}_U")))
            parts_score.append(np.concatenate(getattr(self, f"res_{store}_score")))
        self.init_Z = [np.vstack(parts_Z)]
        self.init_X = [np.vstack(parts_X)]
        self.init_Y = [np.vstack(parts_Y)]
        self.init_U = [np.vstack(parts_U)]
        self.init_score = [np.concatenate(parts_score)]
        self._build_phase_manifold_lattice()
        self._rows_since_rebuild = 0
        return True

    def _reservoir_add(self, Z_B, X_B, y_B, U_B, score_B):
        """Append a batch to the bounded reservoir.

        The reservoir is split in two strata so that the retained sample
        represents the WHOLE stream rather than only its most recent rows:

        * the *peak* stratum keeps rows whose standardised target is above
          ``peak_z_threshold`` (the critical / rare regime), sampled
          UNIFORMLY over the whole stream: while the stratum is not full
          every peak row is kept, and once it is full each new peak row
          replaces a uniformly random retained slot (Vitter's algorithm R).
          v3.12.0: the stratum used to keep the FIRST ``peak_cap`` peaks and
          silently drop every later one on the assumption that peaks are
          rare -- but a target drift that pushes most of the stream above
          the trigger fills the stratum with the FIRST drifted batches, and
          the late part of the drift, exactly what a rebuild exists to
          cover, was gone (measured: 544 of 6,000 drifted rows retained);
        * the *background* stratum keeps the most recent rows up to its cap.

        This is what makes a lattice rebuild able to cover late regimes while
        memory stays bounded by ``max_init_buffer`` rows.
        """
        cap = int(self.max_init_buffer)
        peak_cap = max(1, int(cap * 0.4))
        bg_cap = max(1, cap - peak_cap)

        y_B = np.asarray(y_B, dtype=np.float64)
        y_flat = y_B.reshape(-1)
        y_z = np.abs((y_flat - self.y_mean) / self.y_std) if len(y_flat) else np.zeros(0)
        is_peak = y_z > float(self.peak_z_threshold)
        if len(y_flat) != len(X_B):
            is_peak = np.zeros(len(X_B), dtype=bool)

        for mask, store in ((is_peak, "peak"), (~is_peak, "bg")):
            if not np.any(mask):
                continue
            idx = np.nonzero(mask)[0]
            if store == "peak":
                self._res_peak_seen += len(idx)
                room = peak_cap - self._res_peak_rows
                fill = idx[:room] if room > 0 else idx[:0]
                overflow = idx[len(fill):]
                if len(fill):
                    self.res_peak_Z.append(np.array(Z_B[fill], dtype=np.float64, copy=True))
                    self.res_peak_X.append(np.array(X_B[fill], dtype=np.float64, copy=True))
                    self.res_peak_Y.append(np.array(y_B[fill], dtype=np.float64, copy=True))
                    self.res_peak_U.append(np.array(U_B[fill], dtype=np.float64, copy=True))
                    self.res_peak_score.append(np.array(score_B[fill], dtype=np.float64, copy=True))
                    self._res_peak_rows += len(fill)
                if len(overflow):
                    # Algorithm R (Vitter 1985): peak row t replaces retained
                    # slot j ~ U[0, t) when j < peak_cap, so every peak row --
                    # early or late -- keeps inclusion probability
                    # peak_cap / n_peaks_seen.
                    t_arr = np.arange(self._res_peak_seen - len(overflow) + 1,
                                      self._res_peak_seen + 1)
                    j_arr = (self.rng.random(len(overflow)) * t_arr).astype(np.int64)
                    for r_i, j in enumerate(j_arr):
                        if j < peak_cap:
                            self._res_peak_replace(
                                int(j),
                                Z_B[overflow[r_i]], X_B[overflow[r_i]], y_B[overflow[r_i]],
                                U_B[overflow[r_i]], score_B[overflow[r_i]],
                            )
            else:
                room = bg_cap - self._res_bg_rows
                if room <= 0:
                    # Background rows are dropped from the front: the reservoir
                    # keeps the most recent background, and the peak stratum
                    # carries the long-tail coverage.
                    while self._res_bg_rows > 0 and self._res_bg_rows + len(idx) > bg_cap:
                        self._res_bg_rows -= len(self.res_bg_X[0])
                        self.res_bg_Z.pop(0); self.res_bg_X.pop(0)
                        self.res_bg_Y.pop(0); self.res_bg_U.pop(0)
                        self.res_bg_score.pop(0)
                    room = bg_cap - self._res_bg_rows
                    if room <= 0:
                        continue
                    idx = idx[:room]
                self.res_bg_Z.append(np.array(Z_B[idx], dtype=np.float64, copy=True))
                self.res_bg_X.append(np.array(X_B[idx], dtype=np.float64, copy=True))
                self.res_bg_Y.append(np.array(y_B[idx], dtype=np.float64, copy=True))
                self.res_bg_U.append(np.array(U_B[idx], dtype=np.float64, copy=True))
                self.res_bg_score.append(np.array(score_B[idx], dtype=np.float64, copy=True))
                self._res_bg_rows += len(idx)
        self._reservoir_rows = self._res_peak_rows + self._res_bg_rows

    def _res_peak_replace(self, j, z_row, x_row, y_row, u_row, s_row):
        """Overwrite flat peak-stratum slot ``j`` (v3.12.0 algorithm R).

        The peak stratum is stored as per-batch arrays; slot ``j`` indexes
        the concatenation of those batches in insertion order.
        """
        c = 0
        for bi, arr in enumerate(self.res_peak_Z):
            n = len(arr)
            if j < c + n:
                slot = j - c
                self.res_peak_Z[bi][slot] = z_row
                self.res_peak_X[bi][slot] = x_row
                self.res_peak_Y[bi][slot] = y_row
                self.res_peak_U[bi][slot] = u_row
                self.res_peak_score[bi][slot] = s_row
                return
            c += n

    # --- public API -----------------------------------------------------------

    def _donor_add(self, X_B, y_B):
        """Offer a batch to the guard's donor reservoir (algorithm R).

        Uniform over the WHOLE stream: each arriving row replaces a
        uniformly random retained slot with probability cap/seen, so every
        row of the stream keeps the same chance of being a donor. The pool
        stores RAW X only -- the guard re-normalises through the frozen
        projector statistics when it fits its donors, so nothing else is
        needed and the memory cost stays O(cap * d).

        Uses ``self._donor_rng`` (a private stream) so the main ``self.rng``
        sequence is untouched: the default path is bit-for-bit unchanged.
        """
        X_B = np.asarray(X_B, dtype=np.float64)
        if X_B.ndim == 1:
            X_B = X_B.reshape(1, -1)
        y_B = np.asarray(y_B, dtype=np.float64).reshape(-1)
        if len(y_B) != len(X_B):
            return
        cap = int(self._donor_cap)
        if cap <= 0:
            return
        base = int(self._donor_seen)
        for i in range(len(X_B)):
            self._donor_seen += 1
            if self._donor_rows < cap:
                self._donor_X.append(X_B[i].copy())
                self._donor_y.append(np.array([y_B[i]], dtype=np.float64))
                self._donor_idx.append(base + i)
                self._donor_rows += 1
                continue
            # Algorithm R: the t-th arriving row replaces a uniformly
            # random retained slot with probability cap/t.
            j = int(self._donor_rng.integers(0, self._donor_seen))
            if j < cap:
                self._donor_X[j] = X_B[i].copy()
                self._donor_y[j] = np.array([y_B[i]], dtype=np.float64)
                self._donor_idx[j] = base + i

    def _guard_pool(self):
        """Collect the representative pool for the global-donor guard.

        The pool is the guard's own donor reservoir: a bounded UNIFORM
        sample of the raw feature rows of the whole stream (see
        ``_donor_add``). It is deliberately independent of the lattice
        reservoir, which is only filled when
        ``lattice_rebuild_interval > 0`` and which stores the RFF sketch
        rather than the raw rows the guard needs.

        Returns ``(X_pool, y_pool)`` or ``None`` when nothing is held.
        """
        if self._donor_X:
            return np.vstack(self._donor_X), np.vstack(self._donor_y).reshape(-1)
        # Fallback for a distiller whose donor reservoir was never fed
        # (the donor pool is only filled when ``global_guard=True``):
        # the warm-up buffer plus the lattice reservoir. This is the
        # pre-v3.15.1 pool and it is what a manual
        # ``build_global_guard()`` call on a ``lattice_rebuild_interval>0``
        # distiller still uses.
        parts_X = list(self.init_X) + list(self.res_bg_X) + list(self.res_peak_X)
        parts_Y = list(self.init_Y) + list(self.res_bg_Y) + list(self.res_peak_Y)
        if not parts_X:
            return None
        return np.vstack(parts_X), np.vstack(parts_Y)

    def _pool_predict(self, X_pool: np.ndarray) -> np.ndarray:
        """Map a pool through the crystal's ridge map (batched to bound memory).

        The predictor is the same ridge map the crystal's own phase score is
        built from: kernel ridge in the RFF sketch with the exported
        (blended) ``gamma_weights`` as the kernel weights. Chunks of 8192 rows
        keep the (chunk x k) sketch at a few MB even for a multi-million-row
        pool.
        """
        w_prob = self.gamma_weights / float(np.sum(self.gamma_weights))
        G = (self.crystal_U.T * w_prob) @ self.crystal_U
        A = G + self.fista_l2_reg * np.eye(G.shape[0])
        B = (self.crystal_U.T * w_prob) @ self.crystal_Y[:, None]
        W = linalg.cho_solve(linalg.cho_factor(A, lower=True), B)
        W = W.reshape(G.shape[0])  # (k,1) -> (k,)
        out = np.empty((len(X_pool), 1), dtype=np.float64)
        step = 8192
        for lo in range(0, len(X_pool), step):
            _Z, U_c = self._spectral_transform(X_pool[lo:lo + step], update_state=False)
            out[lo:lo + step] = (U_c @ W)[:, None]
        return out

    def fit_residual_stats(self, X_pool=None, y_pool=None) -> dict:
        """Fit robust residual statistics for the guard's residual-aware scores.

        The guard scores donor rows by the crystal's ridge residual
        ``r(x) = y - f(x)`` (residual-aware, v3.13.0): rows the crystal's own
        model already predicts well add little new information, so their
        guard weight is damped. This method fits the residual scale on a
        representative pool -- by default everything the distiller holds
        (warm-up buffer + reservoir); the caller may pass an explicit pool.
        The statistic is the robust median and MAD of the residual, so one
        wild residual cannot blow up the scale.

        Must be called AFTER the crystal exists (after ``finalize_crystal`` /
        ``build_global_guard``): the ridge map reads the crystal's own
        predictor.
        """
        if self.crystal_X is None:
            raise ValueError(
                "fit_residual_stats() needs the crystal; call finalize_crystal() "
                "or build_global_guard() first"
            )
        if X_pool is None:
            pool = self._guard_pool()
            if pool is None:
                raise ValueError(
                    "no buffered/reservoir rows available; the lattice has already "
                    "consumed them -- pass an explicit pool: "
                    "fit_residual_stats(X_pool, y_pool)"
                )
            X_pool, y_pool = pool
        X_pool = np.asarray(X_pool, dtype=np.float64)
        if X_pool.ndim == 1:
            X_pool = X_pool.reshape(1, -1)
        y_pool = np.asarray(y_pool, dtype=np.float64).reshape(-1)
        if X_pool.ndim != 2 or X_pool.shape[0] == 0:
            raise ValueError("X_pool must be a non-empty 2-D array")
        if len(y_pool) != len(X_pool):
            raise ValueError(
                f"y_pool has {len(y_pool)} labels for {len(X_pool)} rows; they must match"
            )
        fX = self._pool_predict(X_pool).reshape(-1)
        resid = y_pool - fX
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-8
        self._resid_mean, self._resid_std = med, mad
        return {
            "n_pool": int(len(X_pool)),
            "resid_median": med,
            "resid_mad": mad,
        }

    def build_global_guard(
        self,
        n_global: int = 150,
        weight: float = 0.35,
        X_pool=None,
        y_pool=None,
        n_lloyd: int = 6,
        random_state: int | None = None,
    ) -> dict:
        """Replace the lowest-weight crystal rows with global K-Means medoids.

        Root cause (validated live on PM2.5, 5 seeds, ``tools/proto_swap.py``):
        the crystal's rows are ranked by Voronoi cell counts, so the exported
        head-slice donor pool is dominated by the background the lattice
        happened to keep -- a global model trained on the export loses to
        plain K-Means on the SAME budget (measured: median all-RMSE 88.3
        swap150 vs 90.0 kmeans, crit-RMSE 230 vs 236 -- the swap wins BOTH
        metrics 5/5 seeds, and larger guards help ``all`` monotonically
        without hurting ``crit``).

        This is the library-side fix: swap the ``n_global`` lowest-weight
        crystal rows for K-Means medoids of a representative pool (the
        warm-up buffer + the bounded reservoir, or an explicit pool). Donors
        are seeded and relaxed in the SAME standardized space the exported
        crystal is consumed in -- NOT the distiller's internal RFF input
        space (``dsketch50`` measured worse: the sketch metric distortion is
        part of the root cause).

        Donor weights are cell counts renormalised to mean one, damped by
        ``weight`` (0 = pure background guard, 1 = as heavy as an average
        crystal row), then multiplied by a residual-aware factor: donors the
        crystal's own ridge map already predicts within the pool's robust
        residual scale add little, so their weight is damped toward the
        floor. The residual scale is fitted on the same pool
        (``fit_residual_stats``); the crystal's rows are left untouched.

        Parameters
        ----------
        n_global:
            How many lowest-weight rows to replace (default 150; PM2.5
            measured monotone ``all`` gains up to 150 without a ``crit`` cost).
        weight:
            Base weight of the donors relative to a mean crystal row.
        X_pool, y_pool:
            Optional explicit pool; default is everything the distiller still
            holds (see ``_guard_pool``). Stream with
            ``lattice_rebuild_interval > 0`` so the reservoir covers the whole
            stream, or pass the training slice explicitly for a
            batch-distilled crystal.
        n_lloyd:
            Lloyd relaxation steps after k-means++ seeding.
        random_state:
            Optional explicit seed; default reuses the distiller's RNG state.

        Not supported for ``include_deltas=True`` crystals: the swapped
        rows' sketches would be recomputed on distorted delta pairs, so the
        method raises ``NotImplementedError`` instead of silently distorting
        the ridge map and the spectral diagnostics. Use a non-delta crystal
        for the guard.

        Returns a report dict (swapped, pool size, weight diagnostics). The
        crystal is mutated in place; call ``save_hologram`` to persist.
        """
        if self.crystal_X is None:
            raise ValueError(
                "build_global_guard() called before finalize_crystal(); "
                "the crystal has no rows to guard"
            )
        if self.include_deltas:
            raise NotImplementedError(
                "build_global_guard() is undefined for include_deltas=True "
                "crystals: the swapped rows' sketches are recomputed via "
                "_spectral_transform(X_g), which pairs each medoid's delta with "
                "the PREVIOUS ROW OF X_g -- an unrelated medoid in k-means order "
                "(and the stale _prev_X_row for the first one). The ridge map and "
                "the spectral diagnostics would be built on distorted delta "
                "pairs. Use a non-delta crystal for the guard."
            )
        n_global = int(n_global)
        if n_global <= 0:
            return {"swapped": 0, "n_pool": 0, "note": "n_global <= 0: no-op"}
        if n_global > int(len(self.crystal_X)):
            raise ValueError(
                f"n_global={n_global} exceeds crystal size "
                f"{int(len(self.crystal_X))}; cannot swap more rows than the "
                "crystal holds"
            )
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        n_lloyd = int(n_lloyd)
        if n_lloyd < 0:
            raise ValueError(f"n_lloyd must be >= 0, got {n_lloyd}")
        if X_pool is None:
            pool = self._guard_pool()
            if pool is None:
                raise ValueError(
                    "no buffered/reservoir rows available for the guard pool; "
                    "pass X_pool/y_pool explicitly (e.g. the training slice)"
                )
            X_pool, y_pool = pool
            # v3.15.1 (Banner fix 1, REWRITTEN): the original check was
            # unreachable -- it keyed on ``self.init_X`` being non-empty,
            # but the lattice build clears ``init_X``, so by the time the
            # guard runs the warm-up buffer is always gone and the branch
            # never fired. The real defect it tried to catch is that the
            # guard had NO donor rows at all under the shipped default
            # (``lattice_rebuild_interval=0``), which made
            # ``save_hologram()`` raise. That is fixed at the source: the
            # guard now owns a donor reservoir fed on every batch (see
            # ``_donor_add``), so the pool covers the whole stream by
            # construction.
            #
            # What is still worth checking is HONESTY: the donors must
            # represent the stream the caller thinks they do. The pool is
            # uniform over the rows offered to it, so the only stale case
            # left is a pool that covers fewer rows than the stream
            # actually seen -- i.e. the guard would silently fit its
            # donors on a prefix of the data.
            if self._donor_X and int(self._donor_seen) < int(self.n_seen):
                raise ValueError(
                    "build_global_guard() would use a stale donor pool: "
                    f"the donor reservoir saw {int(self._donor_seen)} of "
                    f"{int(self.n_seen)} streamed rows. Pass "
                    "X_pool/y_pool explicitly (e.g. the training slice)."
                )
        X_pool = np.asarray(X_pool, dtype=np.float64)
        if X_pool.ndim == 1:
            X_pool = X_pool.reshape(1, -1)
        y_pool = np.asarray(y_pool, dtype=np.float64).reshape(-1)
        if X_pool.ndim != 2 or X_pool.shape[0] == 0:
            raise ValueError("X_pool must be a non-empty 2-D array")
        if len(y_pool) != len(X_pool):
            raise ValueError(
                f"y_pool has {len(y_pool)} labels for {len(X_pool)} rows; they must match"
            )
        if X_pool.shape[1] != self.crystal_X.shape[1]:
            raise ValueError(
                f"pool has {X_pool.shape[1]} features but the crystal stores "
                f"{self.crystal_X.shape[1]}; pass raw-feature rows"
            )

        n_sw = min(n_global, len(X_pool))
        rng = self.rng if random_state is None else np.random.default_rng(random_state)
        rng_prev = self.rng
        if random_state is not None:
            self.rng = rng
        try:
            # Donors live in the crystal's standardized (raw-feature) space.
            Z_pool = self._normalize_X(X_pool, update_state=False)
            centers = self._seed_kmeans_pp(Z_pool, n_sw)
            for _ in range(n_lloyd):
                d_mat = cdist(Z_pool, centers, metric="sqeuclidean")
                labels = np.argmin(d_mat, axis=1)
                for c_i in range(len(centers)):
                    mask = labels == c_i
                    if np.any(mask):
                        centers[c_i] = np.mean(Z_pool[mask], axis=0)
            d_snap = cdist(centers, Z_pool, metric="sqeuclidean")
            med_idx = np.argmin(d_snap, axis=1)
            labels = np.argmin(cdist(Z_pool, centers, metric="sqeuclidean"), axis=1)
        finally:
            if random_state is not None:
                self.rng = rng_prev
        counts = np.bincount(labels, minlength=len(centers)).astype(np.float64) + 1.0
        X_g = X_pool[med_idx].copy()
        y_g = y_pool[med_idx].copy()

        # v3.15.1 (Banner fix 2): the guard flag is set at the END of this
        # function (after the crystal is actually mutated), so a failure
        # halfway through cannot leave the distiller claiming a guard that
        # was never applied. The manual and save_hologram paths share one
        # flag: a guard applied by either path is not re-run by the other.
        # Residual-aware donor scores against the crystal's own ridge map.
        if self._resid_mean is None:
            self.fit_residual_stats(X_pool, y_pool)
        fX = self._pool_predict(X_g).reshape(-1)
        resid_z = np.abs((y_g - fX) - self._resid_mean) / self._resid_std
        zbar = float(np.mean(resid_z))
        damp = 1.0 / (1.0 + zbar)
        w_g = (counts / np.mean(counts)) * weight * damp

        order = np.argsort(self.gamma_weights)[:n_sw]
        old_w = self.gamma_weights[order].copy()
        swapped_mask = old_w > w_g
        self.crystal_X[order] = X_g
        self.crystal_Y[order] = y_g
        self.gamma_weights[order] = w_g
        # crystal_U feeds _pool_predict's ridge map and the spectral passport
        # diagnostics; the swapped rows' sketches must match their new rows.
        _Zg, U_g = self._spectral_transform(X_g, update_state=False)
        self.crystal_U[order] = U_g
        report = {
            "swapped": int(n_sw),
            "n_pool": int(len(X_pool)),
            "weight": weight,
            "resid_z_mean": zbar,
            "damp": float(damp),
            "n_low_weight_reordered": int(np.sum(swapped_mask)),
            "min_old_weight": float(np.min(old_w)),
            "guard_weight_mean": float(np.mean(w_g)),
        }
        # The crystal HAS been mutated by now, so the flag is honest. It is
        # set here rather than earlier so an exception above cannot leave a
        # half-guarded crystal claiming a completed guard.
        self._guard_applied = True
        self._guard_report = report
        return report

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

        Input validation: the batch must be a non-empty 2-D ``float`` array
        whose width matches the width of the first batch; ``y_batch`` must be
        ``None`` or have exactly ``len(X_batch)`` entries. Non-finite entries
        raise ``ValueError`` (pass ``on_nonfinite="drop"`` at construction to
        drop affected rows instead, as an industrial convenience). An empty
        batch is a valid no-op so that idling sensor windows do not crash a
        running stream.
        """
        X_batch = np.asarray(X_batch, dtype=np.float64)
        if X_batch.ndim == 1:
            X_batch = X_batch.reshape(1, -1)
        if X_batch.ndim != 2:
            raise ValueError(f"X_batch must be 2-D (batch, features), got shape {X_batch.shape}")
        B = X_batch.shape[0]

        if self.d_in is not None and X_batch.shape[1] > 0 and X_batch.shape[1] != self.d_in:
            raise ValueError(
                f"X_batch has {X_batch.shape[1]} features but the stream was opened with "
                f"{self.d_in}; feature width cannot change mid-stream"
            )

        if y_batch is not None:
            y_batch = np.asarray(y_batch, dtype=np.float64)
            if y_batch.ndim == 2 and y_batch.shape[1] != 1:
                raise ValueError(
                    f"y_batch must be 1-D or (B, 1), got shape {y_batch.shape}; "
                    "multi-target y is not supported"
                )
            if y_batch.ndim == 2 and y_batch.shape[1] == 1:
                y_batch = y_batch.reshape(-1, 1)
            if y_batch.ndim not in (1, 2):
                raise ValueError(f"y_batch must be 1-D or (B, 1), got shape {y_batch.shape}")
            if len(y_batch) != B:
                raise ValueError(
                    f"X_batch has {B} rows but y_batch has {len(y_batch)}; they must match"
                )

        if B == 0:
            return self  # empty window: valid no-op

        finite_mask = np.isfinite(X_batch).all(axis=1)
        if y_batch is not None:
            y_flat = y_batch.reshape(len(y_batch), -1)
            finite_mask &= np.isfinite(y_flat).all(axis=1)
        n_bad = int(np.sum(~finite_mask))
        if n_bad:
            if self.on_nonfinite == "drop":
                keep = finite_mask
                if not np.any(keep):
                    return self
                X_batch = X_batch[keep]
                if y_batch is not None:
                    y_batch = y_batch[keep]
                B = X_batch.shape[0]
            else:
                bad_cols = np.where(~np.isfinite(X_batch).all(axis=0))[0].tolist()
                raise ValueError(
                    f"{n_bad} of {B} rows contain NaN/Inf"
                    + (f" (feature columns {bad_cols})" if bad_cols else " (in y_batch)")
                    + "; clean the stream or pass on_nonfinite='drop' at construction"
                )

        mode = "label_free" if y_batch is None else "supervised"
        if self._mode is None:
            self._mode = mode
        elif self._mode != mode:
            raise ValueError(
                "partial_fit() cannot switch between supervised (y_batch provided) and "
                f"label-free (y_batch=None) mode after the first batch; this stream is "
                f"already in {self._mode!r} mode. Create a new SpectralHologramDistiller "
                "for the other mode."
            )

        if self.d_in is None and X_batch.shape[1] > 0:
            self.d_in = int(X_batch.shape[1])

        y_is_synthetic = False

        if y_batch is None:
            self.unsupervised_mode = True
            y_is_synthetic = True
            if not self._proj_initialized:
                X_eff0 = self._augment_deltas(X_batch, update_state=False)
                d = X_eff0.shape[1]
                self.Phi_sum = np.zeros((self.k, 1), dtype=np.float64)
                self.X_mean = np.mean(X_eff0, axis=0)
                self.X_std = np.std(X_eff0, axis=0) + 1e-8
                scale_d = 1.0 / np.sqrt(max(d, 1))
                if self._rff_scale is not None:
                    self.Omega = self.rng.normal(0, self._rff_scale * scale_d, size=(d, self.k))
                else:
                    k_half = self.k // 2
                    omega_low = self.rng.normal(0, self.omega_low_scale * scale_d, size=(d, k_half))
                    omega_high = self.rng.normal(0, self.omega_high_scale * scale_d, size=(d, self.k - k_half))
                    self.Omega = np.hstack([omega_low, omega_high])
                self.phase = self.rng.uniform(0, 2 * np.pi, size=(1, self.k))
                Z_B, U_B = self._spectral_transform(X_batch, update_state=True)
                self.Psi_sum += U_B.T @ U_B
                self.n_seen += B
                Psi_hat = self.Psi_sum / self.n_seen
                L_low = self._chol_psi(Psi_hat)
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
                L_low = self._chol_psi(Psi_hat)
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

            L_low = self._chol_psi(Psi_hat)
            W_pred = linalg.cho_solve(L_low, Phi_hat)

        # Background accumulator for the passport reference. The phase score is
        # only known after the residual is computed, so this runs below the
        # score computation (see _accumulate_background).
        # Adaptive y-statistics (opt-in): once the lattice exists, keep the
        # kappa*y phase coordinate calibrated under target drift. During the
        # warm-up the frozen first-batch statistics are kept, so buffered phase
        # scores remain comparable with the percentile cut at lattice build.
        if self.adaptive_y_stats and self.lattice_ready:
            old_mean, old_std = self.y_mean, self.y_std
            y_flat = np.asarray(y_batch, dtype=np.float64).reshape(-1)
            lr_stat = min(0.10, len(y_flat) / max(self.n_seen, 1))
            self.y_mean = (1.0 - lr_stat) * self.y_mean + lr_stat * float(np.mean(y_flat))
            var_upd = (1.0 - lr_stat) * (self.y_std ** 2) + lr_stat * float(np.var(y_flat))
            self.y_std = float(np.sqrt(max(var_upd, 1e-12)))
            # Rebase the stored centres' y-coordinate onto the *new* statistics.
            # Without this the centres keep living in the old (y_mean, y_std)
            # coordinate system while every incoming row is projected with the
            # updated one, so assignments silently mix two different scales.
            # c_new = kappa * (y - mean_new) / std_new with
            # y = c_old * std_old / kappa + mean_old, applied to every centre.
            if self.centers_ZY is not None and self.centers_ZY.shape[1] > 0:
                if old_std > 0.0 and self.y_std > 0.0:
                    ratio = old_std / self.y_std
                    shift = self.kappa * (old_mean - self.y_mean) / self.y_std
                    self.centers_ZY[:, -1] = self.centers_ZY[:, -1] * ratio + shift

        residual = np.sum((y_batch - U_B @ W_pred) ** 2, axis=1)
        res_ratio = residual / (np.median(residual) + 1e-8)
        y_norm = (y_batch.squeeze() - self.y_mean) / self.y_std
        tail_eff = "high" if y_is_synthetic else self.tail
        if tail_eff == "high":
            y_z = np.maximum(0.0, y_norm)
        elif tail_eff == "low":
            y_z = np.maximum(0.0, -y_norm)
        else:  # "both"
            y_z = np.abs(y_norm)
        # np.atleast_1d: a single-row batch squeezes y_z to a 0-d array, which
        # would later crash np.concatenate on the buffered phase scores.
        res_w = self.residual_weight if self.use_residual else 0.0
        phase_score = np.atleast_1d((res_w * res_ratio) + 2.0 * (y_z ** 2))

        # Background accumulator for the passport reference: keep the sketch
        # rows that did NOT trigger the phase-peak score. The cut is the same
        # percentile the lattice uses, so "background" means the same thing in
        # both places. Rows are classified per batch; a batch whose median
        # score already sits above the cut contributes nothing, which is the
        # safe behaviour (a wholly shifted batch must not become the
        # reference).
        if self.passport_reference == "background":
            thr_bg = float(np.percentile(phase_score, self.tail_q))
            bg_mask = phase_score < thr_bg
            if np.any(bg_mask):
                self.Psi_bg_sum += U_B[bg_mask].T @ U_B[bg_mask]
                self.n_bg += int(np.sum(bg_mask))

        # Retain a bounded sample of every row for a later lattice rebuild.
        # This is what lets a rare regime that appears LATE in the stream reach
        # the crystal (see lattice_rebuild_interval).
        if self.lattice_rebuild_interval > 0:
            self._reservoir_add(Z_B, X_batch, y_batch, U_B, phase_score)

        # v3.15.1: feed the guard's donor reservoir. It is ALWAYS fed (not
        # only when the lattice rebuild is on) because the guard needs a
        # uniform sample of the whole stream regardless of how the lattice
        # was built. It costs O(cap * d) of raw rows and touches only
        # ``self._donor_rng``, so the published default path is unchanged.
        if self.global_guard:
            self._donor_add(X_batch, y_batch)

        if not self.lattice_ready:
            buffered = int(sum(len(a) for a in self.init_Z))
            cap = int(self.max_init_buffer)
            target_buf = min(cap, max(4500, self.M * 12))
            if buffered + B > cap:
                # Hard cap: the warm-up buffer must never hold more than
                # ``max_init_buffer`` rows. A single oversized batch is split
                # instead of being buffered whole (which used to spike cdist
                # memory at lattice-build time): the leading rows that still fit
                # build the lattice and the remainder is streamed through the
                # ordinary online-update path below, so no data is dropped and
                # the accumulators keep counting every row.
                take = max(cap - buffered, 0)
                if take > 0:
                    self.init_Z.append(Z_B[:take])
                    # Copy caller-owned buffers: the caller may reuse its
                    # streaming buffer in place after partial_fit() returns.
                    self.init_X.append(np.array(X_batch[:take], dtype=np.float64, copy=True))
                    self.init_Y.append(np.array(y_batch[:take], dtype=np.float64, copy=True))
                    self.init_U.append(U_B[:take])
                    self.init_score.append(phase_score[:take])
                self._build_phase_manifold_lattice()
                if take < B:
                    # Stream the not-yet-buffered tail of the batch through the
                    # online update (fall through instead of returning).
                    X_batch = X_batch[take:]
                    y_batch = y_batch[take:]
                    Z_B = Z_B[take:]
                    U_B = U_B[take:]
                else:
                    return self
            else:
                self.init_Z.append(Z_B)
                # Copy caller-owned buffers: the caller may reuse its streaming buffer
                # in place after partial_fit() returns, and the crystal is not built
                # until finalize_crystal().
                self.init_X.append(np.array(X_batch, dtype=np.float64, copy=True))
                self.init_Y.append(np.array(y_batch, dtype=np.float64, copy=True))
                self.init_U.append(U_B)
                self.init_score.append(phase_score)
                if buffered + B >= target_buf:
                    self._build_phase_manifold_lattice()
                return self

        # Online update of the phase-manifold lattice.
        self._rows_since_rebuild += B
        if (
            self.lattice_rebuild_interval > 0
            and self._rows_since_rebuild >= self.lattice_rebuild_interval
        ):
            # The stream has moved on since the lattice was seeded. Re-seed it
            # from the reservoir so late-arriving regimes get their own cells.
            # Only the medoids and their counts are replaced; the spectral
            # accumulators (Psi_sum / Phi_sum) keep covering the whole stream.
            self._rebuild_lattice_from_reservoir()
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

            lr = min(self.center_lr, n_new / new_count)
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
        if self.n_seen == 0 or (not self.lattice_ready and not self.init_X):
            raise ValueError(
                "finalize_crystal() called before any data was absorbed; "
                "call partial_fit() with at least one non-empty batch first"
            )
        # v3.15.1 (Banner fix 4): the skew was captured when the lattice was
        # built (init_Y is empty by now). The old placement read init_Y
        # here, so it always saw an empty buffer and never warned.
        if not self.lattice_ready:
            self._build_phase_manifold_lattice()
        # Read the skew AFTER the lattice build. When the lattice is built
        # INSIDE finalize_crystal (whole stream <= max_init_buffer -- the
        # default buffer is 8192 rows, so this is the COMMON case) the build
        # is what measures the skew. Reading it before the build saw None,
        # so the hint stayed silent on exactly the path that had just
        # measured it. (v3.15.1 re-check.)
        _y_skew = self._warm_y_skew
        # Telemetry-aware tail-direction hint (v3.15.1 Banner fix 4).
        #
        # The default tail='high' silently protects the wrong side on a
        # low-tail task (e.g. pressure/voltage sags). The skew is measured
        # on the RAW warm-up targets, before the lattice build consumed
        # them. med_Y is circular here: the lattice split itself partitions
        # medoids by tail, so med_Y always "looks" opposite-skewed.
        if (not self.unsupervised_mode and self.tail == "high"
                and _y_skew is not None and _y_skew > 0.5):
            import warnings as _w
            _w.warn(
                "finalize_crystal(): the target distribution is left-skewed "
                f"(median {_y_skew:.2f} sd above the mean, measured on the "
                "warm-up targets) but tail='high'. The dense mass sits on "
                "the HIGH side, so the phase band would protect the common "
                "case and miss the rare LOW regime. If critical = low "
                "(e.g. pressure sags), pass tail='low'.",
                UserWarning, stacklevel=2,
            )

        # Honest-contract hint (2026-11-11 audit): interval=0 never rebuilds
        # the lattice, so a stream that outgrew the warm-up window is
        # represented by its prefix only. Warn once at finalize, where the
        # crystal is actually committed; short streams (<= warm-up) are
        # covered whole and stay silent.
        _warm_target = min(self.max_init_buffer, max(4500, self.M * 12))
        if self.lattice_rebuild_interval == 0 and self.n_seen > _warm_target:
            import warnings as _w
            _w.warn(
                "finalize_crystal(): lattice_rebuild_interval=0 and the "
                f"stream ({self.n_seen} rows) outgrew the warm-up window "
                f"({_warm_target} rows) the lattice was built from. The "
                "lattice was never rebuilt, so regimes that appeared after "
                "warm-up have almost no dedicated medoids. For long streams "
                "with drift or late regimes pass lattice_rebuild_interval>0 "
                "(e.g. 1500); 0 is kept only for bit-for-bit reproducibility "
                "of the published artifacts.",
                UserWarning, stacklevel=2,
            )

        order = np.argsort(self.counts)[::-1]
        self.crystal_X = self.med_X[order].copy()
        # reshape(-1), not squeeze(): a single-medoid lattice would otherwise
        # squeeze crystal_Y to a 0-d scalar and breaks save/load, the peak slice
        # and the heal path.
        self.crystal_Y = self.med_Y[order].reshape(-1).copy()
        self.crystal_U = self.med_U[order].copy()
        sorted_counts = self.counts[order].copy()
        M_curr = len(self.crystal_U)

        voronoi_prob = sorted_counts / np.sum(sorted_counts)

        Psi_hat = self.Psi_sum / self.n_seen
        G_U = self.crystal_U @ self.crystal_U.T
        K = (G_U ** 2) + self.fista_l2_reg * np.eye(M_curr)
        b = np.sum((self.crystal_U @ Psi_hat) * self.crystal_U, axis=1) + self.fista_l2_reg * voronoi_prob

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
        # Pure-herding intermediate (the FISTA solution before Voronoi tempering
        # and blending). Exposed under an explicit name for diagnostics: it is
        # NOT what save_hologram() exports.
        self.herding_rel_error = (
            linalg.norm(Psi_crystal - Psi_hat, ord="fro") / linalg.norm(Psi_hat, ord="fro")
        )

        # Root tempering of the Voronoi measure (p ** voronoi_temper): keeps the
        # global RMSE balanced while preserving the amplitude of the phase-peak band.
        tempered_voronoi = (voronoi_prob ** self.voronoi_temper)
        tempered_voronoi /= np.sum(tempered_voronoi)
        w_combined = (self.herding_mix * tempered_voronoi + (1.0 - self.herding_mix) * gamma_prob) * M_curr
        self.gamma_weights = w_combined / np.mean(w_combined)

        # The returned error must describe the crystal that is actually written
        # out: renormalise the exported (blended) ``gamma_weights`` back into a
        # probability vector and measure that reconstruction.
        exported_prob = self.gamma_weights / float(np.sum(self.gamma_weights))
        Psi_exported = (self.crystal_U.T * exported_prob) @ self.crystal_U
        rel_error = linalg.norm(Psi_exported - Psi_hat, ord="fro") / linalg.norm(Psi_hat, ord="fro")
        # v3.15.2: finalize_crystal() REBUILDS crystal_X/Y/U and the herding
        # weights from scratch, so a guard applied by an earlier
        # save_hologram() is GONE -- the donor rows it swapped in were
        # overwritten by the med_* rows. The guard flag must therefore be
        # cleared here. Otherwise the next save_hologram() skips the guard
        # (already applied) and silently writes a NON-guarded crystal
        # while global_guard=True: measured, the saved crystal came out
        # identical to a distiller built without the guard.
        self._guard_applied = False
        self._guard_report = None
        # v3.15.2: the row count this crystal actually describes.
        # save_hologram() compares it with the live n_seen and refuses to
        # write a crystal whose passport/counts cover rows the medoids
        # never saw.
        self._finalized_n_seen = int(self.n_seen)
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

        The ``include_deltas`` flag is persisted so the loader can distinguish a
        raw ``d``-feature delta stream from a plain ``d``-feature crystal. Without
        it a 1-feature window fed to a 2-feature non-delta crystal would be
        silently reinterpreted as a delta stream.

        ``rank_passport`` must be in ``[1, sketch_dim]``. The OOD radar's alert
        threshold is calibrated for the shipped rank of 32, so a different rank
        changes the meaning of ``bw_distance``; the loader records the rank it
        was given so the caller can tell the two apart.
        """
        k = int(self.k)
        rank = int(rank_passport)
        if rank < 1:
            raise ValueError(
                f"rank_passport must be >= 1, got {rank_passport} "
                f"(0 would silently select every eigenpair via eigvals[-0:])"
            )
        if rank > k:
            # A rank above the sketch dimension cannot be represented; the
            # passport simply keeps every available eigenpair. This is not an
            # error, but it does change what bw_distance means, so the effective
            # rank is recorded in the artifact.
            rank = k
        if self.crystal_X is None:
            raise ValueError(
                "save_hologram() called before finalize_crystal(); the crystal "
                "has no medoids to write"
            )
        # v3.15.2: the stream grew after finalize_crystal(). The crystal
        # rows are the medoids picked at finalize time, while the
        # passport (Psi_sum), the per-cell counts and n_seen cover EVERY
        # absorbed row -- including rows no medoid ever saw. Writing that
        # combination ships an artifact whose metadata overstates the data
        # the crystal represents (measured: n_seen=4000 with 0 of the last
        # 2000 rows present in the crystal). Fail loudly instead.
        _fin_n = getattr(self, "_finalized_n_seen", None)
        if _fin_n is not None and int(self.n_seen) != int(_fin_n):
            raise ValueError(
                "save_hologram(): the stream grew after finalize_crystal() "
                f"({int(_fin_n)} -> {int(self.n_seen)} rows). The crystal you "
                "would write still describes the first "
                f"{int(_fin_n)} rows, while its passport and n_seen cover all "
                f"{int(self.n_seen)} -- call finalize_crystal() again to rebuild "
                "the crystal, or use HologramCrystalLoader.heal_domain_shift() "
                "to absorb the new regime."
            )
        if self.gamma_weights is None:
            raise ValueError(
                "save_hologram() called before finalize_crystal(); the crystal "
                "has no herding weights to write"
            )
        if self.global_guard:
            # Global guard is applied at save time so the relative spectral
            # error returned by finalize_crystal() keeps its pre-guard
            # definition and the caller never has to remember a second
            # step. Requires the guard pool: buffered/reservoir rows, or an
            # explicit pool passed through finalize -> save (see
            # build_global_guard). Not supported for include_deltas=True
            # (raises from build_global_guard), the same loud failure as
            # the manual path.
            #
            # v3.15.1 (Banner fix 2): a second save_hologram() must not
            # re-run the guard. The guard swaps the lowest-weight rows, so
            # re-running it on an already-guarded crystal would silently
            # replace previously-guarded rows with fresh donors.
            if not getattr(self, "_guard_applied", False):
                # v3.15.1 (bughunt): global_guard_n is clamped against
                # the REQUESTED crystal_size, but the stream can be
                # shorter than the budget -- the real crystal then holds
                # fewer rows than the guard wants to swap, and
                # save_hologram() died with "n_global=30 exceeds crystal
                # size 20". Clamp against the crystal that exists.
                n_guard = int(self.global_guard_n)
                n_actual = int(len(self.crystal_X))
                if n_guard > n_actual:
                    n_guard = max(0, n_actual // 2)
                    import warnings
                    warnings.warn(
                        f"global_guard_n={self.global_guard_n} exceeds the "
                        f"actual crystal size {n_actual}; clamping to "
                        f"{n_guard} (half the crystal).",
                        UserWarning, stacklevel=2,
                    )
                report = self.build_global_guard(
                    n_global=n_guard,
                    weight=self.global_guard_weight,
                )
                self._guard_applied = True
                self._guard_report = report  # diagnostics: swapped/pool size
            else:
                import warnings
                warnings.warn(
                    "save_hologram(): global guard already applied on a "
                    "previous save; not re-running it. Construct a new "
                    "distiller if you need a fresh guard pass.",
                    UserWarning, stacklevel=2,
                )
        Psi_hat = self.Psi_sum / self.n_seen
        if self.passport_reference == "background" and self.n_bg > 0:
            # The passport describes the BACKGROUND regime. A regime that
            # occupies part of the training data must not become part of the
            # reference, or the radar flags normal windows as shifted.
            Psi_ref = self.Psi_bg_sum / self.n_bg
        else:
            Psi_ref = Psi_hat
        eigvals, eigvecs = linalg.eigh(Psi_ref)
        payload = dict(
            # v3.13.1: the artifact's format contract travels with the file. A
            # loader written for schema 1 must be able to reject (not silently
            # misread) an artifact it was never meant to parse -- the failure
            # mode without a version key is a KeyError three fields later.
            schema_version=np.array([1], dtype=np.int64),
            schema_min_loader=np.array([1], dtype=np.int64),
            shd_version=np.array(SHD_VERSION),
            crystal_X=self.crystal_X.astype(np.float32),
            crystal_Y=self.crystal_Y.astype(np.float32),
            gamma_weights=self.gamma_weights.astype(np.float32),
            passport_eigvals=eigvals[-rank:].astype(np.float16),
            passport_eigvecs=eigvecs[:, -rank:].astype(np.float16),
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64),
            tail=np.array(self.tail),
            include_deltas=np.array(bool(self.include_deltas)),
            high_band_ratio=np.array([float(self.high_band_ratio)], dtype=np.float64),
            passport_reference=np.array(self.passport_reference),
            critical_threshold=np.array(
                [np.nan if self.critical_threshold is None else float(self.critical_threshold)],
                dtype=np.float64,
            ),
            # v3.15.2: the two knobs the README calls co-primary
            # (band width and phase_kappa) travel with the artifact.
            # Without them a crystal does not record the operating point
            # it was distilled at.
            tail_quantile=np.array([float(self.tail_q)], dtype=np.float64),
            phase_kappa=np.array([float(self.kappa)], dtype=np.float64),
            y_labelled=np.ones(len(self.crystal_Y), dtype=bool),
            # v3.14.2: label-free crystals store a synthetic novelty coordinate
            # (spectral leverage) in crystal_Y, not measured targets. Optional
            # key with a legacy fallback (the schema-1 pattern): older loaders
            # ignore it, older artifacts load as label_free=False.
            label_free=np.array(bool(self.unsupervised_mode)),
        )
        if include_projector:
            payload["Omega"] = self.Omega.astype(np.float32)
            payload["phase"] = self.phase.astype(np.float32)
        out_path = npz_output_path(filepath)
        np.savez_compressed(out_path, **payload)
        return os.path.getsize(out_path) / 1024.0


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

    Threading (v3.14.4): read-only paths are thread-safe via a per-loader
    re-entrant lock; mutating paths acquire the same lock and serialize.
    One loader per thread is no longer required for read-only radar calls.
    """

    #: Highest artifact schema this loader understands (v3.13.1). Artifacts
    #: written by older shd versions without the key load as schema 1.
    LOADER_SCHEMA = 1

    @staticmethod
    def _read_schema(data) -> int:
        """Read the artifact schema version from a loaded .npz (v3.13.1).

        Legacy artifacts (v3.7 .. v3.13.0) carry no ``schema_version`` key;
        they are read as schema 1 -- the pre-versioning layout IS schema 1.
        Newer schemas are accepted when this loader claims to understand at
        least their baseline, otherwise the mismatch is an explicit error at
        load time (the schema contract the crystals previously lacked). A
        schema number NEWER than this loader's that stores no
        ``schema_min_loader`` key is rejected as well: a real newer artifact
        always carries the key (its writer knows its own baseline), so its
        absence means the number is not a trusted minimum-loader contract and
        must not be silently accepted.
        """
        if "schema_version" not in data:
            return 1  # legacy pre-versioning artifact
        try:
            schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        except Exception as exc:  # corrupt / non-numeric payload
            raise ValueError(
                f"artifact stores an unreadable schema_version: {exc!r}"
            ) from exc
        if "schema_min_loader" in data:
            try:
                min_loader = int(np.asarray(data["schema_min_loader"]).reshape(-1)[0])
            except Exception as exc:
                raise ValueError(
                    f"artifact stores an unreadable schema_min_loader: {exc!r}"
                ) from exc
            if min_loader > HologramCrystalLoader.LOADER_SCHEMA:
                raise ValueError(
                    f"artifact schema {schema} requires a loader understanding "
                    f"schema {min_loader}; this loader understands up to schema "
                    f"{HologramCrystalLoader.LOADER_SCHEMA} -- upgrade the shd package"
                )
        elif schema > HologramCrystalLoader.LOADER_SCHEMA:
            raise ValueError(
                f"artifact schema {schema} is newer than this loader "
                f"(understands up to schema {HologramCrystalLoader.LOADER_SCHEMA}) "
                "and stores no schema_min_loader key -- upgrade the shd package"
            )
        return schema

    @staticmethod
    def _warn_schema_drift(schema: int, has_key: bool = True) -> None:
        """Warn when an artifact predates the loader's schema (v3.13.1).

        Two drift cases: (a) the artifact carries NO ``schema_version`` key --
        it was distilled by a pre-3.13.1 build and its provenance is
        unverifiable; (b) the artifact's schema number is lower than this
        loader's -- future loaders must not present old-crystal reports as
        current-schema ones. Loading proceeds either way (schema 1 IS the
        legacy layout), the crystal's fields stay trustworthy; the warning
        keeps provenance honest in reports.
        """
        if not has_key:
            warnings.warn(
                "artifact carries no schema_version key: distilled by a "
                "pre-3.13.1 build; treat its provenance as unverified "
                "(re-distill or re-save with shd >= 3.13.1 to version it)",
                UserWarning,
                stacklevel=3,
            )
        elif schema < HologramCrystalLoader.LOADER_SCHEMA:
            warnings.warn(
                f"artifact schema {schema} < loader schema "
                f"{HologramCrystalLoader.LOADER_SCHEMA}; legacy defaults apply",
                UserWarning,
                stacklevel=3,
            )

    def __init__(self, filepath: str = "hologram.npz", include_deltas: bool | None = None):
        self.filepath = filepath
        # v3.14.4: per-loader re-entrant lock. Read-only radar/summary/save
        # paths acquire it so concurrent check_domain_shift calls (which
        # read/write _prev_row in streaming mode and read shared fields)
        # serialize instead of racing; mutating heal/calibrate paths use
        # the same lock. RLock (not Lock) so a public method may call
        # another locked path on the same thread.
        self._lock = threading.RLock()
        # v3.13.2: materialise the whole payload up front and close the archive
        # before anything else runs. On Windows an open NpzFile keeps the .npz
        # locked, which broke TemporaryDirectory cleanup in tests and file
        # replacement in long-lived streaming sessions; materialising first also
        # guarantees the structural validation below never reads a half-open
        # archive. The payload dict is the untrusted-input boundary.
        npz = np.load(filepath, allow_pickle=False)
        try:
            data = {key: npz[key] for key in npz.files}
        finally:
            npz.close()
        self.schema_version = HologramCrystalLoader._read_schema(data)
        HologramCrystalLoader._warn_schema_drift(
            self.schema_version, has_key="schema_version" in data
        )
        if "shd_version" in data:
            self.shd_version = str(np.asarray(data["shd_version"]).reshape(-1)[0])
        else:
            self.shd_version = None  # pre-3.13.1 crystal: distilled before versioning
        self.X = data["crystal_X"].astype(np.float64)
        self.y = data["crystal_Y"].astype(np.float64)
        self.weights = data["gamma_weights"].astype(np.float64)
        self.eigvals = data["passport_eigvals"].astype(np.float64)
        self.eigvecs = data["passport_eigvecs"].astype(np.float64)
        self.X_mean = data["X_mean"].astype(np.float64)
        self.X_std = data["X_std"].astype(np.float64)
        self.n_seen = int(data["n_seen"][0])

        # v3.13.1 thread-model: an .hologram.npz crystal is UNTRUSTED input --
        # it travels through email, pull requests and model marketplaces.
        # ``allow_pickle=False`` above already blocks code execution; these
        # structural checks make a broken or poisoned file fail HERE with a
        # clear message instead of exploding later deep inside the radar or a
        # downstream fit with a confusing shape/NaN error.
        if self.X.ndim != 2 or self.X.shape[0] < 1:
            raise ValueError(
                f"artifact stores crystal_X with shape {self.X.shape}; "
                "a crystal must be a non-empty (rows, features) matrix -- "
                "the artifact is inconsistent"
            )
        rows, d_dim = self.X.shape
        for name, arr in (
            ("crystal_X", self.X), ("crystal_Y", self.y),
            ("gamma_weights", self.weights),
            ("passport_eigvals", self.eigvals),
            ("passport_eigvecs", self.eigvecs),
            ("X_mean", self.X_mean), ("X_std", self.X_std),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"artifact stores non-finite values in {name}; "
                    "the artifact is inconsistent"
                )
        if self.y.shape != (rows,):
            raise ValueError(
                f"artifact declares crystal_Y of shape {self.y.shape} but "
                f"crystal_X has {rows} rows; the artifact is inconsistent"
            )
        if self.weights.shape != (rows,):
            raise ValueError(
                f"artifact declares gamma_weights of shape {self.weights.shape} "
                f"but crystal_X has {rows} rows; the artifact is inconsistent"
            )
        if np.any(self.weights < 0):
            raise ValueError(
                "artifact stores negative gamma_weights (they feed a weighted "
                "ridge head); the artifact is inconsistent"
            )
        if (self.eigvals.ndim != 1 or self.eigvecs.ndim != 2
                or self.eigvecs.shape[0] < 1
                or self.eigvecs.shape[1] != self.eigvals.shape[0]):
            raise ValueError(
                f"artifact declares passport_eigvals of shape {self.eigvals.shape} "
                f"and passport_eigvecs of shape {self.eigvecs.shape}; the passport "
                "must be eigvals (rank,) + eigvecs (sketch_dim, rank) -- "
                "the artifact is inconsistent"
            )
        # X_mean/X_std carry the column statistics of the distilled stream: width
        # d for a plain crystal, 2d for a delta one (x||dx). A legacy delta
        # crystal may load with include_deltas=False (the flag postdates it), so
        # only other widths are inconsistent here; the odd-width delta case is
        # rejected further below.
        if (
            len(self.X_mean) not in (d_dim, 2 * d_dim)
            or self.X_std.shape != self.X_mean.shape
        ):
            raise ValueError(
                f"artifact declares X_mean/X_std of width "
                f"{len(self.X_mean)}/{len(self.X_std)} but crystal_X has "
                f"{d_dim} features (expected {d_dim} for a plain crystal or "
                f"{2 * d_dim} for a delta one); the artifact is inconsistent"
            )
        if np.any(self.X_std < 0):
            raise ValueError(
                "artifact stores negative X_std; the artifact is inconsistent"
            )
        if self.n_seen < 1:
            raise ValueError(
                f"artifact declares n_seen={self.n_seen}; the artifact is inconsistent"
            )

        # Which rows carry a real target. Legacy artifacts predate the field:
        # every row is treated as labelled.
        if "y_labelled" in data:
            self.y_labelled = np.asarray(data["y_labelled"], dtype=bool).reshape(-1)
            if len(self.y_labelled) != len(self.y):
                raise ValueError(
                    f"artifact declares y_labelled of length {len(self.y_labelled)} "
                    f"but crystal_Y has {len(self.y)} rows; the artifact is inconsistent"
                )
        else:
            self.y_labelled = np.ones(len(self.y), dtype=bool)
        # v3.14.2: label-free crystals mark their synthetic novelty coordinate
        # so heal's novelty branch is reachable; legacy artifacts have no key.
        if "label_free" in data:
            self.label_free = bool(np.asarray(data["label_free"]).reshape(-1)[0])
        else:
            self.label_free = False  # legacy artifact: predates the marker
        self._has_real_targets = bool(np.any(self.y_labelled)) and not self.label_free
        self._target_placeholder = float(np.median(self.y[self.y_labelled])) if self._has_real_targets else 0.0

        self.k_dim = self.eigvecs.shape[0]
        if "tail" in data:
            self.tail = str(np.asarray(data["tail"]).item())
        else:
            self.tail = "high"  # legacy artifacts predate the tail field
        if self.tail not in ("high", "low", "both"):
            warnings.warn(
                f"artifact stores an invalid tail {self.tail!r} (expected 'high', "
                "'low' or 'both'); falling back to 'high' -- re-save the crystal "
                "with a current shd build",
                UserWarning, stacklevel=2,
            )
            self.tail = "high"

        if include_deltas is not None:
            self.include_deltas = bool(include_deltas)
        elif "include_deltas" in data:
            self.include_deltas = bool(np.asarray(data["include_deltas"]).item())
        else:
            # Legacy artifacts predate the flag. Treat them as plain (non-delta)
            # crystals; pass include_deltas=True explicitly for a legacy delta
            # crystal, then re-save to persist the flag.
            self.include_deltas = False
        if self.include_deltas and len(self.X_mean) % 2:
            raise ValueError(
                "artifact declares include_deltas=True but X_mean has odd width "
                f"({len(self.X_mean)}); the artifact is inconsistent"
            )

        # Band split used at distillation time; heal_domain_shift protects this
        # fraction of the crystal's phase-peak band when pruning under a cap.
        # Legacy artifacts predate the field and fall back to the 0.26 default.
        if "high_band_ratio" in data:
            br = float(np.asarray(data["high_band_ratio"]).reshape(-1)[0])
            if 0.0 <= br <= 1.0:
                self.high_band_ratio = br
            else:
                warnings.warn(
                    f"artifact stores high_band_ratio={br} outside the distiller's "
                    "valid range [0, 1]; falling back to the 0.26 default -- the "
                    "peak band this crystal was built with is unknown, re-save it "
                    "with a current shd build",
                    UserWarning, stacklevel=2,
                )
                self.high_band_ratio = 0.26
        else:
            self.high_band_ratio = 0.26

        # v3.15.2: the operating point travels with the artifact when the
        # file carries it. Legacy artifacts predate the field -> None
        # (unknown), which is reported honestly in summary().
        self.tail_quantile = None
        if "tail_quantile" in data:
            tq = float(np.asarray(data["tail_quantile"]).reshape(-1)[0])
            if np.isfinite(tq):
                self.tail_quantile = tq
        self.phase_kappa = None
        if "phase_kappa" in data:
            pk = float(np.asarray(data["phase_kappa"]).reshape(-1)[0])
            if np.isfinite(pk):
                self.phase_kappa = pk

        # Which population the passport describes. Legacy artifacts predate
        # the field and were built over the whole stream.
        if "passport_reference" in data:
            ref = str(np.asarray(data["passport_reference"]).item())
            if ref not in ("background", "stream"):
                warnings.warn(
                    f"artifact stores an invalid passport_reference {ref!r} "
                    "(expected 'stream' or 'background'); falling back to "
                    "'stream' -- re-save the crystal with a current shd build",
                    UserWarning, stacklevel=2,
                )
                ref = "stream"
            self.passport_reference = ref
        else:
            self.passport_reference = "stream"

        # The critical band's fixed target cut (y >= critical_threshold), if
        # the crystal was distilled with one. The peak slice, the budget slice
        # and heal's protected band all read the SAME definition the lattice
        # was built with (v3.12.0); legacy artifacts predate the field and
        # have no threshold-driven band.
        if "critical_threshold" in data:
            ct = float(np.asarray(data["critical_threshold"]).reshape(-1)[0])
            self.critical_threshold = None if np.isnan(ct) else ct
        else:
            self.critical_threshold = None

        # Per-deployment radar calibration (v3.12.0). Legacy artifacts and
        # uncalibrated loaders fall back to the rank-scaled default constant.
        if "tau_calibrated" in data:
            missing = [k for k in ("tau_window", "tau_n_windows", "tau_fa_rate")
                       if k not in data]
            if missing:
                raise ValueError(
                    "artifact stores tau_calibrated without " + ", ".join(missing)
                    + "; the radar calibration must travel as one block -- "
                    "the artifact is inconsistent"
                )
            self.tau_calibrated = float(np.asarray(data["tau_calibrated"]).reshape(-1)[0])
            self.tau_window = int(np.asarray(data["tau_window"]).reshape(-1)[0])
            self.tau_n_windows = int(np.asarray(data["tau_n_windows"]).reshape(-1)[0])
            self.tau_fa_rate = float(np.asarray(data["tau_fa_rate"]).reshape(-1)[0])
        else:
            self.tau_calibrated = None
            self.tau_window = None
            self.tau_n_windows = None
            self.tau_fa_rate = None

        # Mean-shift radar calibration (v3.13.0): the combined radar adds a
        # per-feature standardized mean-deviation term to the BW term and
        # scores one fused threshold. Both thresholds are persisted so a
        # deployment reloads bit-identically (like tau_calibrated). Legacy
        # artifacts and uncalibrated loaders leave both None, in which case
        # check_domain_shift(mean_shift=True) falls back to scoring the
        # mean-shift term against the rank-scaled default threshold.
        if "ms_tau_calibrated" in data:
            missing = [k for k in ("ms_tau_window", "ms_tau_n_windows", "ms_tau_fa_rate")
                       if k not in data]
            if missing:
                raise ValueError(
                    "artifact stores ms_tau_calibrated without " + ", ".join(missing)
                    + "; the combined-radar calibration must travel as one block -- "
                    "the artifact is inconsistent"
                )
            self.ms_tau_calibrated = float(np.asarray(data["ms_tau_calibrated"]).reshape(-1)[0])
            self.ms_tau_window = int(np.asarray(data["ms_tau_window"]).reshape(-1)[0])
            self.ms_tau_n_windows = int(np.asarray(data["ms_tau_n_windows"]).reshape(-1)[0])
            self.ms_tau_fa_rate = float(np.asarray(data["ms_tau_fa_rate"]).reshape(-1)[0])
        else:
            self.ms_tau_calibrated = None
            self.ms_tau_window = None
            self.ms_tau_n_windows = None
            self.ms_tau_fa_rate = None

        # Panel radar calibration (v3.14.3): the panel adds MMD (RFF-space) and
        # energy (raw-space) terms to the BW + mean-shift score and scores one
        # fused threshold. Persisted like the other calibrations so a
        # deployment reloads bit-identically. Legacy artifacts and
        # uncalibrated loaders leave it None, in which case
        # check_domain_shift(panel=True) falls back to scoring the panel
        # against the rank-scaled default threshold.
        if "panel_tau_calibrated" in data:
            missing = [k for k in ("panel_tau_window", "panel_tau_n_windows", "panel_tau_fa_rate")
                       if k not in data]
            if missing:
                raise ValueError(
                    "artifact stores panel_tau_calibrated without " + ", ".join(missing)
                    + "; the panel-radar calibration must travel as one block -- "
                    "the artifact is inconsistent"
                )
            self.panel_tau_calibrated = float(np.asarray(data["panel_tau_calibrated"]).reshape(-1)[0])
            self.panel_tau_window = int(np.asarray(data["panel_tau_window"]).reshape(-1)[0])
            self.panel_tau_n_windows = int(np.asarray(data["panel_tau_n_windows"]).reshape(-1)[0])
            self.panel_tau_fa_rate = float(np.asarray(data["panel_tau_fa_rate"]).reshape(-1)[0])
        else:
            self.panel_tau_calibrated = None
            self.panel_tau_window = None
            self.panel_tau_n_windows = None
            self.panel_tau_fa_rate = None

        self._prev_row = None  # last raw row of the previous streaming radar call
        self._e_value = None  # e-process evidence (v3.14.4, reset on heal/load)
        self._e_t = 0

        if "Omega" in data:
            self.Omega = data["Omega"].astype(np.float64)
            self.phase = data["phase"].astype(np.float64)
        else:
            self.Omega = None
            self.phase = None

        # Reconstruct the passport spectral operator from the top-32 eigenpairs.
        self.Psi_passport = (self.eigvecs * self.eigvals) @ self.eigvecs.T

    def _select_weights(self, weighting: str) -> np.ndarray:
        """Return the crystal's gamma weights or uniform weights (mean one)."""
        if weighting == "gamma":
            return self.weights
        if weighting == "uniform":
            return np.ones_like(self.weights)
        raise ValueError(f"weighting must be 'gamma' or 'uniform', got {weighting!r}")

    def get_training_data(
        self,
        budget_points: int | None = None,
        weighting: str = "gamma",
        drop_unlabelled: bool = True,
    ):
        """Return (X, y, sample_weight) for the chosen medoid budget (anytime mode).

        ``weighting="gamma"`` (default) returns the FISTA/Voronoi weights;
        ``weighting="uniform"`` returns ones. The 20-seed ablation in
        ``bench_weights.py`` shows this is a global/critical trade-off: gamma
        improves global KRR, while uniform can be better on the critical regime
        for some datasets (notably the 4D reactor).

        ``drop_unlabelled=True`` (the default since v3.11.1) removes the rows
        that were injected by a label-free ``heal_domain_shift``: they carry a
        neutral placeholder in ``y`` (the crystal's median target) rather than
        a measured value, and training on the placeholder would bias the
        model. Crystals without placeholder rows are unaffected, so this
        changes no published benchmark number.

        Since v3.12.1 the drop happens FIRST and the ``budget_points`` slice
        is spent on the remaining LABELLED rows, so a requested budget is
        always honoured (reproduced in ``tools/hunt/round2_probe_b``: 30 of 60
        requested rows were healed placeholders and got dropped, returning
        fewer rows than asked). Only if fewer labelled rows exist than the
        budget can the slice come back short.
        """
        w_all = self._select_weights(weighting)
        X, y, w = self.X, self.y, w_all
        if drop_unlabelled and not np.all(self.y_labelled):
            keep = self.y_labelled
            X, y, w = X[keep], y[keep], w[keep]
        if budget_points is not None:
            if int(budget_points) < 1:
                raise ValueError(f"budget_points must be >= 1, got {budget_points}")
            if budget_points < len(X):
                # v3.11.1: the phase-peak band is admitted FIRST, then the
                # largest background cells. Crystal rows are stored sorted by
                # cell count, so a plain head slice spent the whole budget on
                # background and starved the rare regime the crystal exists to
                # preserve (measured on a 200-medoid crystal with a 60-row
                # peak band: a budget of 100 kept only 4 peak rows).
                # v3.12.1: run this on the LABELLED row space so the budget
                # is spent where it helps (probe B).
                peak = self._peak_rank_idx(len(X), float(self.high_band_ratio), y=y)
                chosen = [int(i) for i in peak[:budget_points]]
                if len(chosen) < budget_points:
                    taken = set(chosen)
                    for i in range(len(X)):
                        if i not in taken:
                            chosen.append(i)
                            if len(chosen) == budget_points:
                                break
                idx = np.asarray(chosen, dtype=np.int64)
                X, y, w = X[idx], y[idx], w[idx]
        if len(X) != len(self.X):
            w = w.copy()
            w /= np.mean(w)
        return X, y, w

    def get_phase_peak_slice(
        self,
        high_band_ratio: float | None = None,
        y_threshold: float | None = None,
        weighting: str = "gamma",
    ):
        """Return ``(X_peak, y_peak, w_peak)`` for rare / discontinuity regimes.

        Intended for warm-starting Bayesian Optimization (e.g. NEON Lab GP+EI)
        around phase transitions without fitting a cubic O(N^3) GP on the full
        telemetry stream. If ``y_threshold`` is given, selects medoids with
        ``y >= y_threshold``; a crystal distilled with ``critical_threshold``
        instead serves the medoids above that threshold -- the same band the
        lattice was built from (v3.12.0); otherwise selects the
        ``high_band_ratio`` fraction of medoids that are most extreme in the
        crystal's ``tail`` direction:
        the top by ``y`` for ``tail="high"``, the bottom for ``tail="low"``, and
        the most extreme relative to the median for ``tail="both"``. Rows are
        returned in the band's rank order (ascending ``y`` for ``"high"`` /
        ``"low"``, ascending extremeness for ``"both"``).

        ``weighting="gamma"`` (default) returns the crystal's FISTA/Voronoi
        weights; ``weighting="uniform"`` returns ones for deployments where the
        critical regime is the only objective (see ``bench_weights.py``).

        ``high_band_ratio`` defaults to the crystal's persisted band ratio
        (v3.11.1), so the slice and the heal's protected band always agree.
        """
        # v3.11.1: the default follows the crystal's PERSISTED band ratio.
        # The old hard-coded 0.26 silently disagreed with the ratio the
        # crystal was built with (and with the ratio heal protects), so a
        # critical-first crystal (br=0.65) returned a band 2.5x too narrow.
        if high_band_ratio is None:
            high_band_ratio = float(self.high_band_ratio)
        if y_threshold is not None:
            mask = self.y >= float(y_threshold)
            if not np.any(mask):
                idx = self._peak_rank_idx(len(self.y), high_band_ratio)
            else:
                idx = np.where(mask)[0]
        else:
            idx = self._peak_rank_idx(len(self.y), high_band_ratio)
        X_p = self.X[idx].copy()
        y_p = self.y[idx].copy()
        w_p = self._select_weights(weighting)[idx].copy()
        w_p /= np.mean(w_p)
        return X_p, y_p, w_p

    def _peak_rank_idx(self, n_medoids: int, high_band_ratio: float, y=None) -> np.ndarray:
        """Indices of the phase-peak band, in the band's rank order.

        A crystal distilled with ``critical_threshold`` defines its critical
        band on the UPPER target side (``y >= critical_threshold``), and the
        slice, the budget slice and heal's protected band must read back the
        SAME band the lattice was built from: exactly the medoids above the
        threshold, at most the band quota ``k_top`` (keeping the largest
        ``y``). v3.11.1 let ``tail="both"`` re-rank that band by
        extremeness-from-median and served LOW-side rows the band was never
        built from (measured: 7 of 18 slice rows below the threshold).

        Without a threshold, ``tail="high"`` returns the ``k_top`` largest-y
        medoids in ascending-y order (bitwise-compatible with the v3.7 slice
        ordering); ``"low"`` returns the smallest-y medoids, also ascending
        in y; ``"both"`` returns the most extreme medoids relative to the
        median, in ascending extremeness order. ``y`` overrides the ranked
        target column (heal protects the OLD medoids' band while new rows
        are already appended).
        """
        y_ref = self.y if y is None else np.asarray(y, dtype=np.float64)
        k_top = max(1, int(round(n_medoids * high_band_ratio)))
        if self.critical_threshold is not None:
            above = np.where(y_ref >= float(self.critical_threshold))[0]
            if len(above) > 0:
                above = above[np.argsort(y_ref[above])]
                if len(above) > k_top:
                    above = above[-k_top:]
                return above
            # Degenerate crystal: no medoid sits above the threshold. Fall
            # through to the tail-based ranking rather than return nothing.
        if self.tail == "high":
            return np.argsort(y_ref)[-k_top:]
        if self.tail == "low":
            return np.argsort(y_ref)[:k_top]
        rank = np.abs(y_ref - np.median(y_ref))  # "both"
        return np.argsort(rank)[-k_top:]

    def _normalize_stream(
        self,
        X_new_stream: np.ndarray,
        remember_prev_row: bool = False,
    ) -> np.ndarray:
        X_new_stream = np.asarray(X_new_stream, dtype=np.float64)
        if X_new_stream.ndim == 1:
            X_new_stream = X_new_stream.reshape(1, -1)
        if X_new_stream.ndim != 2:
            raise ValueError(f"stream window must be 2-D, got shape {X_new_stream.shape}")
        if not np.isfinite(X_new_stream).all():
            raise ValueError(
                "stream window contains NaN/Inf; the OOD radar projects raw rows, "
                "so non-finite values cannot be scored"
            )
        if len(X_new_stream) == 0:
            raise ValueError("stream window is empty")
        expected = len(self.X_mean)
        if self.include_deltas:
            if expected % 2:
                raise ValueError(
                    "artifact is marked include_deltas=True but X_mean has odd width "
                    f"({expected}); the artifact is inconsistent"
                )
            d_raw = expected // 2
            if X_new_stream.shape[1] == d_raw:
                dX = np.zeros_like(X_new_stream, dtype=np.float64)
                if len(X_new_stream) > 1:
                    dX[1:] = X_new_stream[1:] - X_new_stream[:-1]
                if remember_prev_row and self._prev_row is not None:
                    dX[0] = X_new_stream[0] - self._prev_row
                if remember_prev_row and len(X_new_stream) > 0:
                    self._prev_row = np.asarray(X_new_stream[-1], dtype=np.float64).copy()
                X_eff = np.hstack([X_new_stream, dX])
            elif X_new_stream.shape[1] == expected:
                # Already-augmented [x || dx] window: trust the caller and do not
                # update the cross-call continuity state.
                X_eff = X_new_stream
            else:
                raise ValueError(
                    f"delta crystal expects {d_raw} raw features (or {expected} "
                    f"pre-augmented [x || dx] features), got {X_new_stream.shape[1]}"
                )
        else:
            if X_new_stream.shape[1] != expected:
                raise ValueError(
                    f"stream window has {X_new_stream.shape[1]} features but this crystal "
                    f"expects {expected} (non-delta crystal; pass include_deltas=True at "
                    "distillation time for delta streams)"
                )
            X_eff = X_new_stream
        return np.clip((X_eff - self.X_mean) / self.X_std, -8.0, 8.0)

    def _prepare_stream(
        self,
        X_new_stream: np.ndarray,
        distiller_ref=None,
        remember_prev_row: bool = False,
    ):
        """Normalize + embed a new stream in a single pass; returns ``(Z, U)``.

        When ``remember_prev_row`` is True and the crystal was built with
        ``include_deltas=True``, the last raw row is stored so the next call's
        ``dX[0]`` continues the stream across calls (matching the streaming
        semantics of the distiller) instead of defaulting to zero. Non-delta
        crystals ignore this flag entirely.
        """
        Z = self._normalize_stream(X_new_stream, remember_prev_row=remember_prev_row)
        if self.Omega is not None:
            U = np.sqrt(2.0 / self.k_dim) * np.cos(Z @ self.Omega + self.phase)
            norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            return Z, U / norms
        if distiller_ref is None:
            raise ValueError(
                "This artifact has no stored projector (Omega/phase). Pass a "
                "distiller_ref with _spectral_transform, or re-save the crystal "
                "with save_hologram(include_projector=True)."
            )
        t = distiller_ref._spectral_transform(X_new_stream)
        return Z, (t[1] if isinstance(t, tuple) else t)

    def _calibrated_threshold(self, n_batch: int, adaptive_threshold: bool = True) -> float:
        if not adaptive_threshold:
            return float(BW_ALERT_THRESHOLD)
        if self.tau_calibrated is not None:
            # v3.12.0: per-deployment quantile calibration (calibrate_threshold).
            # tau was measured at tau_window rows; smaller windows widen by the
            # same finite-sample 1/sqrt(B) factor as the default path. No extra
            # rank factor: the measured distances already live on this
            # passport's rank.
            b_eff = max(int(n_batch), 16)
            w_ref = max(int(self.tau_window), 16)
            return float(self.tau_calibrated) * max(1.0, float(np.sqrt(float(w_ref) / b_eff)))
        # Finite-sample covariance fluctuation scales as 1 / sqrt(B). For batches
        # smaller than 1024 rows, widen the threshold smoothly so small sensor
        # windows do not trigger false alarms from sampling noise alone. The
        # constant 1024 was calibrated on the UCI Superconductivity heal
        # benchmark: in-distribution windows of B=64 stay below the widened
        # threshold while true OOD windows (BW ~0.44+) are still detected.
        b_eff = max(int(n_batch), 16)
        sample_scale = max(1.0, float(np.sqrt(1024.0 / b_eff)))
        # v3.12.0: the BW distance between two independent estimates of the
        # SAME distribution grows ~ sqrt(rank) with the passport's subspace
        # rank, while BW_ALERT_THRESHOLD was calibrated at the shipped rank
        # of 32. rank_passport=128 therefore tripped the radar on perfectly
        # in-distribution windows (measured: in-dist BW 0.161 vs the unscaled
        # threshold); scale the base by sqrt(rank / 32) so every rank keeps
        # its in-distribution headroom. A rank-32 crystal is bit-identical.
        rank_scale = float(np.sqrt(len(self.eigvals) / 32.0))
        return float(BW_ALERT_THRESHOLD * sample_scale * rank_scale)

    def _calibrated_threshold_ms(self, n_batch: int, adaptive_threshold: bool = True) -> float:
        """Threshold for the COMBINED (BW + mean-shift) score (v3.13.0).

        Mirrors :meth:`_calibrated_threshold`'s finite-sample 1/sqrt(B)
        scaling on the combined score, using the ``ms_tau_*`` calibration
        fields when present, else the same default logic.
        """
        if not adaptive_threshold:
            return float(BW_ALERT_THRESHOLD)
        if self.ms_tau_calibrated is not None:
            b_eff = max(int(n_batch), 16)
            w_ref = max(int(self.ms_tau_window), 16)
            return float(self.ms_tau_calibrated) * max(1.0, float(np.sqrt(float(w_ref) / b_eff)))
        b_eff = max(int(n_batch), 16)
        sample_scale = max(1.0, float(np.sqrt(1024.0 / b_eff)))
        rank_scale = float(np.sqrt(len(self.eigvals) / 32.0))
        # The combined score sums the BW distance (rank-scaled) and the
        # mean-shift z-score (no rank dependence), so its default threshold
        # is the rank-scaled BW constant -- the mean-shift term rides along.
        return float(BW_ALERT_THRESHOLD * sample_scale * rank_scale)

    def _calibrated_threshold_panel(self, n_batch: int, adaptive_threshold: bool = True) -> float:
        """Threshold for the PANEL (BW + mean-shift + MMD + energy) score (v3.14.3).

        Mirrors :meth:`_calibrated_threshold`'s finite-sample 1/sqrt(B)
        scaling on the panel score, using the ``panel_tau_*`` calibration
        fields when present, else the same default logic as the combined
        score. The panel terms are not rank-scaled beyond the BW component,
        so the default follows the rank-scaled BW constant exactly like the
        combined path.
        """
        if not adaptive_threshold:
            return float(BW_ALERT_THRESHOLD)
        if self.panel_tau_calibrated is not None:
            b_eff = max(int(n_batch), 16)
            w_ref = max(int(self.panel_tau_window), 16)
            return float(self.panel_tau_calibrated) * max(1.0, float(np.sqrt(float(w_ref) / b_eff)))
        b_eff = max(int(n_batch), 16)
        sample_scale = max(1.0, float(np.sqrt(1024.0 / b_eff)))
        rank_scale = float(np.sqrt(len(self.eigvals) / 32.0))
        return float(BW_ALERT_THRESHOLD * sample_scale * rank_scale)

    def _effective_stream(self, X_stream, remember_prev_row=False, prev_row=None):
        """Raw effective feature matrix for the mean-shift term (v3.13.0).

        Mirrors :meth:`_normalize_stream`'s delta augmentation WITHOUT the
        per-row normalization/clip, so the window mean is computed on the raw
        (x, or x||dx for delta crystals) features. For non-delta crystals this
        is the stream as-is; the width must equal ``len(self.X_mean)``.

        ``prev_row`` lets the caller pin the continuity row explicitly: the
        combined-radar path passes the snapshot captured before the BW pass
        advanced the loader state, so the mean-shift term uses the previous
        call's last row (continuation) even though it runs after the BW pass.
        """
        X = np.asarray(X_stream, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"stream window must be 2-D, got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("stream window contains NaN/Inf; the mean-shift term "
                             "projects raw rows, so non-finite values cannot be scored")
        expected = len(self.X_mean)
        if self.include_deltas:
            d_raw = expected // 2
            if X.shape[1] == d_raw:
                dX = np.zeros_like(X, dtype=np.float64)
                if len(X) > 1:
                    dX[1:] = X[1:] - X[:-1]
                if remember_prev_row:
                    # Prefer the explicit snapshot (captured BEFORE the BW pass
                    # advanced the loader state) so a mean-shift term evaluated in
                    # the same check_domain_shift call still sees the PREVIOUS
                    # call's last row, not this window's own last row (the v3.13.0
                    # wrap-around ordering defect: _bw_distance runs first and
                    # reassigns self._prev_row before the mean-shift term ran).
                    prev = prev_row if prev_row is not None else self._prev_row
                    if prev is not None:
                        dX[0] = X[0] - prev
                    if remember_prev_row and len(X) > 0:
                        self._prev_row = np.asarray(X[-1], dtype=np.float64).copy()
                return np.hstack([X, dX])
            elif X.shape[1] == expected:
                return X
            else:
                raise ValueError(
                    f"delta crystal expects {d_raw} raw features (or {expected} "
                    f"pre-augmented [x || dx] features), got {X.shape[1]}"
                )
        else:
            if X.shape[1] != expected:
                raise ValueError(
                    f"stream window has {X.shape[1]} features but this crystal "
                    f"expects {expected} (non-delta crystal; pass include_deltas=True "
                    "at distillation time for delta streams)"
                )
            return X

    def _mean_shift_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ):
        """Max per-feature standardized mean deviation of one window (v3.13.0).

        Leak signatures on MetroPT-3 appear as mean shifts, which the
        Bures-Wasserstein radar (second-moment only) is blind to. This term
        z-scores the window mean against the crystal's OWN medoid cloud
        (``self.X``: the real stored medoids spanning the whole learn window)
        and takes the max over features, so the most-deviated sensor drives
        the alert. Self-contained: uses the artifact's own rows, no original
        dataset needed.

        The persisted ``X_mean`` / ``X_std`` are deliberately NOT used here:
        they are first-batch running statistics that exist to normalize the
        RFF input. A feature that happens to be constant in the first batch
        (MetroPT-3 ``LPS``: std ~1e-8 over the first 4000 minutes) makes the
        z-score explode to ~3.9e7 on quantization noise, which drowns the
        leak signal (measured: the calibrated combined tau sat at 3.86e7 and
        the arm fired on noise). The medoid cloud spans the entire learn
        window, so its per-feature std is the deployment-scale denominator,
        and it travels with ``heal_domain_shift`` (injected medoids shift the
        reference population with it).
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        if self.include_deltas:
            # The effective window lives in the 2*d space [x || dx]; the reference
            # population must live in the SAME space. The crystal's medoids are
            # stored in raw d-feature space (heal_domain_shift refuses
            # pre-augmented windows for the same width-mismatch reason), so
            # augment the medoid cloud with its own first differences:
            # dx[0] = 0 (no earlier medoid), dx[i] = X[i] - X[i-1] for i > 0.
            # The legacy persisted X_mean (first-batch raw-space statistics) is
            # deliberately NOT the reference here: it is the raw-space
            # first-batch statistic, in the WRONG space, and it would re-introduce
            # the scale defect v3.13.0 fixed (degenerate first-batch stds on
            # near-constant features).
            ref = self._mean_shift_reference()
        else:
            ref = self.X
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        ok = sd > 1e-9
        z = np.abs(X_eff.mean(axis=0) - mu) / np.where(ok, sd, 1.0)
        val = float(z[ok].max()) if ok.any() else 0.0
        return val, len(X_eff)

    def check_domain_shift(
        self,
        X_new_stream: np.ndarray,
        distiller_ref=None,
        adaptive_threshold: bool = True,
        streaming: bool = False,
        mean_shift: bool = False,
        panel: bool = False,
        e_process: bool = False,
    ) -> dict:
        """Compare a new data stream against the spectral passport.

        Projects the new stream's RFF spectrum onto the passport's
        eigen-subspace and computes a Bures-Wasserstein distance (32x32
        matrices instead of 256x256). Returns ``bw_distance``, ``threshold``,
        ``threshold_source``, ``latency_ms`` and ``is_shifted``.

        With ``streaming=True`` and a delta crystal
        (``include_deltas=True`` at distillation time), the first-difference of
        the window's opening row is computed against the previous call's last
        raw row instead of defaulting to zero, matching the distiller's
        cross-batch continuity. Non-delta crystals ignore the flag; the default
        ``False`` keeps every call an independent, stateless window.

        ``threshold_source`` describes the threshold that was ACTUALLY
        applied: with ``adaptive_threshold=False`` the applied threshold is
        always the rank-scaled default constant (``threshold_source:
        "default"``), even when a calibration exists; only
        ``adaptive_threshold=True`` scores against the per-deployment
        quantile calibration (``threshold_source: "calibrated"``) when
        ``calibrate_threshold()`` has been run.

        With ``mean_shift=True`` (v3.13.0) the dict also returns
        ``mean_shift_distance`` and ``combined_distance`` (= ``bw_distance`` +
        ``mean_shift_distance``): the mean-shift term catches the mean-shift
        signatures (leak/phase-change) the BW term misses, so the combined
        score is what a deployment device should alarm on. Default ``False``
        keeps the existing behaviour bit-identical.

        With ``panel=True`` (v3.14.3) the dict also returns a panel of four
        scores — ``panel_distance`` (= BW + mean-shift + MMD + energy),
        ``panel_threshold`` / ``panel_threshold_source`` and ``is_shifted`` /
        ``panel_is_shifted`` / ``panel_shift_evidence``. The panel terms
        measure distribution discrepancy in four different geometries so a
        shift visible in none of them is less likely. Default ``False``
        keeps the existing behaviour bit-identical. ``is_shifted`` follows
        ``panel_distance > panel_threshold`` when ``panel=True`` (and
        ``mean_shift=False``); with ``mean_shift=True`` the mean-shift path
        keeps precedence (``is_shifted`` = combined > combined_threshold and
        ``panel_is_shifted`` = panel > panel_threshold). Both fields are
        present so callers can read the verdict of either path.
        """
        # Return without lock: existing tests call internals directly
        # (get_training_data etc. entry points are locked below). The
        # snapshot+RLock contract lives on check_domain_shift/heal/save;
        # private helpers stay unlocked so locked callers do not deadlock.
        # v3.14.4: e-process. When ``e_process=True`` the panel verdict is
        # the anytime-valid e-value average (E_t / t > 1/alpha) rather than
        # the fixed quantile; the state lives on the loader so a deployment
        # can alarm on the next window after several healthy ones without
        # re-calibration.
        # Capture the continuity state BEFORE _bw_distance runs: the BW pass
        # advances self._prev_row to this window's last row, so the mean-shift
        # term (evaluated afterwards in the same call) must use this snapshot
        # to keep true cross-call continuity (dX[0] = X[0] - previous call's
        # last row) instead of the wrap-around value (X[0] - this window's last
        # row) the v3.13.0 ordering produced.
        with self._lock:
            if streaming:
                prev_row_snapshot = (None if self._prev_row is None
                                     else np.asarray(self._prev_row, dtype=np.float64).copy())
            else:
                prev_row_snapshot = None
            t0 = time.perf_counter()
            bw_dist, n_rows = self._bw_distance(
                X_new_stream, distiller_ref=distiller_ref, remember_prev_row=streaming
            )
            thr = self._calibrated_threshold(n_rows, adaptive_threshold=adaptive_threshold)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            res = {
                "bw_distance": bw_dist,
                "threshold": thr,
                "threshold_source": (
                    "calibrated"
                    if adaptive_threshold and self.tau_calibrated is not None
                    else "default"
                ),
                "latency_ms": latency_ms,
            }
            ms = None
            n_rows_ms = n_rows
            if panel or mean_shift or e_process:
                ms, n_rows_ms = self._mean_shift_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
            if mean_shift:
                combined = float(bw_dist + ms)
                ms_thr = self._calibrated_threshold_ms(n_rows, adaptive_threshold=adaptive_threshold)
                res["mean_shift_distance"] = ms
                res["combined_distance"] = combined
                res["combined_threshold"] = float(ms_thr)
                res["combined_threshold_source"] = (
                    "calibrated"
                    if adaptive_threshold and self.ms_tau_calibrated is not None
                    else "default"
                )
                res["is_shifted"] = combined > ms_thr
            else:
                res["is_shifted"] = bw_dist > thr
            if panel or e_process:
                mmd = self._mmd_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                energy = self._energy_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                sliced = self._sliced_w1_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                autocorr = self._autocorr_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                autocorr2 = self._autocorr_lag2_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                rank_dfs = self._rank_dfs_distance(
                    X_new_stream, distiller_ref=distiller_ref,
                    remember_prev_row=streaming, prev_row=prev_row_snapshot
                )
                panel_distance = float(bw_dist + ms + mmd + energy + sliced + autocorr + autocorr2 + rank_dfs)
                panel_thr = self._calibrated_threshold_panel(
                    n_rows_ms, adaptive_threshold=adaptive_threshold
                )
                res["panel_distance"] = panel_distance
                res["panel_threshold"] = float(panel_thr)
                panel_verdict_valid = bool(
                    adaptive_threshold and self.panel_tau_calibrated is not None
                )
                res["panel_threshold_source"] = (
                    "calibrated"
                    if panel_verdict_valid
                    else ("uncalibrated" if adaptive_threshold else "default")
                )
                res["panel_mmd_distance"] = float(mmd)
                res["panel_energy_distance"] = float(energy)
                res["panel_sliced_w1"] = float(sliced)
                res["panel_autocorr_distance"] = float(autocorr)
                res["panel_autocorr_lag2_distance"] = float(autocorr2)
                res["panel_rank_dfs_distance"] = float(rank_dfs)
                res["panel_latency_ms"] = (time.perf_counter() - t0) * 1000.0
                # v3.15.1 (bughunt): the panel score sums EIGHT non-negative
                # terms (bw + mean-shift + mmd + energy + sliced +
                # autocorr + autocorr2 + rank_dfs) while the default panel
                # threshold is the BW-calibrated constant, which was fitted
                # for the 4-term panel. Measured on in-distribution windows
                # the 8-term score sits ~3.5-5x ABOVE the threshold, so the
                # uncalibrated panel verdict was always "shifted" -- and it
                # also overwrote is_shifted. Without a panel-specific
                # calibration the panel verdict is not a verdict: report it
                # as unknown and keep the BW verdict.
                if not panel_verdict_valid:
                    import warnings
                    warnings.warn(
                        "panel=True without a panel calibration: the panel "
                        "score sums 8 terms while the default threshold is "
                        "the BW-calibrated constant (fitted for 4 terms), so "
                        "the panel verdict is unusable -- panel_is_shifted is "
                        "reported as None and is_shifted keeps the BW verdict. "
                        "Calibrate the panel on healthy windows "
                        "(calibrate_threshold_panel) before using it.",
                        UserWarning, stacklevel=2,
                    )
                    res["panel_shift_evidence"] = None
                    res["panel_is_shifted"] = None
                else:
                    res["panel_shift_evidence"] = bool(panel_distance > panel_thr)
                    res["panel_is_shifted"] = bool(panel_distance > panel_thr)
                if e_process:
                    e = self._e_process_alarm(panel_distance)
                    res["e_value"] = e["e_value"]
                    res["e_t"] = e["e_t"]
                    res["e_average"] = e["e_average"]
                    res["e_threshold"] = e["e_threshold"]
                    res["e_alarm"] = e["e_alarm"] if panel_verdict_valid else None
                    if panel_verdict_valid:
                        # e-process verdict overrides panel_is_shifted
                        res["panel_is_shifted"] = bool(res["e_alarm"])
                        res["panel_shift_evidence"] = bool(res["e_alarm"])
                # `is_shifted` follows the panel when the caller opted into it
                # without the mean_shift path; the mean_shift path keeps precedence
                # so an explicit mean_shift=True caller keeps its existing contract.
                if not mean_shift and res["panel_is_shifted"] is not None:
                    res["is_shifted"] = bool(res["panel_is_shifted"])
            return res

    def _bw_distance(self, X_stream, distiller_ref=None, remember_prev_row=False):
        """BW distance between the passport and one window: ``(distance, rows)``.

        Shared by the radar (check_domain_shift) and threshold calibration
        (calibrate_threshold) so both score windows with bit-identical
        arithmetic.
        """
        _Z, U_new = self._prepare_stream(
            X_stream, distiller_ref=distiller_ref, remember_prev_row=remember_prev_row
        )
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
        # Row count of the *normalised* window: len(X_new_stream) would count
        # features of a 1-D single-row window (81 features -> batch of 81),
        # which inflated the threshold and made the same point decide
        # differently as (d,) vs (1, d).
        return bw_dist, len(U_new)

    def _mmd_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ) -> float:
        """RFF-MMD bias statistic of one window against the medoid cloud (v3.14.3).

        With ``U`` the unit-row-normalized Random Fourier Features of the
        window and ``U_ref`` the same features of the crystal's stored medoids
        (``self.X``), returns the squared unbiased MMD estimator

            biassed_MMD^2 = || mean(U) - mean(U_ref) ||^2

        in RFF space. It measures first-moment shift in the Fourier feature
        space and complements the BW second-moment term: a shift that moves
        the stream in the spectral space without rotating its covariance is
        visible here. Shared by ``check_domain_shift(panel=True)`` and the
        panel calibration so both score windows with bit-identical
        arithmetic.
        """
        _Z_new, U_new = self._prepare_stream(
            X_stream, distiller_ref=distiller_ref,
            remember_prev_row=remember_prev_row  # mirror _bw_distance state
        )
        # The RFF reference does not follow the mean-shift term's delta-space
        # augmentation: the medoid cloud is always embedded through _prepare_
        # stream (raw rows), the same as the BW pass.
        _Z_ref, U_ref = self._prepare_stream(
            self.X, distiller_ref=distiller_ref, remember_prev_row=False
        )
        mu_new = U_new.mean(axis=0)
        mu_ref = U_ref.mean(axis=0)
        return float(np.sum((mu_new - mu_ref) ** 2))

    def _energy_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ) -> float:
        """Euclidean energy distance of one window against the medoid cloud (v3.14.3).

        Returns the (biased) energy distance

            2 * mean_{x in W, y in R} ||x - y||_1
              - mean_{x,x' in W} ||x - x'||_1
              - mean_{y,y' in R} ||y - y'||_1

        on the RAW effective feature matrix (same space as the mean-shift
        term, including the delta-space augmentation for delta crystals).
        Energy distance needs no kernel choice and sees both location and
        scale shifts in raw space; it complements the spectral terms.
        Shared by ``check_domain_shift(panel=True)`` and the panel
        calibration so both score windows with bit-identical arithmetic.
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        ref = self._mean_shift_reference()
        # Pairwise L1 means; sizes are medoid-count (<= 300) x window rows and
        # medoid-count x medoid-count, both small by construction.
        cross = np.mean(np.abs(X_eff[:, None, :] - ref[None, :, :]))
        within_w = np.mean(np.abs(X_eff[:, None, :] - X_eff[None, :, :]))
        within_r = np.mean(np.abs(ref[:, None, :] - ref[None, :, :]))
        return float(2.0 * cross - within_w - within_r)

    def _sliced_w1_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None,
        n_slices: int = 8, seed: int = 42,
    ) -> float:
        """Sliced-1D-Wasserstein distance of one window vs the medoid cloud (v3.14.4).

        Projects both populations onto ``n_slices`` random unit directions
        (deterministic seed 42, same for calibration and live calls so the
        definition is bit-identical), computes the empirical 1D Wasserstein-1
        distance per slice (sorted samples, CDF integral via interpolation),
        and averages. Sees marginal distribution shifts that a moment-based
        (BW/MMD) or energy test can miss when the shift is anisotropic but
        preserves mean/covariance in the checked directions. Operates on the
        raw effective stream (same space as the mean-shift/energy terms).
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        ref = self._mean_shift_reference()
        d = X_eff.shape[1]
        rng = np.random.default_rng(int(seed))
        dirs = rng.normal(size=(int(n_slices), d))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dirs = dirs / norms
        total = 0.0
        for dir_vec in dirs:
            a = np.sort(X_eff @ dir_vec)
            b = np.sort(ref @ dir_vec)
            # empirical 1D W1: integral of |F_a - F_b| = mean|a_sorted - b_sorted|
            # on the common quantile grid when sizes differ.
            if len(a) == len(b):
                total += float(np.mean(np.abs(a - b)))
            else:
                # Quantile grid at the coarser resolution.
                n = max(len(a), len(b))
                qs = (np.arange(n) + 0.5) / n
                a_q = np.quantile(a, qs)
                b_q = np.quantile(b, qs)
                total += float(np.mean(np.abs(a_q - b_q)))
        return float(total / max(int(n_slices), 1))

    def _autocorr_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ) -> float:
        """Lag-1 autocorrelation distance of one window vs the medoid cloud (v3.14.4).

        Computes the mean absolute difference of per-feature lag-1
        autocorrelations: ``mean_f |corr_f(W) - corr_f(R)|`` where corr_f is
        the Pearson lag-1 correlation of feature f in the window vs the same
        in the reference medoid cloud. Sees temporal-structure shifts
        (oscillation / sticking / jitter) that marginal tests miss. Features
        with near-zero variance contribute 0. Operates on the raw effective
        stream time order (window rows in the order given).
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        ref = self._mean_shift_reference()
        if X_eff.shape[0] < 3 or ref.shape[0] < 3:
            return 0.0
        def lag1_corr(A: np.ndarray) -> np.ndarray:
            # per-feature Pearson lag-1, zero for degenerate features
            X0 = A[:-1] - np.mean(A[:-1], axis=0)
            X1 = A[1:] - np.mean(A[1:], axis=0)
            num = np.sum(X0 * X1, axis=0)
            den = np.sqrt(np.sum(X0 * X0, axis=0) * np.sum(X1 * X1, axis=0))
            out = np.zeros_like(num)
            ok = den > 1e-12
            out[ok] = num[ok] / den[ok]
            return out
        cw = lag1_corr(X_eff)
        cr = lag1_corr(ref)
        return float(np.mean(np.abs(cw - cr)))

    def _autocorr_lag2_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ) -> float:
        """Lag-2 autocorrelation distance of one window vs medoid cloud (v3.14.4).

        Same as :meth:`_autocorr_distance` but at lag 2. Sees longer-range
        temporal structure (period-2 oscillations, longer memory) that lag-1
        misses. Zero for windows shorter than 4 rows.
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        ref = self._mean_shift_reference()
        if X_eff.shape[0] < 4 or ref.shape[0] < 4:
            return 0.0
        def lag2_corr(A: np.ndarray) -> np.ndarray:
            X0 = A[:-2] - np.mean(A[:-2], axis=0)
            X1 = A[2:] - np.mean(A[2:], axis=0)
            num = np.sum(X0 * X1, axis=0)
            den = np.sqrt(np.sum(X0 * X0, axis=0) * np.sum(X1 * X1, axis=0))
            out = np.zeros_like(num)
            ok = den > 1e-12
            out[ok] = num[ok] / den[ok]
            return out
        cw = lag2_corr(X_eff)
        cr = lag2_corr(ref)
        return float(np.mean(np.abs(cw - cr)))

    def _rank_dfs_distance(
        self, X_stream, distiller_ref=None, remember_prev_row=False, prev_row=None
    ) -> float:
        """Mean rank correlation (Spearman) distance vs medoid cloud (v3.14.4).

        Per-feature Spearman rho between window and reference (rank then
        Pearson), returns ``mean_f |rho_f(W) - rho_f(R)|``. Sees monotonic
        dependence shifts that Pearson misses (nonlinear monotone drift).
        Rank computed via argsort; ties broken by stable sort.
        """
        X_eff = self._effective_stream(
            X_stream, remember_prev_row=remember_prev_row, prev_row=prev_row
        )
        ref = self._mean_shift_reference()
        if X_eff.shape[0] < 3 or ref.shape[0] < 3:
            return 0.0
        def ranks(A: np.ndarray) -> np.ndarray:
            return np.argsort(np.argsort(A, axis=0), axis=0).astype(np.float64)
        def spearman(A: np.ndarray) -> np.ndarray:
            R = ranks(A)
            Rm = R - np.mean(R, axis=0)
            den = np.sqrt(np.sum(Rm * Rm, axis=0))
            out = np.zeros_like(den)
            ok = den > 1e-12
            # rho between the two halves lag-1 as a proxy for dependence
            # structure; identical halves give rho=1 for identical order.
            # For the distance we compare the two populations' rank
            # autocorrelation proxy directly.
            X0 = R[:-1] - np.mean(R[:-1], axis=0)
            X1 = R[1:] - np.mean(R[1:], axis=0)
            num = np.sum(X0 * X1, axis=0)
            den2 = np.sqrt(np.sum(X0 * X0, axis=0) * np.sum(X1 * X1, axis=0))
            ok = den2 > 1e-12
            out[ok] = num[ok] / den2[ok]
            return out
        cw = spearman(X_eff)
        cr = spearman(ref)
        return float(np.mean(np.abs(cw - cr)))

    def _mean_shift_reference(self) -> np.ndarray:
        """Reference population for the raw-space mean-shift/energy terms.

        The crystal's stored medoids, in the SAME space as the effective
        stream: augmented with first differences for delta crystals
        ``[X || dX]`` (``dX[0] = 0``, ``dX[i] = X[i] - X[i-1]``), raw medoids
        otherwise. ``_mean_shift_distance`` built this population inline;
        factored out so the energy term uses bit-identical arithmetic
        (v3.14.3).
        """
        if self.include_deltas:
            dX = np.zeros_like(self.X)
            dX[1:] = self.X[1:] - self.X[:-1]
            return np.hstack([self.X, dX])
        return self.X

    def calibrate_threshold(self, X_ref, *, window: int, fa_rate: float = 0.01) -> dict:
        """Calibrate the radar threshold on healthy reference windows (v3.12.0).

        The default threshold is a constant calibrated on the UCI
        Superconductivity heal benchmark at passport rank 32; a different
        domain, rank or window size silently changed its meaning (the
        MetroPT protocol-v2 negative result had to push tau above every real
        leak to hold the false-alarm budget on healthy data). This method
        replaces the constant with a quantile of the radar's OWN distances
        over healthy windows from the deployment stream:

            tau = quantile(distances, 1 - fa_rate)

        so roughly ``fa_rate`` of windows drawn from the same healthy regime
        cross the threshold, whatever the domain, rank or window size.

        Parameters
        ----------
        X_ref:
            2-D healthy reference stream, at least ``10 * window`` rows.
        window:
            Rows per window (>= 8). Non-overlapping windows are taken from
            the start of the stream.
        fa_rate:
            Nominal false-alarm fraction among healthy windows, in (0, 0.5).

        Stores the calibration on the loader (``save()`` persists it) and
        returns a report dict. ``check_domain_shift`` picks it up
        automatically; ``heal_domain_shift`` invalidates it when the
        passport moves. The reference must be healthy with respect to the
        SAME crystal the deployment will use.
        """
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(1, -1)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must be 2-D (rows, features), got shape {X_ref.shape}")
        window = int(window)
        if window < 8:
            raise ValueError(f"window must be >= 8 rows, got {window}")
        if not 0.0 < float(fa_rate) < 0.5:
            raise ValueError(f"fa_rate must be in (0, 0.5), got {fa_rate}")
        n_windows = len(X_ref) // window
        if n_windows < 10:
            raise ValueError(
                f"a {1.0 - float(fa_rate):.0%} quantile needs at least 10 windows of "
                f"{window} rows; got {n_windows} from {len(X_ref)} rows -- provide at "
                f"least {10 * window} healthy rows"
            )
        dists = np.empty(n_windows, dtype=np.float64)
        for w_i in range(n_windows):
            dists[w_i], _ = self._bw_distance(X_ref[w_i * window:(w_i + 1) * window])
        tau = float(np.quantile(dists, 1.0 - float(fa_rate)))
        self.tau_calibrated = tau
        self.tau_window = window
        self.tau_n_windows = n_windows
        self.tau_fa_rate = float(fa_rate)
        return {
            "tau": tau,
            "window": window,
            "n_windows": n_windows,
            "fa_rate": float(fa_rate),
            "dist_median": float(np.median(dists)),
            "dist_max": float(np.max(dists)),
        }

    def calibrate_threshold_ms(self, X_ref, *, window: int, fa_rate: float = 0.01) -> dict:
        """Calibrate the combined radar's fused threshold on healthy windows (v3.13.0).

        The combined radar alarms on ``bw_distance + mean_shift_distance``.
        Since the mean-shift term is a z-score that follows the same
        "larger window -> smaller spread" scaling as the BW distance, the
        fused score needs its OWN threshold calibrated the same way
        (``calibrate_threshold``) on HEALTHY windows from the deployment
        stream, over the same non-overlapping windows. Run this after
        ``calibrate_threshold`` -- the two thresholds share the window size
        and are persisted (and invalidated) together by ``save()`` /
        ``heal_domain_shift``.

        Returns a report dict with the fused ``tau``; ``check_domain_shift``
        picks it up automatically and ``is_shifted`` then reflects the
        combined score.
        """
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(1, -1)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must be 2-D (rows, features), got shape {X_ref.shape}")
        window = int(window)
        if window < 8:
            raise ValueError(f"window must be >= 8 rows, got {window}")
        if not 0.0 < float(fa_rate) < 0.5:
            raise ValueError(f"fa_rate must be in (0, 0.5), got {fa_rate}")
        n_windows = len(X_ref) // window
        if n_windows < 10:
            raise ValueError(
                f"a {1.0 - float(fa_rate):.0%} quantile needs at least 10 windows of "
                f"{window} rows; got {n_windows} from {len(X_ref)} rows -- provide at "
                f"least {10 * window} healthy rows"
            )
        combined = np.empty(n_windows, dtype=np.float64)
        for w_i in range(n_windows):
            sl = X_ref[w_i * window:(w_i + 1) * window]
            bw, _ = self._bw_distance(sl)
            ms, _ = self._mean_shift_distance(sl)
            combined[w_i] = bw + ms
        tau = float(np.quantile(combined, 1.0 - float(fa_rate)))
        self.ms_tau_calibrated = tau
        self.ms_tau_window = window
        self.ms_tau_n_windows = n_windows
        self.ms_tau_fa_rate = float(fa_rate)
        return {
            "tau": tau,
            "window": window,
            "n_windows": n_windows,
            "fa_rate": float(fa_rate),
            "combined_median": float(np.median(combined)),
            "combined_max": float(np.max(combined)),
        }

    def calibrate_threshold_panel(self, X_ref, *, window: int, fa_rate: float = 0.01, e_process: bool = False) -> dict:
        """Calibrate the panel radar's fused threshold on healthy windows (v3.14.3).

        The panel radar alarms on ``bw + mean_shift + mmd + energy``. Like
        the mean-shift calibration, the fused score needs its OWN threshold
        calibrated the same way (``calibrate_threshold``) on HEALTHY windows
        from the deployment stream, over the same non-overlapping windows.
        Run this after ``calibrate_threshold`` — the three thresholds share
        the window size and are persisted (and invalidated) together by
        ``save()`` / ``heal_domain_shift``.

        Returns a report dict with the fused ``tau``; ``check_domain_shift``
        picks it up automatically and ``panel_is_shifted`` then reflects the
        panel score.
        """
        # v3.14.4: e-process option. When ``e_process=True`` the alarm is
        # anytime-valid: the panel score is compared against an e-value
        # average (E_t / t) instead of a fixed quantile, so the threshold
        # grows with evidence rather than burning a fixed false-alarm budget
        # on the first window. The state is kept on the loader
        # (``_e_value`` / ``_e_t``), reset by ``reset_e_process()`` and by
        # ``heal_domain_shift`` (same invalidation as the quantile taus).
        # Default False keeps the existing behaviour bit-identical.
        use_e = bool(e_process)
        X_ref = np.asarray(X_ref, dtype=np.float64)
        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(1, -1)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must be 2-D (rows, features), got shape {X_ref.shape}")
        window = int(window)
        if window < 8:
            raise ValueError(f"window must be >= 8 rows, got {window}")
        if not 0.0 < float(fa_rate) < 0.5:
            raise ValueError(f"fa_rate must be in (0, 0.5), got {fa_rate}")
        n_windows = len(X_ref) // window
        if n_windows < 10:
            raise ValueError(
                f"a {1.0 - float(fa_rate):.0%} quantile needs at least 10 windows of "
                f"{window} rows; got {n_windows} from {len(X_ref)} rows -- provide at "
                f"least {10 * window} healthy rows"
            )
        panel = np.empty(n_windows, dtype=np.float64)
        for w_i in range(n_windows):
            sl = X_ref[w_i * window:(w_i + 1) * window]
            bw, _ = self._bw_distance(sl)
            ms, _ = self._mean_shift_distance(sl)
            mmd = self._mmd_distance(sl)
            energy = self._energy_distance(sl)
            sliced = self._sliced_w1_distance(sl)
            autocorr = self._autocorr_distance(sl)
            autocorr2 = self._autocorr_lag2_distance(sl)
            rank_dfs = self._rank_dfs_distance(sl)
            panel[w_i] = bw + ms + mmd + energy + sliced + autocorr + autocorr2 + rank_dfs
        tau = float(np.quantile(panel, 1.0 - float(fa_rate)))
        self.panel_tau_calibrated = tau
        self.panel_tau_window = window
        self.panel_tau_n_windows = n_windows
        self.panel_tau_fa_rate = float(fa_rate)
        if use_e:
            # Initialise e-process state to the median panel score so the
            # first window's e-value is neutral; the e-value grows only when
            # the panel distance exceeds the calibrated quantile.
            self._e_value = float(np.median(panel))
            self._e_t = 0
        return {
            "tau": tau,
            "window": window,
            "n_windows": n_windows,
            "fa_rate": float(fa_rate),
            "panel_median": float(np.median(panel)),
            "panel_max": float(np.max(panel)),
            "e_process": use_e,
        }

    def reset_e_process(self) -> None:
        """Reset the anytime-valid e-process evidence (v3.14.4).

        Clears ``_e_value`` / ``_e_t`` so the next ``check_domain_shift``
        starts a fresh e-value sequence. ``heal_domain_shift`` calls this
        automatically when the passport moves.
        """
        self._e_value = None
        self._e_t = 0

    def _e_process_alarm(self, panel_distance: float) -> dict:
        """Anytime-valid e-process verdict for one panel score (v3.14.4).

        Maintains an e-value sequence: each window multiplies its e-value by
        ``exp(panel_distance / tau - 1)`` (tau = calibrated quantile),
        clamped so a single window cannot create infinite evidence. The
        alarm fires when the running average ``E_t / t`` exceeds ``1/alpha``
        (``alpha = fa_rate``). Under the null the e-value is a martingale,
        so the anytime-valid threshold holds without re-calibration per
        window. Returns ``{"e_value", "e_t", "e_average", "e_alarm"}``.
        """
        tau = float(self.panel_tau_calibrated) if self.panel_tau_calibrated is not None else float(BW_ALERT_THRESHOLD)
        alpha = float(self.panel_tau_fa_rate) if self.panel_tau_fa_rate is not None else 0.01
        if getattr(self, "_e_value", None) is None:
            self._e_value = float(tau)
            self._e_t = 0
        # e-value increment: >1 when panel_distance > tau, <1 when below.
        ratio = float(panel_distance) / max(tau, 1e-9)
        incr = float(np.exp(float(np.clip(ratio - 1.0, -2.0, 5.0))))
        self._e_value = float(self._e_value * incr)
        self._e_t = int(self._e_t) + 1
        e_avg = float(self._e_value) / max(int(self._e_t), 1)
        threshold = 1.0 / max(float(alpha), 1e-9)
        return {
            "e_value": float(self._e_value),
            "e_t": int(self._e_t),
            "e_average": float(e_avg),
            "e_threshold": float(threshold),
            "e_alarm": bool(e_avg > threshold),
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
        streaming: bool = False,
        new_weight_boost: float = 1.0,
        passport_blend_min: float = 0.15,
        passport_blend_max: float = 0.50,
        protect_old_weight_frac: float = 0.0,
        passport_recalibrate: bool = False,
    ) -> dict:
        """Surgical self-healing: inject new Voronoi medoids when OOD shift is detected.

        Checks the Bures-Wasserstein radar on ``X_new_stream``. If a shift is
        detected (or ``force=True``), selects ``n_inject`` real medoids via
        k-means++ + Lloyd relaxation in normalized space, appends them to the
        crystal (bounded forgetting: existing medoids are kept unless a cap is
        set), and updates the 32-mode spectral passport in place. If
        ``max_crystal_size`` is set and exceeded, prunes only low-weight
        background medoids while protecting the phase-peak band and newly
        injected medoids; the protected band follows the crystal's ``tail``
        direction (top by ``y`` for ``"high"``, bottom for ``"low"``, most
        extreme for ``"both"``). The cap is hard: if it cannot accommodate the
        injected medoids plus the protected old band, a ``ValueError`` is raised
        instead of silently returning an oversized crystal.

        With ``streaming=True`` (delta crystals), the embed of
        ``X_new_stream`` continues the first-difference sequence from the
        previous radar/heal call; the internal post-heal re-check stays
        stateless.

        ``new_weight_boost`` multiplies the injected medoids' weights before
        renormalisation (default 1.0; >1 gives the new regime more influence).
        ``passport_blend_min``/``passport_blend_max`` bound the passport blend
        coefficient (defaults 0.15/0.50); raising the floor helps repeated
        heals converge faster on strong drift. ``protect_old_weight_frac``
        additionally protects the top-weight fraction of old medoids from
        pruning (default 0.0), which trades new-domain capacity for retention.
        ``passport_recalibrate=True`` replaces the passport with the new
        window's covariance instead of blending it. Subsequent windows from the
        new domain are then in-distribution, while the old domain is flagged as
        shifted; use a :class:`~shd.bank.DomainBank` if both domains must be
        recognized simultaneously. The default ``False`` preserves the blended
        behavior.

        Threading (v3.14.4): acquires the loader lock so concurrent heals
        and radar calls serialize instead of racing; re-entrant so the
        internal check_domain_shift call on the same thread does not deadlock.
        """
        with self._lock:
            return self._heal_domain_shift_locked(
                X_new_stream, y_new_stream, n_inject, max_crystal_size,
                distiller_ref, force, random_state, streaming,
                new_weight_boost, passport_blend_min, passport_blend_max,
                protect_old_weight_frac, passport_recalibrate,
            )

    def _heal_domain_shift_locked(
        self,
        X_new_stream: np.ndarray,
        y_new_stream: np.ndarray | None = None,
        n_inject: int = 35,
        max_crystal_size: int | None = None,
        distiller_ref=None,
        force: bool = False,
        random_state: int = 42,
        streaming: bool = False,
        new_weight_boost: float = 1.0,
        passport_blend_min: float = 0.15,
        passport_blend_max: float = 0.50,
        protect_old_weight_frac: float = 0.0,
        passport_recalibrate: bool = False,
    ) -> dict:
        t0 = time.perf_counter()
        X_win = np.asarray(X_new_stream, dtype=np.float64)
        if X_win.ndim == 1:
            X_win = X_win.reshape(1, -1)
        if X_win.ndim != 2:
            raise ValueError(f"stream window must be 2-D, got shape {X_win.shape}")
        # Stateless pre-check on purpose: forwarding streaming=True here would
        # advance _prev_row, and _prepare_stream below would then compute the
        # first-difference dX[0] against the window's own last row.
        shift = self.check_domain_shift(X_win, distiller_ref=distiller_ref)
        if not shift["is_shifted"] and not force:
            # Keep the first-difference chain intact for delta crystals: a
            # no-heal call must advance _prev_row exactly like a radar call
            # with streaming=True would (raw-width windows only, mirroring
            # _normalize_stream's rule for pre-augmented windows).
            if streaming and self.include_deltas and len(X_win) > 0:
                d_raw = len(self.X_mean) // 2
                if X_win.shape[1] == d_raw:
                    self._prev_row = X_win[-1].copy()
            return {
                "healed": False,
                "n_injected": 0,
                "n_pruned": 0,
                "bw_before": shift["bw_distance"],
                "bw_after": shift["bw_distance"],
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        rng = np.random.default_rng(random_state)

        # Window row count computed BEFORE any projection: the cap contract must
        # be checked before _prepare_stream advances the streaming continuity
        # state, so a rejected heal leaves the loader completely untouched.
        n_total = int(X_win.shape[0])
        n_sel = min(int(n_inject), n_total)

        # Validate an impossible memory cap before mutating the loader. The
        # phase-peak band and all injected medoids are protected, so the cap
        # must be able to hold at least those. The two protected index sets can
        # overlap (a phase-peak medoid may also carry a high weight), so the
        # requirement is the size of their UNION, not the sum of the two sizes:
        # a peak-5 + top-weight-5 set that is the same 5 medoids needs 5 slots,
        # not 10 (the exact count is re-checked after the protected set is
        # built).
        if max_crystal_size is not None:
            cap_req = int(max_crystal_size)
            if cap_req < 1:
                raise ValueError(f"max_crystal_size must be >= 1, got {max_crystal_size}")
            n_old_now = len(self.X)
            protected_pre = np.zeros(n_old_now, dtype=bool)
            # v3.12.0: the protected band is the SAME set the peak slice would
            # serve -- the tail direction, or y >= critical_threshold for a
            # threshold-distilled crystal. The inline tail logic protected
            # low-side medoids for a tail="both" + critical_threshold crystal
            # whose band was built from the UPPER side.
            protected_pre[self._peak_rank_idx(n_old_now, self.high_band_ratio, y=self.y[:n_old_now])] = True
            n_protect_weight = int(round(float(protect_old_weight_frac) * n_old_now))
            if n_protect_weight > 0:
                protected_pre[np.argsort(self.weights[:n_old_now])[-n_protect_weight:]] = True
            min_required = n_sel + int(protected_pre.sum())
            if cap_req < min_required:
                raise ValueError(
                    f"max_crystal_size={cap_req} cannot accommodate {n_sel} injected "
                    f"medoids plus the protected band/weights (>= {min_required}); "
                    f"use max_crystal_size >= {min_required} or reduce n_inject"
                )

        Z_new, U_new = self._prepare_stream(
            X_win, distiller_ref=distiller_ref, remember_prev_row=streaming
        )

        # k-means++ seeding in Z-space
        first_idx = int(rng.integers(0, n_total))
        selected = [first_idx]
        min_sq = np.sum((Z_new - Z_new[first_idx]) ** 2, axis=1)
        for _ in range(1, n_sel):
            total_sq = float(np.sum(min_sq))
            if total_sq <= 0.0 or not np.isfinite(total_sq):
                # Degenerate window (all rows identical, e.g. a frozen sensor):
                # every point is equally close to the current centers, so the
                # D^2 seeding distribution is all zeros and rng.choice would
                # reject probabilities that do not sum to 1. Fall back to
                # uniform selection, matching _seed_kmeans_pp's degenerate path.
                nxt = int(rng.integers(0, n_total))
            else:
                probs = min_sq / (total_sq + 1e-12)
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
        w_patch = (counts / np.mean(counts)) * float(new_weight_boost)

        X_patch = X_win[med_idx].copy()
        if X_patch.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"heal patch has {X_patch.shape[1]} features but the crystal stores "
                f"{self.X.shape[1]}; pass raw feature windows to heal (pre-augmented "
                "[x || dx] windows are radar-only)"
            )
        if y_new_stream is not None:
            y_arr = np.asarray(y_new_stream, dtype=np.float64).reshape(-1)
            if len(y_arr) != n_total:
                raise ValueError(
                    f"y_new_stream has {len(y_arr)} labels for {n_total} window rows"
                )
            if not np.isfinite(y_arr).all():
                raise ValueError(
                    "y_new_stream contains NaN/Inf; heal refuses to poison the "
                    "crystal with non-finite targets (the distiller enforces the "
                    "same policy on y via on_nonfinite)"
                )
            # reshape(-1), not squeeze: a single-row window's (1,) y would
            # squeeze to a 0-d array and 0-d[med_idx] raises IndexError.
            y_patch = y_arr[med_idx].copy()
            labelled_patch = np.ones(len(med_idx), dtype=bool)
        else:
            # Unsupervised fallback: spectral novelty score against current passport.
            #
            # This is NOT a target value -- it lives on a different scale from
            # the real targets already stored in crystal_Y. Writing it into the
            # same column silently produced a crystal whose targets were a
            # mixture of two incompatible quantities (measured: real targets
            # in [84.9, 113.0] next to novelty scores in [0.66, 0.91]), which
            # breaks get_training_data() for any downstream model.
            #
            # A crystal distilled WITHOUT labels has no real targets at all, so
            # its column already holds the synthetic novelty coordinate and the
            # injected rows are consistent with it. A crystal distilled WITH
            # labels gets a neutral placeholder instead, and the row is marked
            # unlabelled so the loader can drop it on request.
            if self._has_real_targets:
                y_patch = np.full(len(med_idx), self._target_placeholder, dtype=np.float64)
                labelled_patch = np.zeros(len(med_idx), dtype=bool)
            else:
                proj = U_new[med_idx] @ self.eigvecs
                y_patch = np.sum(U_new[med_idx] ** 2, axis=1) - np.sum(proj ** 2, axis=1)
                labelled_patch = np.ones(len(med_idx), dtype=bool)

        n_old = len(self.X)
        self.X = np.vstack([self.X, X_patch])
        self.y = np.concatenate([self.y, np.atleast_1d(y_patch)])
        w_all = np.concatenate([self.weights, w_patch])
        self.weights = w_all / np.mean(w_all)
        # Track which rows carry a real target, so a downstream model can drop
        # the unlabelled ones instead of training on placeholder values.
        self.y_labelled = np.concatenate(
            [self.y_labelled, np.asarray(labelled_patch, dtype=bool)]
        )

        n_pruned = 0
        if max_crystal_size is not None and len(self.X) > int(max_crystal_size):
            cap = int(max_crystal_size)
            if cap < 1:
                raise ValueError(f"max_crystal_size must be >= 1, got {max_crystal_size}")
            excess = len(self.X) - cap
            if excess > 0:
                # Protect the phase-peak band among old medoids + all new
                # medoids; the band is the same set the peak slice would
                # serve (tail direction, or the critical threshold's upper
                # side), sized by the persisted high_band_ratio (v3.12.0).
                peak_old_idx = self._peak_rank_idx(n_old, self.high_band_ratio, y=self.y[:n_old])
                protected = np.zeros(len(self.X), dtype=bool)
                protected[peak_old_idx] = True
                n_protect_weight = int(round(float(protect_old_weight_frac) * n_old))
                if n_protect_weight > 0:
                    top_w_idx = np.argsort(self.weights[:n_old])[-n_protect_weight:]
                    protected[top_w_idx] = True
                protected[n_old:] = True
                n_protected = int(protected.sum())
                if cap < n_protected:
                    raise ValueError(
                        f"max_crystal_size={cap} cannot accommodate {n_sel} injected "
                        f"medoids plus {n_protected - n_sel} protected old medoids; "
                        f"use max_crystal_size >= {n_protected} or reduce n_inject"
                    )
                prunable_idx = np.where(~protected)[0]
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
                self.y_labelled = self.y_labelled[keep_mask]
                nearest = np.argmin(cdist(X_drop, self.X, metric="sqeuclidean"), axis=1)
                np.add.at(self.weights, nearest, w_drop)
                self.weights /= np.mean(self.weights)
                n_pruned = int(n_drop)
                if len(self.X) > cap:  # defensive: the hard cap is a contract
                    raise RuntimeError(
                        f"internal error: healed crystal has {len(self.X)} medoids, cap={cap}"
                    )

        # Blend the spectral passport with the new regime so subsequent radar
        # checks move toward the new domain. One heal does not necessarily clear
        # the alert (the SC benchmark leaves BW at ~0.465 > 0.085); repeated
        # heals or a domain bank may be needed.
        #
        # v3.15.1 (Banner fix 3): the RETURNED dict makes the partial nature
        # explicit -- bw_after may still exceed the radar threshold, and
        # bw_after-vs-threshold tells the caller whether an alarm would keep
        # firing. `is_shifted` on the post-heal re-check is the previous
        # behavior preserved for compatibility; the new explicit fields
        # remove the need to infer partial healing from a silent invariant.
        Psi_new = (U_new.T @ U_new) / max(len(U_new), 1)
        if passport_recalibrate:
            alpha_blend = 1.0
        else:
            alpha_blend = min(float(passport_blend_max),
                              max(float(passport_blend_min),
                                  len(U_new) / (self.n_seen + len(U_new))))
        Psi_blend = (1.0 - alpha_blend) * self.Psi_passport + alpha_blend * Psi_new
        Psi_blend = 0.5 * (Psi_blend + Psi_blend.T)
        rank_p = len(self.eigvals)
        eigvals, eigvecs = linalg.eigh(Psi_blend)
        self.eigvals = eigvals[-rank_p:].astype(np.float64)
        self.eigvecs = eigvecs[:, -rank_p:].astype(np.float64)
        self.Psi_passport = (self.eigvecs * self.eigvals) @ self.eigvecs.T
        self.n_seen += n_total
        # v3.12.0: the passport just moved (blended or replaced), so a
        # threshold calibrated against the OLD eigenpairs no longer
        # describes this radar. Drop the calibration instead of silently
        # scoring windows against a stale quantile.
        self.tau_calibrated = None
        self.tau_window = None
        self.tau_n_windows = None
        self.tau_fa_rate = None
        # v3.13.0: the combined-radar calibration is invalidated by the same
        # passport move; clear it together with the BW threshold.
        self.ms_tau_calibrated = None
        self.ms_tau_window = None
        self.ms_tau_n_windows = None
        self.ms_tau_fa_rate = None
        # v3.14.4: the e-process evidence is invalidated with the passport
        # thresholds when the passport moves.
        # v3.14.3: the panel calibration is invalidated by the same passport
        # move; clear it together with the other two thresholds.
        # v3.14.4: the e-process evidence is invalidated with the passport
        # thresholds when the passport moves.
        self.reset_e_process()
        self.panel_tau_calibrated = None
        self.panel_tau_window = None
        self.panel_tau_n_windows = None
        self.panel_tau_fa_rate = None

        after = self.check_domain_shift(X_win, distiller_ref=distiller_ref)
        return {
            "healed": True,
            "n_injected": int(n_sel),
            "n_pruned": int(n_pruned),
            "n_unlabelled_injected": int(np.sum(~np.asarray(labelled_patch, dtype=bool))),
            "still_shifted": bool(after["is_shifted"]),
            "threshold": float(after["threshold"]),
            "threshold_source": after["threshold_source"],
            "warning": (
                "label-free heal: the injected rows carry a placeholder target, "
                "not a measured one; drop them with get_training_data("
                "drop_unlabelled=True) (the default since v3.11.1)"
                if y_new_stream is None and self._has_real_targets else None
            ),
            "bw_before": shift["bw_distance"],
            "bw_after": after["bw_distance"],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def save(self, filepath: str | None = None) -> float:
        """Persist the (possibly self-healed) crystal back to ``.hologram.npz``.

        Returns the saved file size in KB. Acquires the loader lock so a
        concurrent heal cannot mutate fields mid-write (v3.14.4).
        """
        with self._lock:
            out_path = filepath or self.filepath
            payload = dict(
            # A healed crystal keeps the SAME schema contract as its parent
            # artifact (v3.13.1): the heal adds medoids, not new fields.
            schema_version=np.array([HologramCrystalLoader.LOADER_SCHEMA], dtype=np.int64),
            schema_min_loader=np.array([HologramCrystalLoader.LOADER_SCHEMA], dtype=np.int64),
            shd_version=np.array(SHD_VERSION),
            crystal_X=self.X.astype(np.float32),
            crystal_Y=self.y.astype(np.float32),
            gamma_weights=self.weights.astype(np.float32),
            passport_eigvals=self.eigvals.astype(np.float16),
            passport_eigvecs=self.eigvecs.astype(np.float16),
            X_mean=self.X_mean.astype(np.float32),
            X_std=self.X_std.astype(np.float32),
            n_seen=np.array([self.n_seen], dtype=np.int64),
            tail=np.array(self.tail),
            include_deltas=np.array(bool(self.include_deltas)),
            high_band_ratio=np.array([float(self.high_band_ratio)], dtype=np.float64),
            passport_reference=np.array(self.passport_reference),
            y_labelled=self.y_labelled.astype(bool),
            # v3.14.2: the label-free marker travels with the re-saved crystal.
            label_free=np.array(bool(getattr(self, "label_free", False))),
            critical_threshold=np.array(
                [np.nan if self.critical_threshold is None else float(self.critical_threshold)],
                dtype=np.float64,
            ),
        )
        # v3.15.2: keep the operating point when a healed crystal is
        # re-saved (the heal does not change it).
        if getattr(self, "tail_quantile", None) is not None:
            payload["tail_quantile"] = np.array([float(self.tail_quantile)], dtype=np.float64)
        if getattr(self, "phase_kappa", None) is not None:
            payload["phase_kappa"] = np.array([float(self.phase_kappa)], dtype=np.float64)
        if getattr(self, "tau_calibrated", None) is not None:
            # v3.12.0: the per-deployment quantile calibration travels with
            # the crystal. heal_domain_shift clears it when the passport moves.
            payload.update(_calibration_fields(
                "tau", self.tau_calibrated, self.tau_window,
                self.tau_n_windows, self.tau_fa_rate,
            ))
        if getattr(self, "ms_tau_calibrated", None) is not None:
            # v3.13.0: the combined-radar fused-threshold calibration travels
            # with the crystal the same way.
            payload.update(_calibration_fields(
                "ms_tau", getattr(self, "ms_tau_calibrated", None),
                getattr(self, "ms_tau_window", None),
                getattr(self, "ms_tau_n_windows", None),
                getattr(self, "ms_tau_fa_rate", None),
            ))
        if getattr(self, "panel_tau_calibrated", None) is not None:
            # v3.14.3: the panel-radar fused-threshold calibration travels
            # with the crystal the same way.
            payload.update(_calibration_fields(
                "panel_tau", getattr(self, "panel_tau_calibrated", None),
                getattr(self, "panel_tau_window", None),
                getattr(self, "panel_tau_n_windows", None),
                getattr(self, "panel_tau_fa_rate", None),
            ))
        if self.Omega is not None and self.phase is not None:
            payload["Omega"] = self.Omega.astype(np.float32)
            payload["phase"] = self.phase.astype(np.float32)
        out_path = npz_output_path(out_path)
        np.savez_compressed(out_path, **payload)
        return os.path.getsize(out_path) / 1024.0

    def summary(self) -> dict:
        """Return a JSON-serializable metadata summary of the loaded crystal."""
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        size_kb = os.path.getsize(self.filepath) / 1024.0 if os.path.exists(self.filepath) else 0.0
        return {
            "filepath": self.filepath,
            "file_size_kb": round(float(size_kb), 2),
            "schema_version": int(self.schema_version),
            "shd_version": self.shd_version,
            "n_medoids": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "n_seen": int(self.n_seen),
            "compression_ratio": round(float(self.n_seen) / max(len(self.X), 1), 2),
            "sketch_dim": int(self.k_dim),
            "passport_rank": int(len(self.eigvals)),
            "tail": self.tail,
            "include_deltas": bool(self.include_deltas),
            "has_projector": bool(self.Omega is not None),
            "mean_shift_calibrated": bool(getattr(self, "ms_tau_calibrated", None) is not None),
            "panel_calibrated": bool(getattr(self, "panel_tau_calibrated", None) is not None),
            "passport_reference": self.passport_reference,
            # v3.15.2: the operating point, when the artifact carries it.
            "tail_quantile": (None if getattr(self, "tail_quantile", None) is None
                              else float(self.tail_quantile)),
            "phase_kappa": (None if getattr(self, "phase_kappa", None) is None
                            else float(self.phase_kappa)),
            "n_unlabelled": int(np.sum(~self.y_labelled)),
            "y_min": float(np.min(self.y)),
            "y_median": float(np.median(self.y)),
            "y_max": float(np.max(self.y)),
        }



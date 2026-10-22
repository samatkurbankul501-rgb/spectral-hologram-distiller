# Spectral Hologram Distiller (SHD v3.12.0)

> **v3.12.0 (2026-10-22): round-2 repairs + per-deployment radar calibration.**
> The second defect hunt (probes in `tools/hunt/round2_probes.py`, regressions
> in `tests/test_v312_repairs.py`) found and fixed three more suppressors of
> the method's true performance, each reproduced live before the fix: a
> `tail="both"` crystal distilled with `critical_threshold` was read back by
> extremeness-from-median, so the peak slice, the budget slice and heal's
> protected band could serve LOW-side rows the band was never built from (the
> threshold now travels with the artifact); the rebuild reservoir's peak
> stratum kept the FIRST `peak_cap` peaks and silently dropped every later
> one, so a target drift past the trigger lost its late part (peaks are now
> sampled uniformly, Vitter algorithm R); and the radar threshold was
> hard-wired for passport rank 32, so `rank_passport >= 128` tripped false
> alarms on in-distribution windows (the base now scales with
> `sqrt(rank/32)`). New: `loader.calibrate_threshold(X_healthy, window=...,
> fa_rate=...)` — a per-deployment quantile of the radar's own distances over
> healthy windows, persisted with the crystal and invalidated by heal — the
> recommended replacement for the cross-dataset `0.085` constant.

> **Repair status (v3.11.1, 2026-10-21):** the v3.11.0 code review found and
> fixed four defects in this tree, each reproduced before the fix:
> the anytime budget (`get_training_data(budget_points=...)`) spent its whole
> budget on background cells and starved the phase-peak band; the peak slice
> ignored the crystal's persisted `high_band_ratio`; a label-free heal wrote
> placeholder targets without saying so; and `critical_threshold` combined
> with `tail="low"` produced a self-contradictory crystal. The experimental
> background-only spectral passport is now opt-in
> (`--passport-reference background`) instead of the default, because it
> changes the radar's sensitivity and the published artifacts were distilled
> with the stream reference. A controlled benchmark
> (`bench_passport_reference.py`) then measured that the background reference
> changes no outcome and does not repair the early-warning failure, so it
> stays experimental — see "Known limitations". Regression tests for all five
> live in `tests/test_v311_repairs.py`, and the whole workspace's audit chain
> now lives in this repository (`tools/`, `docs/MIGRATION.md`).

> **v3.11.0:** `shd_watch.py` (learn/calibrate/watch, protocol v2) with the
> honest MetroPT-3 negative result, the Scania APS official-cost benchmark, the
> corrected latency artifact, bank scaling, the PM2.5 rehab + 300-medoid
> guard, `ARTIFACTS.md` / `PITCH.md` / `EMAIL_ENGINEER.md`, and the paper
> update. Published JSONs below are historical experiments, not reruns of
> these changes. In particular, old Metro budget claims and oversize latency
> labels are invalid; see §9 and `ARTIFACTS.md`. No new release or benchmark
> parity is implied by the version string.

> **v3.10.0:** `DomainBank` (per-domain crystals + nearest-passport routing),
> `passport_recalibrate=True` for single-crystal heals, `weighting="uniform"` on
> the loader, PM2.5 section (§6 operating-point presets and the known-limitations
> entry below); plus the v3.9.2 hard cap, NPZ `--target-col`, golden regression
> test, and MIT LICENSE.

**Zero-GPU streaming dataset distillation with phase-manifold Voronoi medoids, kernel-herding spectral weights, and a 16-bit spectral-passport OOD radar.**

---

## What it does

Compress a streaming dataset into **0.3%–1.8% of the training rows it sees** —
120–300 *real* medoids
(a `.hologram.npz` crystal, tens of KB) — while (a) keeping rare critical regimes
(phase transitions, high-$T_c$ superconductors) from being absorbed by the background
and (b) shipping a millisecond OOD radar that detects domain shift *without the
original data*. A larger budget (`M=600`, ≈3.5% of the superconductivity training
rows) buys more critical-regime fidelity; see §6 for the operating-point presets.

Honest terminology: the spectral reweighting step is **kernel herding** solved with
FISTA; the cell-mass prior with root tempering `p^0.55` is **importance weighting**.
(Earlier notes called these "Hadamard-Gram quadrature" and "Radon-Nikodym
tempering" — same math.)

## Install

```bash
pip install -e .
# or point PYTHONPATH at the repo root; dependencies: numpy, scipy, scikit-learn
```

## Quickstart

```python
from shd import SpectralHologramDistiller, HologramCrystalLoader

d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                              sketch_dim=256, tail_quantile=84.0, phase_kappa=3.2,
                              max_init_buffer=8192,
                              tail="high",          # "high" | "low" | "both"
                              on_nonfinite="raise", # or "drop" to skip dirty rows
                              )
for i in range(0, len(X_stream), 2500):          # O(1) RAM: peak < 60 MB
    d.partial_fit(X_stream[i:i+2500], y_stream[i:i+2500])  # or y_batch=None (label-free!)
rel_spec_err = d.finalize_crystal()               # kernel-herding weights gamma*
d.save_hologram("my_crystal.hologram.npz")        # stores the RFF projector too

loader = HologramCrystalLoader("my_crystal.hologram.npz")
X_c, y_c, w_c = loader.get_training_data()        # train any model on 1.7% of data
X_pk, y_pk, w_pk = loader.get_phase_peak_slice()  # rare / discontinuity slice for BO
report = loader.check_domain_shift(X_new_stream)  # BW radar, self-contained
print(report)  # {'bw_distance': ..., 'latency_ms': ..., 'is_shifted': False}
# Delta crystals can carry first differences across radar calls:
# loader.check_domain_shift(X_new_stream, streaming=True)
# v3.12.0: calibrate the radar threshold on YOUR healthy windows (recommended
# over the cross-dataset 0.085 constant; persisted with the crystal):
loader.calibrate_threshold(X_healthy, window=256, fa_rate=0.01)
report = loader.check_domain_shift(X_new_stream)  # threshold_source: "calibrated"

# Surgical self-healing when a new regime arrives (bounded forgetting):
heal_info = loader.heal_domain_shift(X_drifted, y_drifted, n_inject=35)
loader.save("my_crystal_healed.hologram.npz")

# Accept the new domain explicitly (the old domain will be flagged):
# loader.heal_domain_shift(..., passport_recalibrate=True)
# Or keep both domains in-distribution with a bank:
# from shd import DomainBank
# bank = DomainBank(["a.hologram.npz", "b.hologram.npz"])

# Per-deployment weighting (see bench_weights_*):
X_g, y_g, w_g = loader.get_training_data(weighting="gamma")    # default
X_u, y_u, w_u = loader.get_training_data(weighting="uniform")  # critical-only
```

## CLI (`python -m shd` / `shd`)

```bash
# Inspect any .hologram.npz crystal
python -m shd inspect artifacts/superconduct_v37.hologram.npz

# Distill a CSV/NPY/NPZ stream (omit --target-col for label-free sensor mode)
python -m shd distill --input stream.csv --target-col -1 --output crystal.hologram.npz
# ...or make the crystal sensitive to rate-of-change transients:
python -m shd distill --input sensors.csv --target-col -1 --include-deltas --output crystal.hologram.npz
# ...protect the LOW tail (pressure/voltage sags) and tolerate dirty sensor rows:
python -m shd distill --input sensors.csv --target-col -1 --tail low --on-nonfinite drop --output crystal.hologram.npz

# Run the Bures-Wasserstein OOD radar on a new batch
python -m shd radar --crystal crystal.hologram.npz --input new_batch.csv --target-col -1
# (delta crystals: --streaming carries first differences across radar calls)

# Surgically inject new Voronoi medoids on domain shift and save
python -m shd heal --crystal crystal.hologram.npz --input drifted.csv --target-col -1 --output healed.hologram.npz
# ...with a hard cap on total crystal size (prunes low-weight background only):
python -m shd heal --crystal crystal.hologram.npz --input drifted.csv --target-col -1 --output healed.hologram.npz --max-crystal-size 220
```

## Pipeline

1. **RFF sketch (Bochner):** `U = cos(Z·Ω + φ)`, 256 harmonics, row-normalised;
   online accumulators `Ψ = UᵀU`, `Φ = Uᵀy`, with a hard warm-up buffer cap
   `max_init_buffer = 8192` so memory remains strictly $\mathcal{O}(1)$ even for
   large `crystal_size`.
2. **Residual / leverage trigger (supervised & label-free):** ridge predictor on
   the sketch; top-`q` percentile of `residual_ratio + 2·y_z²` marks phase-peak
   candidates (`y_z` direction follows `tail`: `high` default | `low` | `both`).
   When `y_batch=None` (label-free sensor telemetry), uses Bochner
   spectral leverage $s_i = u_i^T (\hat{\Psi} + \beta_{\text{low}} I_k)^{-1} u_i$
   as the synthetic novelty coordinate on `[Z ‖ κ·s_norm]`.
3. **Phase-manifold lattice:** one non-overlapping Voronoi tessellation on
   `[Z ‖ κ·y]` (κ = 3.2), medoid-snapped — rare regimes get their own cells.
4. **Kernel-herding weights (FISTA):** `Σ γᵢuᵢuᵢᵀ ≈ Ψ̂`; Voronoi mass prior,
   root-tempered `p^0.55`.
5. **Passport + OOD radar + Self-Healing:** top-32 eigenpairs of `Ψ̂` in float16 →
   Bures-Wasserstein distance on a 32-dim subspace (base threshold 0.085,
   auto-calibrated to batch size — see below), plus
   `loader.heal_domain_shift(...)` to surgically inject new Voronoi medoids and
   blend the spectral passport (old medoids are kept unless a `max_crystal_size` cap prunes low-weight cells).

## Industrial hardening (v3.7.1)

- **Adaptive BW threshold.** The radar threshold scales as
  `T(B) = 0.085 · max(1, sqrt(1024/B))` for batch size `B`: small sensor windows
  get a wider gate so sampling noise alone cannot trigger false alarms, while
  `B ≥ 1024` uses the classic `0.085`. `check_domain_shift(..., adaptive_threshold=True)`
  (default) returns the used `threshold` inside the report. Calibrated on the
  UCI Superconductivity heal benchmark (0/20 false alarms on in-distribution
  windows of 64 rows, 20/20 detections on true OOD windows). Since v3.12.0
  the base also scales with the passport rank as `0.085 · sqrt(rank/32)`
  (a `rank_passport >= 128` passport otherwise tripped false alarms on
  in-distribution windows), and `calibrate_threshold()` can replace the
  cross-dataset constant with a per-deployment quantile over healthy windows.
- **Bounded crystals.** `heal_domain_shift(..., max_crystal_size=220)` prunes
  only low-weight *background* medoids when the cap is exceeded; the top-26%
  phase-peak band and every newly injected medoid are always protected, and the
  pruned spectral mass is transferred to the nearest surviving medoids.
- **Streaming deltas.** `SpectralHologramDistiller(include_deltas=True)` augments
  the internal RFF sketch and Voronoi lattice with first differences
  `[X_t ‖ ΔX_t]` (storing only the previous batch's last row, O(d) memory), so
  the crystal becomes sensitive to rate-of-change transients in industrial
  telemetry. `crystal_X` and `get_training_data()` stay in the original
  `d`-dimensional feature space — downstream models are unaffected. *Measured
  trade-off* (see section 5 below): better transient representation, measurably
  worse background RMSE — use it only when rate-of-change transients are the
  target regime; the default remains `False`.

## Two-tailed trigger, adaptive y-stats, streaming radar (v3.8)

- **Two-tailed phase trigger.** `SpectralHologramDistiller(tail=...)` chooses
  which tail of the target the phase-peak score protects: `"high"` (default,
  numerically identical to v3.7), `"low"` (pressure/voltage sags and other
  low-y critical regimes), or `"both"` (either extreme). The crystal records
  the mode; the loader's `get_phase_peak_slice()` and the heal size-cap
  protection follow the same direction (`"both"` = most extreme relative to
  the median). In label-free mode the synthetic coordinate is high spectral
  leverage by construction, so `"high"` is always used there. *Measured
  honesty note:* on the synthetic streams the residual trigger and the
  κ·y phase geometry already isolate extreme-y regimes in either tail, so
  crystal-level capture of the matching regime is within noise across tail
  modes — the deterministic value of the option is the correct band direction
  for peak slicing and heal protection, plus the score-side boost for the
  chosen tail. All published tables use the default `tail="high"`.
- **Adaptive y-statistics (opt-in).** `adaptive_y_stats=True` keeps
  `y_mean`/`y_std` calibrated after the Voronoi lattice is built (exponential
  moving average, rate `min(0.1, B/n_seen)`), so the κ·y phase coordinate does
  not go stale on long streams with target drift. Default `False` = frozen
  first-batch statistics, exactly as in the published benchmarks.
- **Streaming radar continuity (opt-in).** `check_domain_shift(...,
  streaming=True)` and `heal_domain_shift(..., streaming=True)`: for delta
  crystals, the first difference of a window's opening row continues from the
  previous call's last raw row (the distiller's cross-batch semantics)
  instead of defaulting to zero. Non-delta crystals ignore the flag; the
  default remains fully stateless per-call windows.
- CLI: `--tail {high,low,both}` and `--adaptive-y-stats` on `distill`;
  `--streaming` on `radar`/`heal`. Package version aligned to 3.9.0.

## Constant sensitivity (ablation) and input robustness (v3.9)

Every architectural constant is now a constructor argument, so sensitivity can be
measured instead of asserted. Defaults are unchanged bit-for-bit (verified against
the previous commit: identical crystals, weights, slices and radar distances).

`bench_ablation.py` sweeps one constant at a time on the 4D reactor
(5 seeds, $M=300$, critical-regime KRR RMSE, lower is better):

| Constant | Values swept | Critical RMSE | Reading |
| :--- | :--- | :--- | :--- |
| `phase_kappa` | 0.0 / 1.6 / **3.2** / 6.4 | 4.199 / 3.236 / **3.346** / 3.404 | $\kappa=0$ (no phase manifold) is a large degradation, but a too-narrow band (0.10) is worse (4.433); the phase geometry is load-bearing, not decoration |
| `high_band_ratio` | 0.10 / **0.26** / 0.50 | 4.433 / **3.346** / 2.718 | most sensitive knob; reserving more volume for the phase band keeps helping on this stream |
| `tail_quantile` | 74 / **84** / 94 | 3.667 / **3.346** / 2.739 | a stricter trigger (94th pct) captures 43 rare points vs 18 |
| `voronoi_temper` | 0.0 / 0.30 / **0.55** / 1.0 | 3.200 / 3.266 / **3.346** / 3.518 | flat-ish optimum; raw cell volumes (1.0) are the worst |
| `herding_mix` | 1.0 / **0.75** / 0.50 / 0.0 | 3.326 / **3.346** / 3.372 / 3.453 | FISTA herding contributes little here; the tempered Voronoi term carries the weighting |
| `center_lr` | 0.05 / **0.25** / 0.50 | 3.201 / **3.346** / 3.414 | slow centroid updates slightly better on a stationary stream |
| `omega_low_scale` | 0.35 / **0.75** / 1.5 | 3.245 / **3.346** / 3.410 | modest effect |
| `omega_high_scale` | 0.75 / **2.10** / 4.2 | 3.202 / **3.346** / 3.217 | modest, non-monotone effect |
| `fista_l2_reg` | 0.05 / **0.20** / 0.60 | 3.347 / **3.346** / 3.347 | insensitive over a 12× range |

Two honest conclusions: (i) the published defaults are conservative, not
tuned-to-death — per-dataset tuning gives up to ~19% better critical RMSE
(`high_band_ratio=0.5`, `tail_quantile=94`), which is expected behaviour for a
coreset method and should be done on validation data; (ii) the single constant
whose removal destroys performance is `phase_kappa`, i.e. the coupled
feature-target manifold is the actual contribution.

### Input robustness

`partial_fit` and the OOD radar now validate their inputs instead of failing
inside NumPy/SciPy:

| Input | Behaviour |
| :--- | :--- |
| NaN/Inf in features or targets | `ValueError` naming the offending count and columns; `on_nonfinite="drop"` skips those rows instead |
| Empty batch/window | `partial_fit` is a no-op; the radar reports an error (nothing to score) |
| `len(y) != len(X)` | `ValueError` with both lengths |
| Feature width changed mid-stream | `ValueError` with both widths |
| `finalize_crystal()` with no data | `ValueError` explaining the required call order |
| Constant features **and** constant target | now builds a valid crystal (previously `ValueError: high <= 0` from k-means++ seeding) |
| Radar window with wrong width or NaN | `ValueError` stating the expected width |
| Single-row radar window | supported |

`verify_kt_port.py` additionally re-executes the official Microsoft `goodpoints`
X-path (pure-numpy, Cython stub) and confirms the KT port matches it
bit-for-bit on identical data and seeds (2D/4D, `m=1..3`, 3 seeds).

## Multi-seed benchmark results (20 seeds, paired Wilcoxon)

Single source of truth: `bench_seeds_results.json` (generated by `bench_seeds.py`)
and `bridge_neon_bo_results.json` (generated by `bridge_neon_bo.py`).
All medians are reported with interquartile ranges (IQR) across 20 seeds `0..19`.
For the other result files — which question each answers and which
are historical or working evidence — see [`ARTIFACTS.md`](ARTIFACTS.md).

### 1. UCI Superconductivity ($N = 21,263$, $D = 81$) → $M = 300$ medoids (1.76%)

Fixed 80/20 split (`n_train = 17,010`, `n_test_crit = 778` with $T_c \ge 77\text{ K}$).
SHD captures a median of **64.5** (IQR 3.5) high-$T_c$ superconductors out of 300 medoids.

| Coreset ($M=300$) | KRR RMSE All (med / IQR) | KRR RMSE $T_c \ge 77\text{K}$ (med / IQR) | KRR $R^2$ (med / IQR) | GBM RMSE All (med / IQR) | GBM RMSE $T_c \ge 77\text{K}$ (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle (KRR 9,000; GBM full train)** | 12.86 K | 17.59 K | 0.8562 | 10.27 K | 12.72 K |
| **Random (1.76%)** | 16.58 / 0.54 K | 22.43 / 2.49 K | 0.761 / 0.015 | 16.38 / 0.84 K | 22.57 / 2.49 K |
| **K-Means (1.76%)** | 17.34 / 0.55 K | 22.93 / 3.18 K | 0.739 / 0.017 | 17.19 / 0.97 K | 23.15 / 2.47 K |
| **Kernel Thinning (KT)** | 16.47 / 0.46 K | 23.26 / 2.23 K | 0.764 / 0.013 | 16.55 / 1.05 K | 22.25 / 2.37 K |
| **Supervised KT (SKT)** | 15.78 / 0.31 K | 21.20 / 1.64 K | 0.784 / 0.008 | **15.97 / 0.87 K** | 21.39 / 2.06 K |
| **SHD (Ours)** | **15.42 / 0.25 K** | **18.81 / 0.93 K** | **0.793 / 0.007** | 16.06 / 0.65 K | **18.71 / 3.21 K** |
| *Wilcoxon SHD vs Random* | 20/20 wins, $p=1.9\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 13/20, $p=0.28$ (n.s.) | 17/20 wins, $p=1.3\times 10^{-4}$ |
| *Wilcoxon SHD vs K-Means* | 20/20 wins, $p=1.9\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 18/20 wins, $p=9.5\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ |
| *Wilcoxon SHD vs KT* | 20/20 wins, $p=1.9\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 15/20 wins, $p=3.3\times 10^{-2}$ | 17/20 wins, $p=1.0\times 10^{-3}$ |
| *Wilcoxon SHD vs SKT* | 17/20 wins, $p=6.4\times 10^{-3}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=6.4\times 10^{-3}$ | 6/20, $p=0.070$ (n.s.) | 17/20 wins, $p=3.2\times 10^{-4}$ |

**KT/SKT duel note.** Kernel Thinning and Supervised Kernel Thinning are run via a
faithful pure-numpy port of the official Microsoft `goodpoints` X-path
(`kt_baseline.py`; the Cython extension does not link on Windows). The port is
**verified bit-exact against the author's own code**: `verify_kt_port.py` loads
`goodpoints/kt.py` from the PyPI sdist with a stub `ktc` module and reproduces the
identical coreset indices on 2D and 4D data, `m = 1..3`, three seeds. Settings follow
the official numpy experiment configuration: Gaussian kernel with median-heuristic
bandwidth, split kernel at `var/2`, swap kernel at `var`, the author's default
`unique=False`, a seeded i.i.d. pre-sample of $300\cdot 2^4 = 4{,}800$ points, and
exactly $M=300$ output. Honest reading: on the critical regime
($T_c \ge 77\,\mathrm{K}$) SHD beats KT **20/20 seeds** ($23.26 \to 18.81\,\mathrm{K}$)
and SKT **20/20** ($21.20 \to 18.81\,\mathrm{K}$), both at $p=1.9\times10^{-6}$, and
captures $64.5$ critical superconductors per crystal vs $59.0$ (KT) and $55.0$
(SKT). SKT is the strongest baseline on global metrics: SHD wins $17/20$ on
KRR-all ($p=6.4\times10^{-3}$) but is numerically behind it on GBM-all
($15.97$ vs $16.06$ K, SHD wins only $6/20$, n.s. $p=0.070$) — KT/SKT optimize
global MMD, SHD trades a sliver of background quality for a decisive rare-regime
edge. On the 4D reactor SHD wins the critical
regime $18/20$ against KT ($p=9.5\times10^{-6}$) and $20/20$ against SKT
($p=1.9\times10^{-6}$), and captures $19.5$ peak points per crystal vs $10.0$ (KT),
$8.5$ (SKT), $9.0$ (Random) and $11.5$ ($K$-Means).

### 2. Nonlinear 4D reactor with phase transitions ($N_{\text{train}} = 16,000 \to M = 300$)

| Coreset ($M=300$) | KRR RMSE All (med / IQR) | KRR RMSE Peak (med / IQR) | KRR $R^2$ (med / IQR) | GBM RMSE All (med / IQR) | GBM RMSE Peak (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle (KRR 9,000; GBM 16,000)** | 0.80 | 2.78 | 0.809 | 0.38 | 1.11 |
| **Random** | 1.134 / 0.104 | 4.424 / 0.814 | 0.621 / 0.070 | 0.711 / 0.081 | 2.409 / 0.549 |
| **K-Means** | 1.069 / 0.073 | 4.390 / 0.722 | 0.663 / 0.046 | **0.596 / 0.079** | 2.033 / 0.572 |
| **Kernel Thinning (KT)** | 1.129 / 0.030 | 4.516 / 0.833 | 0.625 / 0.020 | 0.719 / 0.113 | 2.382 / 0.772 |
| **Supervised KT (SKT)** | 1.056 / 0.029 | 4.060 / 0.347 | 0.672 / 0.018 | 0.695 / 0.128 | 2.432 / 0.811 |
| **SHD (Ours)** | **0.976 / 0.021** | **3.436 / 0.225** | **0.720 / 0.012** | **0.624 / 0.092** | **1.732 / 0.579** |
| *Wilcoxon SHD vs Random* | 20/20, $p=1.9\times 10^{-6}$ | 20/20, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 16/20, $p=7.1\times 10^{-4}$ | 18/20, $p=1.7\times 10^{-4}$ |
| *Wilcoxon SHD vs K-Means* | 20/20, $p=1.9\times 10^{-6}$ | 20/20, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 8/20, $p=0.25$ (n.s.) | 13/20, $p=0.064$ |
| *Wilcoxon SHD vs KT* | 20/20, $p=1.9\times 10^{-6}$ | 18/20, $p=9.5\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 16/20, $p=7.3\times 10^{-3}$ | 14/20, $p=2.7\times 10^{-3}$ |
| *Wilcoxon SHD vs SKT* | 20/20, $p=1.9\times 10^{-6}$ | 20/20, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 14/20, $p=1.7\times 10^{-3}$ | 16/20, $p=5.9\times 10^{-4}$ |

### 3. Bridge with NEON Lab: Discontinuity-Aware Bayesian Optimization (`bridge_neon_bo.py`)

A cubic $\mathcal{O}(N^3)$ Matérn-5/2 Gaussian Process surrogate cannot fit raw
streaming telemetry ($N = 10,000$). When fitted on a $M = 120$ coreset and paired
with Expected Improvement (EI) over an unseen candidate pool ($N_{\text{pool}} = 3,000$,
phase-transition pocket = 12 points / 0.40% volume):

| Coreset feeding GP+EI ($M=120$) | GP RMSE inside Phase Pocket (med / IQR) | Top-20 EI Hits in Pocket out of 12 (med / IQR) | 10-step BO Simple Regret (med / IQR) | GP RMSE All Background (med / IQR) |
| :--- | :---: | :---: | :---: | :---: |
| **Random (1.2%)** | 6.028 / 0.591 | 0.0 / 2.25 | 1.783 / 5.304 | 0.437 / 0.127 |
| **K-Means (1.2%)** | 6.073 / 0.238 | 0.0 / 0.5 | 5.578 / 5.200 | **0.425 / 0.087** |
| **SHD v3.7 (1.2%)** | **4.164 / 0.693** | **6.0 / 2.0** | **0.099 / 0.266** | 0.546 / 0.031 |
| *Wilcoxon SHD vs Random* | 20/20 wins, $p=1.9\times 10^{-6}$ | 19/20 wins, $p=1.0\times 10^{-4}$ | 16/20 wins, $p=5.9\times 10^{-4}$ | 6/20, $p=0.0073$ (trade-off) |
| *Wilcoxon SHD vs K-Means* | 19/20 wins, $p=9.5\times 10^{-6}$ | 19/20 wins, $p=1.1\times 10^{-4}$ | 18/20 wins, $p=3.4\times 10^{-4}$ | 3/20, $p=1.7\times 10^{-4}$ (trade-off) |

Headline: standard GP+EI on Random or K-Means coresets gets a median of **0 out
of 12** pocket hits in its top-20 EI acquisitions, whereas the same GP+EI on the
SHD crystal hits **6 out of 12** immediately and drives 10-step BO simple regret
down to **0.099** (reaching exact 0.000 regret in 8/20 seeds).

#### 3b. End-to-end Phase Boundary Rediscovery (`SHD v3.7 → NEON Lab CPD-rank`)

When a composition-temperature stream ($N = 8,000$, quadrant phase transition
$A > 0.625 \land T > 720\text{ K}$ covering $\approx 2.8\%$ of the stream) is
compressed to $M = 120$ points and fed into NEON Lab's two-layer rank-residual
permutation detector (`discover_phase_boundary_cpd(use_rank=True, alpha=0.01)`):

| Coreset feeding CPD-rank ($M=120$) | Planted Phase Points Captured (med / IQR) | CPD-rank F1 (med / IQR) | Verified Detections ($p < 0.01 \land \text{F1} \ge 0.5$) | Boundary Error $\Delta A$ (med / IQR) | Boundary Error $\Delta T$ (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random (1.5%)** | 3.0 / 2.0 | 0.038 / 0.138 | 1 / 20 (5%) | 0.200 / 0.138 | 105.0 / 153.1 K |
| **K-Means (1.5%)** | 5.0 / 2.0 | 0.171 / 0.439 | 4 / 20 (20%) | 0.163 / 0.194 | 105.0 / 105.0 K |
| **SHD v3.7 (1.5%)** | **10.0 / 3.0** | **1.000 / 0.000** | **18 / 20 (90%)** | **0.050 / 0.025** | **43.8 / 52.5 K** |
| *Wilcoxon SHD vs Random* | 20/20 wins, $p=8.3\times 10^{-5}$ | 17/20 wins, $p=2.5\times 10^{-4}$ | **18/20 vs 1/20** | 17/20 wins, $p=3.3\times 10^{-4}$ | 16/20 wins, $p=1.5\times 10^{-3}$ |
| *Wilcoxon SHD vs K-Means* | 20/20 wins, $p=8.3\times 10^{-5}$ | 15/20 wins, $p=3.6\times 10^{-3}$ | **18/20 vs 4/20** | 15/20 wins, $p=8.0\times 10^{-3}$ | 14/20 wins, $p=2.8\times 10^{-3}$ |

### 4. Self-healing under a real physical domain shift (`bench_heal_superconduct.py`)

UCI Superconductivity split by physics, not at random: **Domain A** = $T_c < 40$ K
(train 11,229 / test 2,808), **Domain B** = $T_c \ge 40$ K cuprates
(train 5,780 / test 1,446). A crystal of $M = 200$ is built on Domain A, the radar
watches Domain B arrive, and heal injects 45 cuprate medoids. Single source of
truth: `bench_heal_superconduct_results.json` (20 seeds, paired Wilcoxon).

| Check | Result (median over 20 seeds) | Verdict |
| :--- | :--- | :--- |
| False alarms on in-distribution Domain A | BW = 0.035 vs threshold 0.085 | **0/20** |
| OOD detections on Domain B (cuprates) | BW = 0.636 | **20/20** |
| Post-heal BW on Domain B | 0.465 | default blend leaves both A (0.203) and B (0.458) above 0.085; `passport_recalibrate=True` gives B 0.038 / A 0.497; `DomainBank` accepts both |
| KRR RMSE on new Domain B | **26.88 K** healed vs **57.17 K** pre-heal | 20/20, $p=1.9\times10^{-6}$ |
| GBM RMSE on new Domain B | **19.38 K** vs **54.57 K** | 20/20, $p=1.9\times10^{-6}$ |
| Old Domain A retention vs naive overwrite | KRR **13.36 K** vs **20.62 K** | 20/20, $p=1.9\times10^{-6}$ |
| Combined A+B vs pre-heal / naive overwrite | **19.10 K** vs 33.86 / 21.58 | 20/20 both |
| Bounded heal (`max_crystal_size=220`) | combined **19.15 K** vs 33.86 | 20/20 — size control costs ≈ nothing |

Passport modes (5 seeds, `bench_passport_results.json`): pre-heal A 0.035 / B 0.629;
blend A 0.203 / B 0.458 (both OOD); recalibrate A 0.497 / B 0.038 (new in, old
flagged); bank routes 5/5 and accepts both 5/5. Use recalibration only if the old
domain will not return; use a bank when both regimes remain live.

Headline: on a physically meaningful shift (low-$T_c$ iron-based → cuprate
superconductors) the naive response is either blindness (RMSE 57 K) or catastrophic
forgetting (old domain degrades 13.4 → 20.6 K). Surgical heal captures the new
domain *and* keeps the old one better than overwrite (13.36 vs 20.62 K), but not
for free: pre-heal Domain A is 7.32 K, so the heal trades ≈6 K of old-domain
accuracy for ≈30 K on the new domain. The crystal can be hard-capped for
production memory budgets.

### 4b. Sequential drift: banking vs repeated healing (`bench_bank_scaling.py`)

`heal_domain_shift` on a *single* crystal is a three-way trade-off and it
**accumulates forgetting** over a long sequence of regimes. `DomainBank` keeps
one crystal per regime and routes by spectral passport, so nothing is overwritten.
10 successive regimes on the same 4D sensor (each shifts the inputs and changes
the input→output law), 3 seeds, M=300, drift strength varied by a `shift_scale`
factor on the per-regime shift:

| Drift | Routing accuracy | Bank R0 | Heal R0 (after 9 shifts) | Newest regime: bank | newest: single heal | newest: frozen on R0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1.00 (separated) | **1.00** | **0.265** | 0.549 | **1.307** | 3.988 | 9.646 |
| 0.50 | **1.00** | **0.265** | 0.402 | **0.863** | 1.550 | 2.744 |
| 0.25 (near-identical) | 0.60 | **0.265** | 0.272 | **0.651** | 0.990 | 1.277 |

`R0` is RMSE on regime 0; a crystal built on regime 0 scores 0.265, so **bank
retention is exactly the fresh-crystal value at every drift strength** — it is
immune by construction, not by clever optimisation. Sequential healing degrades
R0 from 0.265 to 0.549 at strong drift (**×2.07**), and leaves the *newest*
regime 3× worse than the bank (3.988 vs 1.307). Freezing on regime 0 preserves
R0 perfectly but collapses on every later regime (9.646).

Cost: **14 KB per regime**, i.e. 140 KB for ten regimes — nothing for an edge
controller, and the in-domain false-shift rate is 0/10 with every unseen regime
detected (3/3 seeds).

**Honest limits of this result.** Routing is exact whenever regimes are
meaningfully separated (shift ≥ 0.5) and degrades to 0.60 when they are nearly
identical. In that band a misroute costs little — the regimes are almost the
same, and the bank still beats both alternatives on the newest regime — but the
bank's advantage *shrinks with the drift*, because if regimes barely differ a
single crystal already suffices. The dangerous middle band is drift large enough
to hurt one crystal but small enough to make routing ambiguous; it is not mapped
here. Also note this is a synthetic 4D sequence, not a recorded plant.

Source of truth: `bench_bank_scaling_results.json`.

### 5. Streaming deltas under vibration bursts (`bench_deltas.py`)
Does `include_deltas=True` earn its keep? Synthetic 4D AR(1) telemetry with rare
3-step vibration bursts (innovation variance ×6, y jump +3.5): marginally the
burst rows sit in the *same* X distribution as the background, so their only
geometric signature is the rate of change ΔX. Temporal split (6,400 train / 2,600
test, 1.45% burst rows), paired crystals per seed (identical except
`include_deltas`), 20 seeds, single source of truth: `bench_deltas_results.json`.

| Metric (median over 20 seeds) | deltas off | deltas on | Random coreset | Paired Wilcoxon (on vs off) |
| :--- | :---: | :---: | :---: | :---: |
| Burst rows captured in crystal (of 93) | 15.0 | **17.0** | 3.0 | 12/20, $p=0.12$ (n.s.) |
| GBM RMSE on test burst rows | 3.77 | **3.50** | — | **14/20 wins, $p=0.036$** |
| KRR RMSE on test burst rows | 4.15 | 4.18 | — | 10/20, $p=0.62$ (n.s.) |
| KRR RMSE all (background cost) | **0.554** | 0.610 | — | 3/20, $p=4.8\times10^{-5}$ (deltas worse) |
| GBM RMSE all (background cost) | **0.710** | 0.778 | — | 7/20, $p=3.7\times10^{-3}$ (deltas worse) |

Honest verdict: `include_deltas` is a *targeted* mode, not a free win. It
significantly improves GBM accuracy on the transient regime and lifts burst
capture (17 vs 15 vs 3 for Random), but pays a measurable background cost
(≈10% higher KRR RMSE on in-distribution rows). Both coreset variants capture
bursts far above Random, and the residual/leverage trigger already finds burst
rows through their elevated |y| — the delta channel refines *which* of them get
their own Voronoi cells. Recommendation: enable it only for transient-dominated
telemetry; the default remains `False`.

### 6. Choosing the operating point: the global ↔ critical frontier

`high_band_ratio` and `tail_quantile` are **deployment presets, not universal
constants**. Sweeping them at equal budget (M=600, 20 seeds, paired Wilcoxon
against KMeans at the same M) shows a monotone frontier: every point wins the
critical regime, and the price you pay is global fit.

**Reactor 4D** (critical = 3% of the stream), vs equal-budget KMeans:

| `high_band_ratio` | KRR RMSE peak | GBM RMSE all | Rare captured |
| :---: | :---: | :---: | :---: |
| 0.26 (default) | −14.3%, 20/20 wins, $p=1.9\times10^{-6}$ | +4.2% (n.s., $p=0.19$) | 36.5 |
| 0.50 | −26.9%, 20/20 wins | +5.5% (n.s.) | 68.5 |
| 0.65 | **−33.2%, 20/20 wins** | +2.5% (n.s.) | **90.5** |

**UCI Superconductivity** (critical = 18.6%), vs equal-budget KMeans:

| `high_band_ratio` | KRR RMSE $T_c\ge77$K | GBM RMSE $T_c\ge77$K | GBM RMSE all | Rare captured |
| :---: | :---: | :---: | :---: | :---: |
| 0.26 (default) | −9.3%, 19/20 | −19.4%, 20/20 | −5.2%, 19/20 | 131.5 |
| 0.50 | −16.5%, 20/20 | −34.3%, 20/20 | −0.8% (n.s.) | 238.0 |
| 0.65 | **−20.9%, 20/20** | **−40.0%, 20/20** | **+4.5%, 3/20 (cost)** | **304.5** |

The working sweep, **not backed by a saved per-seed artifact**, reports that
at `high_band_ratio=0.65` a 600-point crystal (3.5% of the data) beats the
oracle reference on both critical metrics of the superconductivity set:
KRR 15.51 K vs 17.59 K oracle (−11.8%) and GBM 12.08 K vs 12.72 K (−5.0%).

> **Provenance note (honest status of §6).** The M=600 / `high_band_ratio`
> sweep runs behind this section were executed as one-off working scripts and
> their per-seed outputs were **not committed as JSON artifacts** — unlike
> every other section of this README, §6 currently has no
> `bench_*_results.json` behind it (§1's oracle denominators 17.59/12.72 are
> artifact-backed and match; the sweep deltas are not). Treat these numbers
> as working results pending an artifact commit: before citing the
> oracle-beating row externally, re-run the sweep with `--save-json` (or
> equivalent) and commit the artifact next to this README.

**Practical presets:**

| Goal | `high_band_ratio` | `tail_quantile` |
| :--- | :---: | :---: |
| global accuracy first | 0.26 | 84 (validate separately from a top-3% target) |
| balanced | 0.50 | 84 |
| critical / rare-event first | 0.65 | 84 |
| sparse stream (top-3% regime) | 0.26 | **97** |

`tail_quantile` should match the *percentile of the regime you care about*: on
the Beijing PM2.5 time-shifted stream (critical = top 3%) the default `84`
only reaches a tie, whereas `tail_quantile=97` gives a significant critical
win (−11.9% KRR, 18/20, $p=2.1\times10^{-4}$) at the cost of global fit. See
`bench_pm25_rehab.py` / `bench_pm25_rehab_results.json`.

**The shipped defaults are not the per-dataset optimum.** The bootstrap
ablation in `bench_ablation_ci_summary.json` puts `high_band_ratio=0.50` at
2.737 K and `tail_quantile=94` at 2.735 K, versus 3.440 K for the shipped
`0.26` / `84` on the 4D reactor (10 seeds; difference CIs exclude 0). The
defaults are kept at `0.26` / `84` because every table in §1–2 was generated
with them; changing them silently would invalidate the published artifacts.
Choose from the preset table above rather than assuming the default is optimal.

**Enlarging the crystal instead of re-tuning the band.** The tree-model (GBM)
background deficit is a *budget* artefact, not a property of the method. On the
4D reactor the median deficit against equal-budget KMeans shrinks monotonically
with M — +4.7% at M=300, +4.2% at M=600, −0.2% at M=900, −4.7% at M=1200 — and
is **never statistically significant in either direction** at any of these
budgets (8/20, $p=0.25$; 7/20, $p=0.19$; 9/20, $p=0.62$; 13/20, $p=0.37$).
Note that M=600 already costs 3.5–3.8% of the stream, above the 0.2–1.8% band
quoted at the top of this README; treat M=600 as a *quality mode*.

### 7. Deployment envelope on one CPU core (`bench_edge_envelope.py`)

The compression claim is only useful if the whole thing runs where there is no
GPU and no network. Measured on one host with BLAS/OpenMP pinned to a single
thread, medians over 3 passes (latency is noisy; spread shown):

| Case | Artifact | Crystal load | Radar, B=64 | B=256 | B=1024 | Crystal RAM | Fit throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4D sensor (16k rows, M=300) | **28.7 KB** | 3.5 ms | **1.23 ms** | 3.65 ms | 18.6 ms | +0.28 MB | 27.1k rows/s |
| 81D synthetic Gaussian/sine (17k rows, M=300) | **184.7 KB** | 5.9 ms | **1.26 ms** | 4.93 ms | 17.9 ms | +1.25 MB | 10.0k rows/s |

Latency spread across passes was 1.07–2.07 ms (B=64) and 15.5–20.3 ms (B=1024).

**Read this correctly.** The *artifact* is 29–185 KB, but the *process* holding it
needs 58–69 MB because Python + NumPy is the baseline; the crystal itself adds
only **0.28–1.25 MB** on top. So:

- on an industrial PC, a Raspberry-Pi-class box, or an IPC at a wellhead, the
  memory cost is negligible — this fits easily;
- on a bare microcontroller it does **not** fit without a rewrite (the 58 MB is
  the interpreter, not the data). Do not quote "29 KB total footprint".

This is a single-host, single-run-family measurement: treat absolute latencies as
order-of-magnitude, not as a spec. Re-measure on your target hardware before
promising a number to a customer.

**Corrected-code rerun (2026-10-20; `bench_edge_envelope_v2_results.json`).**
Same synthetic cases on the repaired implementation, medians over 3 passes: 4D
sensor 28.9 KB artifact, radar 1.22 / 3.75 / 14.55 ms (B=64/256/1024), +0.73 MB
crystal RSS delta, 30.9k rows/s; 81D Gaussian/sine 184.9 KB, radar 1.24 / 4.73
/ 16.10 ms, +0.82 MB, 10.9k rows/s. The numbers track the historical table;
absolute latencies remain single-host order-of-magnitude estimates.

### 8. What a buyer actually pays: failure cost, not RMSE (`bench_scania_cost.py`)

Every table above scores regression error. A maintenance department instead pays
for missed failures and wasted inspections. The UCI *APS Failure at Scania
Trucks* experiment below uses a **custom 10:1 cost scenario**, not the official
challenge metric. [UCI specifies 500·FN + 10·FP](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)
(a normalized 50:1 ratio). The historical results below must not be compared
with official challenge costs; the retuned rerun under the official metric is
reported at the end of this section.
The historical experiment used:

$$\text{cost} = 10\cdot\text{FN} + 1\cdot\text{FP}$$

48,000 fit / 12,000 validation / 16,000 test rows, 170 features, 1.67%
failures in fit. Validation is held out from the original 60,000 training rows;
the decision threshold is tuned there, never on test. The same protocol is
applied to every coreset.

| Coreset (300 rows) | Model | Failures missed | False alarms | Recall | Precision | Cost = 10·FN + 1·FP |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Full train (oracle)** | HGB | 30 | 231 | 0.920 | 0.599 | **531** |
| **Full train (oracle)** | logit | 59 | 180 | 0.843 | 0.637 | **770** |
| Random | HGB | 66.0 | 555.5 | 0.824 | 0.354 | 1436.5 |
| K-Means | HGB | 77.5 | 408.0 | 0.793 | 0.390 | 1271.5 |
| **SHD** | **HGB** | **55.5** | 463.5 | **0.852** | 0.408 | **993.0** |
| Random | logit | 87.5 | 418.5 | 0.767 | 0.420 | 1242.5 |
| K-Means | logit | 56.5 | 378.5 | 0.849 | 0.429 | **964.5** |
| SHD | logit | 120.5 | 217.5 | 0.679 | 0.532 | 1419.0 |

Cells are medians over 10 seeds, printed with one decimal: a median of an even
number of integer counts is fractional (e.g. 77.5 missed failures), and a
0-decimal format would silently banker-round it to "78" or "964".

Positive rows inside the 300-row coreset (median over 10 seeds; a median of an
even number of integer counts can be fractional): Random 7.0, K-Means **7.5**,
SHD 25.0. The never-alert policy costs 3750.

Paired Wilcoxon over 10 seeds (SHD cheaper = win):

| Comparison | Trees (HGB) | Linear (logit) |
| :--- | :---: | :---: |
| SHD vs Random | **8/10 wins, $p=0.020$** | 3/10, $p=0.63$ (n.s.) |
| SHD vs K-Means | **8/10 wins, $p=0.020$** | 2/10, $p=0.43$ (n.s.) |

**The honest verdict: the advantage is model-dependent, and it is on the model
that industry actually deploys.** Against a gradient-boosted tree ensemble — what
runs in most predictive-maintenance stacks — a 300-row (0.6% of train) SHD
crystal buys a **21.9% cheaper** maintenance decision than K-Means and **30.9%**
cheaper than a random subsample, both significant at $p=0.020$, while missing
fewer actual failures than either (medians 55.5 vs 77.5 vs 66.0) and reaching 0.852 recall
against a 0.920 full-data oracle.

Against a balanced linear model the median ranking **reverses**: SHD costs 1419.0 vs
K-Means 964.5 (the paired difference is not statistically significant). The mechanism is visible in the counters — SHD deliberately holds
25 failure rows in 300 (vs 7.5 for K-Means), which is a biased sample for
estimating a linear boundary but a useful density for axis-aligned trees. Note
also that the test failure rate (2.34%) exceeds the train rate (1.67%), so
thresholds do not transfer perfectly; SHD+logit has medians of 120.5 missed and
217.5 false alarms, while K-Means+logit has 56.5 missed and 378.5 false alarms.

Do **not** sell this as a universal improvement or guaranteed monetary saving.
This historical HGB experiment favours SHD under the chosen 10:1 cost scenario;
linear-model medians favour K-Means. Deployment needs a pre-agreed cost model
and fresh validation on the target equipment.

Source of truth: `bench_scania_cost_results.json`; the table above is generated
from it by `make_scania_table.py`.

**Official UCI cost rerun (500·FN + 10·FP, 2026-10-20).** The official metric
rewards recall at 50:1, and SHD's rare-regime capture is exactly what it pays
for. Thresholds were retuned on the same held-out validation slice for the
official costs (fresh run of the repaired code, 10 seeds; artifact:
`bench_scania_cost_official_results.json`):

| Coreset (300 rows) | Model | Failures missed | False alarms | Recall | Cost = 500·FN + 10·FP |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Full train (oracle)** | HGB | 13 | 396 | 0.965 | **10460** |
| **Full train (oracle)** | logit | 24 | 480 | 0.936 | **16800** |
| Random | HGB | 34.0 | 1074.5 | 0.909 | 32355.0 |
| K-Means | HGB | 41.5 | 828.5 | 0.889 | 31340.0 |
| **SHD** | **HGB** | **17.5** | 969.0 | **0.953** | **18645.0** |
| Random | logit | 45.0 | 988.5 | 0.880 | 32545.0 |
| K-Means | logit | 28.0 | 833.0 | 0.925 | 23080.0 |
| SHD | logit | 95.0 | 583.0 | 0.747 | 53330.0 |

Medians over 10 seeds, one decimal for the same fractional-median reason as
above. The never-alert policy costs 187500. Paired Wilcoxon (SHD cheaper = win):
SHD vs Random (HGB) **9/10 wins, $p=0.0039**; SHD vs K-Means (HGB) **8/10
wins, $p=0.0098**; both logit arms n.s. (3/10 and 2/10). SHD+HGB lands at
1.8× the full-train oracle while Random/K-Means sit near 3×, and misses the
fewest failures of any 300-row coreset (17.5 vs 34.0/41.5). The model-family
caveat cuts both ways: SHD+logit is the *worst* arm under the official metric
(95.0 missed), so the crystal helps the learner that can exploit its density,
not every learner.

### 9. Early warning on a real metro compressor (`shd_watch.py`, `bench_metro_watch.py`)

**Historical protocol v1 — not evidence for corrected protocol v2.** The preserved
`bench_metro_watch_results.json` reports the following on MetroPT-3 (252,720
minutes, four labelled air leaks). Its calibration had three defects: it could
merge sustained alerts into one episode, used different EWMA update semantics
from watch, and allowed row windows to cross telemetry gaps. A short calibration
slice could also round a 1/month budget up to one episode. These numbers are
retained for traceability, not as proof of budget compliance.

| Detector (historical v1) | Watch unmatched FA/calendar month | Episodes | Incidents matched |
| :--- | ---: | ---: | :--- |
| SHD passport, detrended | 0.0 | 0 | 0/4 |
| Feature z-score, detrended | 3.2 | 27 | 4/4 |
| IsolationForest, detrended | 0.0 | 1 | 1/4 |
| SHD passport, raw | 0.0 | 0 | 0/4 |

The passport had a negative result in this historical experiment. Neither
representation as the sole bottleneck nor equal-budget superiority follows from
a defective calibration protocol. An earlier cold-start EWMA warning also vanished
when the baseline was warmed: this warm-up artefact must not be cited as lead time.
The historical `watch_scan_sec` timer did not surround scoring and is invalid.

**Current protocol v2.** Learn Feb 1–Mar 15, calibrate Mar 15–Apr 10, watch
Apr 10–Aug 1. All arms share complete contiguous 60-minute windows, 30-minute
steps and a 6-hour episode cooldown. Missing minutes break windows. EWMA uses
actual elapsed timestamps and freezes on alerts during both candidate selection
and deployment. The observed episode allowance is floored, not rounded up;
missing telemetry is not healthy calibration exposure. A separate default
`max_alert_fraction=0.05` prevents sustained alarms from gaming episode counts.
These are empirical calibration constraints, **not future false-alarm guarantees**.
Threshold files carry protocol version, feature order, window/step, baseline and
baseline timestamp. Old thresholds are rejected: **recalibrate** before watch.

Run `python bench_metro_watch.py` for a new experiment. It writes
`bench_metro_watch_v2_results.json` and `metro_demo_v2_crystal.hologram.npz`, never
the v1 files by default. A result counts as current only after that run completes;
old JSON numbers do not automatically transfer to the repaired implementation.
Ground truth remains `metro_incidents.json`. Incident matching uses a 21-day
pre-incident horizon; unmatched FA rates use the same observed-exposure
denominator as the calibration budget (calendar-month rates are recorded
alongside), so the reported rate and the budget are directly comparable.

**Protocol v2 result (fresh run of the repaired code, 2026-10-20; artifacts:
`bench_metro_watch_v2_results.json`, `metro_demo_v2_crystal.hologram.npz`).**
All four arms are silent on calibration (0 episodes, alert fraction 0.000
over 994 windows). On the 112-day watch (4,192 windows, 2.87 observed-exposure
months):

| Detector (protocol v2) | Leaks detected | Lead times (h) | Watch episodes | Unmatched FA/month (observed) |
| :--- | :---: | :---: | ---: | ---: |
| SHD passport, detrended | **0/4** | — | 0 | 0.00 |
| Feature-z, detrended | **4/4** | 131.3 / 392.9 / 415.4 / 163.7 | 15 | 1.74 |
| IsolationForest, detrended | 2/4 | 392.4 / 394.6 (leaks 2–3) | 2 | 0.00 |
| SHD passport, raw | 0/4 | — | 0 | 0.00 |

This is a **negative result for the SHD spectral passport on this task**: at
the budget-calibrated threshold the radar produces no alerts and misses all
four labelled leaks, while the plain feature-z baseline — same windows, same
EWMA detrending, same budget — catches all four with 5.5–17.3 days of lead at
1.74 unmatched false alarms per observed month. Crystal: 54.8 KB, learn 4.7 s,
watch scan 15.3 s.

The radar is not completely blind to the faults: the single largest detrended
deviation of the entire 112-day watch slice occurs inside the 21-day window of
Air leak #1 (`dev = 0.5292`, reaching 85.3% of the calibrated `tau = 0.6207`;
Air leaks #2 and #3 peak at 64.4% of tau, #4 at 34.6%). But the budget
calibration forces tau above those peaks to keep the false-alarm rate at or
below 1/month on the healthy slice, so none of them crosses into an alarm.
The v1 hypothesis that representation quality was the bottleneck does not
survive the corrected protocol; any SHD claim about compressor early warning
needs a different detector design, not a defence of this one.

## Known limitations

- **Negative result: the background-only spectral passport is not a fix.**
  `passport_reference="background"` was added because a whole-stream passport
  absorbs a shifted regime that occupies part of the training data. Measured
  under a controlled protocol (`bench_passport_reference.py`; evaluation
  windows held out from the training stream; 10 seeds synthetic, 5 seeds on
  the Beijing PM2.5 cache):
  - *Synthetic*: the contamination penalty appears only at **40%**
    contamination (BW 0.205 against a 0.157 threshold), and there the passport
    is **degenerate** — windows taken from the training stream alarm as well
    (in-sample FA 1.00). At 10% and 20% contamination nothing alarms
    (FA 0.00, detection 1.00). The background reference moves the distances by
    less than 0.01 and changes no outcome.
  - *Beijing PM2.5*: the radar alarms on **in-sample** windows as well
    (FA 1.00 for windows that are part of the training stream), with the
    repository's 14-feature set and with sensors only. That is not reference
    contamination: at these settings the detector is degenerate on this
    telemetry independently of which population the reference describes.

  The background reference therefore stays **opt-in**
  (`--passport-reference background`), is never the default, and must not be
  presented as a remedy for the early-warning failure.

- **Negative result: Beijing PM2.5 (real sensor telemetry, time-ordered split).**
  This is the second real dataset and the method **does not** recommend itself for
  global regression on it. With the default configuration SHD at M=300 loses
  significantly to KMeans on global accuracy (KRR-all +6.9%, 4/20 wins,
  $p=4.8\times10^{-4}$; GBM-all +10.0%, 1/20 wins, $p=5.7\times10^{-6}$) and only
  ties on the critical regime (KRR $T=324\,\mu g/m^3$: 11/20, n.s.). Narrowing the
  band does **not** fix global fit (+10.7% at `high_band_ratio=0.10`); but aligning
  the tail with the 3% regime (`tail_quantile=97`) does deliver a significant
  critical win (−11.9%, 18/20, $p=2.1\times10^{-4}$, 26.5 rare captured vs 10) at
  a further global cost (+13.7%). Verdict: use SHD here for rare-episode
  monitoring with `tail_quantile=97`, **not** as a global regressor. Source of
  truth: `bench_pm25.py` / `bench_pm25_results.json` (default configuration) and
  `bench_pm25_rehab.py` / `bench_pm25_rehab_results.json` (knob sweep).

- **Forgetting is real for a single crystal, bounded for a bank.** Repeated
  `heal_domain_shift` on one crystal degrades the original regime monotonically
  (RMSE 0.265 → 0.549 over nine shifts at strong drift, ×2.07; see §4b). Do not
  describe this path as "forgetting-free". `DomainBank` retains
  perfectly because nothing is overwritten, at 14 KB per regime — but its
  passport routing is only exact when regimes are meaningfully separated
  (accuracy 1.00 at drift ≥ 0.5, 0.60 when regimes are nearly identical).
  Source: `bench_bank_scaling_results.json`.

- **Single-passport healing is a three-way trade-off:** the default blend can
  leave *both* domains uncalibrated on a strong shift; `passport_recalibrate=True`
  accepts the new domain but flags the old; `DomainBank` accepts both at the cost
  of one crystal per domain. See `bench_passport_results.json`.


- **Deliberate background trade-off:** reserving 26% of medoid cells for the
  phase-peak band slightly increases background GP RMSE (`0.546` vs `0.425` on
  the BO bridge) and ties with K-Means on tree-based global RMSE (`p = 0.25` on
  4D reactor GBM). Use SHD when rare / critical regimes matter.
- **Legacy artifact:** `artifacts/superconduct_v37.hologram.npz` predates the
  stored projector; its OOD radar requires `check_domain_shift(X, distiller_ref=...)`.
  Crystals saved by `save_hologram(include_projector=True)` are self-contained.
- **Legacy delta artifacts:** artifacts saved before v3.9.1 do not store
  `include_deltas`; the loader treats them as non-delta. Load a legacy delta
  crystal with `HologramCrystalLoader(path, include_deltas=True)`, then re-save
  it with `save()` so the flag is persisted.
- **Weighting is a trade-off, not a free win:** on 20 seeds with identical crystals,
  gamma weights improve global KRR (SC 15.57 vs 15.99; reactor 0.975 vs 0.992;
  both 20/20) but on the reactor they cost critical KRR (3.426 vs 3.335; uniform
  wins 20/20). See `bench_weights_*_results.json`.
- **OOD threshold:** the base `BW_ALERT_THRESHOLD = 0.085` (auto-widened for
  small batches as `0.085·sqrt(1024/B)`, rank-scaled as `0.085·sqrt(rank/32)`
  since v3.12.0) is calibrated on the superconductivity heal benchmark, not a
  distribution-free constant. For a new domain prefer
  `loader.calibrate_threshold(X_healthy, window=..., fa_rate=...)`, which
  stores a per-deployment quantile of the radar's own distances over healthy
  windows (persisted by `save()`, invalidated by `heal_domain_shift`);
  otherwise re-check the radar's in-distribution false-alarm rate on your own
  data.

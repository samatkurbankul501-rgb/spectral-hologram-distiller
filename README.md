# Spectral Hologram Distiller (SHD v3.8)

**Zero-GPU streaming dataset distillation with phase-manifold Voronoi medoids, kernel-herding spectral weights, and a 16-bit spectral-passport OOD radar.**

---

## What it does

Compress a streaming dataset into **0.2%–1.8% of its size** — 120–300 *real* medoids
(a `.hologram.npz` crystal, tens of KB) — while (a) keeping rare critical regimes
(phase transitions, high-$T_c$ superconductors) from being absorbed by the background
and (b) shipping a millisecond OOD radar that detects domain shift *without the
original data*.

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
                              max_init_buffer=8192)
for i in range(0, len(X_stream), 2500):          # O(1) RAM: peak < 60 MB
    d.partial_fit(X_stream[i:i+2500], y_stream[i:i+2500])  # or y_batch=None (label-free!)
rel_spec_err = d.finalize_crystal()               # kernel-herding weights gamma*
d.save_hologram("my_crystal.hologram.npz")        # stores the RFF projector too

loader = HologramCrystalLoader("my_crystal.hologram.npz")
X_c, y_c, w_c = loader.get_training_data()        # train any model on 1.7% of data
X_pk, y_pk, w_pk = loader.get_phase_peak_slice()  # rare / discontinuity slice for BO
report = loader.check_domain_shift(X_new_stream)  # BW radar, self-contained
print(report)  # {'bw_distance': ..., 'latency_ms': ..., 'is_shifted': False}

# Surgical self-healing when a new regime arrives (zero catastrophic forgetting):
heal_info = loader.heal_domain_shift(X_drifted, y_drifted, n_inject=35)
loader.save("my_crystal_healed.hologram.npz")
```

## CLI (`python -m shd` / `shd`)

```bash
# Inspect any .hologram.npz crystal
python -m shd inspect artifacts/superconduct_v37.hologram.npz

# Distill a CSV/NPY/NPZ stream (omit --target-col for label-free sensor mode)
python -m shd distill --input stream.csv --target-col -1 --output crystal.hologram.npz
# ...or make the crystal sensitive to rate-of-change transients:
python -m shd distill --input sensors.csv --target-col -1 --include-deltas --output crystal.hologram.npz

# Run the Bures-Wasserstein OOD radar on a new batch
python -m shd radar --crystal crystal.hologram.npz --input new_batch.csv --target-col -1

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
   blend the spectral passport without touching old medoids.

## Industrial hardening (v3.7.1)

- **Adaptive BW threshold.** The radar threshold scales as
  `T(B) = 0.085 · max(1, sqrt(1024/B))` for batch size `B`: small sensor windows
  get a wider gate so sampling noise alone cannot trigger false alarms, while
  `B ≥ 1024` uses the classic `0.085`. `check_domain_shift(..., adaptive_threshold=True)`
  (default) returns the used `threshold` inside the report. Calibrated on the
  UCI Superconductivity heal benchmark (0/20 false alarms on in-distribution
  windows of 64 rows, 20/20 detections on true OOD windows).
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
  `--streaming` on `radar`/`heal`. Package version aligned to 3.8.0.

## Multi-seed benchmark results (20 seeds, paired Wilcoxon)

Single source of truth: `bench_seeds_results.json` (generated by `bench_seeds.py`)
and `bridge_neon_bo_results.json` (generated by `bridge_neon_bo.py`).
All medians are reported with interquartile ranges (IQR) across 20 seeds `0..19`.

### 1. UCI Superconductivity ($N = 21,263$, $D = 81$) → $M = 300$ medoids (1.76%)

Fixed 80/20 split (`n_train = 17,010`, `n_test_crit = 778` with $T_c \ge 77\text{ K}$).
SHD captures a median of **64.5** (IQR 3.5) high-$T_c$ superconductors out of 300 medoids.

| Coreset ($M=300$) | KRR RMSE All (med / IQR) | KRR RMSE $T_c \ge 77\text{K}$ (med / IQR) | KRR $R^2$ (med / IQR) | GBM RMSE All (med / IQR) | GBM RMSE $T_c \ge 77\text{K}$ (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle (full train)** | 12.86 K | 17.59 K | 0.8562 | 10.27 K | 12.72 K |
| **Random (1.76%)** | 16.58 / 0.54 K | 22.43 / 2.49 K | 0.761 / 0.015 | 16.38 / 0.84 K | 22.57 / 2.49 K |
| **K-Means (1.76%)** | 17.34 / 0.55 K | 22.93 / 3.18 K | 0.739 / 0.017 | 17.19 / 0.97 K | 23.15 / 2.47 K |
| **SHD v3.7 (1.76%)** | **15.42 / 0.25 K** | **18.81 / 0.94 K** | **0.793 / 0.007** | **16.07 / 0.66 K** | **18.71 / 3.22 K** |
| *Wilcoxon SHD vs Random* | 20/20 wins, $p=1.9\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 13/20, $p=0.28$ (n.s.) | 17/20 wins, $p=1.3\times 10^{-4}$ |
| *Wilcoxon SHD vs K-Means* | 20/20 wins, $p=1.9\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 18/20 wins, $p=9.5\times 10^{-6}$ | 20/20 wins, $p=1.9\times 10^{-6}$ |

### 2. Nonlinear 4D reactor with phase transitions ($N_{\text{train}} = 16,000 \to M = 300$)

| Coreset ($M=300$) | KRR RMSE All (med / IQR) | KRR RMSE Peak (med / IQR) | KRR $R^2$ (med / IQR) | GBM RMSE All (med / IQR) | GBM RMSE Peak (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle (full train)** | 0.80 | 2.78 | 0.809 | 0.38 | 1.11 |
| **Random** | 1.134 / 0.104 | 4.424 / 0.814 | 0.621 / 0.070 | 0.711 / 0.081 | 2.409 / 0.549 |
| **K-Means** | 1.069 / 0.073 | 4.390 / 0.722 | 0.663 / 0.046 | **0.596 / 0.079** | 2.033 / 0.572 |
| **SHD v3.7** | **0.976 / 0.021** | **3.436 / 0.225** | **0.720 / 0.012** | 0.624 / 0.092 | **1.732 / 0.579** |
| *Wilcoxon SHD vs Random* | 20/20, $p=1.9\times 10^{-6}$ | 20/20, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 16/20, $p=7.1\times 10^{-4}$ | 18/20, $p=1.7\times 10^{-4}$ |
| *Wilcoxon SHD vs K-Means* | 20/20, $p=1.9\times 10^{-6}$ | 20/20, $p=1.9\times 10^{-6}$ | $p=1.9\times 10^{-6}$ | 8/20, $p=0.25$ (n.s.) | 13/20, $p=0.064$ |

### 3. Bridge with NEON Lab: Discontinuity-Aware Bayesian Optimization (`bridge_neon_bo.py`)

A cubic $\mathcal{O}(N^3)$ Matérn-5/2 Gaussian Process surrogate cannot fit raw
streaming telemetry ($N = 10,000$). When fitted on a $M = 120$ coreset and paired
with Expected Improvement (EI) over an unseen candidate pool ($N_{\text{pool}} = 3,000$,
phase-transition pocket = 12 points / 0.40% volume):

| Coreset feeding GP+EI ($M=120$) | GP RMSE inside Phase Pocket (med / IQR) | Top-20 EI Hits in Pocket out of 12 (med / IQR) | 10-step BO Simple Regret (med / IQR) | GP RMSE All Background (med / IQR) |
| :--- | :---: | :---: | :---: | :---: |
| **Random (1.2%)** | 6.028 / 0.684 | 0.0 / 2.2 | 1.783 / 5.249 | 0.437 / 0.067 |
| **K-Means (1.2%)** | 6.073 / 0.144 | 0.0 / 0.2 | 5.578 / 5.317 | **0.425 / 0.056** |
| **SHD v3.7 (1.2%)** | **4.164 / 0.693** | **6.0 / 2.0** | **0.099 / 0.266** | 0.546 / 0.031 |
| *Wilcoxon SHD vs Random* | 20/20 wins, $p=1.9\times 10^{-6}$ | 19/20 wins, $p=1.0\times 10^{-4}$ | 16/20 wins, $p=5.9\times 10^{-4}$ | 6/20, $p=0.0073$ (trade-off) |
| *Wilcoxon SHD vs K-Means* | 19/20 wins, $p=9.5\times 10^{-6}$ | 19/20 wins, $p=1.1\times 10^{-4}$ | 18/20 wins, $p=3.4\times 10^{-4}$ | 3/20, $p=1.7\times 10^{-4}$ (trade-off) |

Headline: standard GP+EI on Random or K-Means coresets gets a median of **0 out
of 12** pocket hits in its top-20 EI acquisitions, whereas the same GP+EI on the
SHD crystal hits **6 out of 12** immediately and drives 10-step BO simple regret
down to **0.099** (reaching exact 0.000 regret in 7/20 seeds).

#### 3b. End-to-end Phase Boundary Rediscovery (`SHD v3.7 → NEON Lab CPD-rank`)

When a composition-temperature stream ($N = 8,000$, quadrant phase transition
$A > 0.625 \land T > 720\text{ K}$ covering $\approx 2.8\%$ of the stream) is
compressed to $M = 120$ points and fed into NEON Lab's two-layer rank-residual
permutation detector (`discover_phase_boundary_cpd(use_rank=True, alpha=0.01)`):

| Coreset feeding CPD-rank ($M=120$) | Planted Phase Points Captured (med / IQR) | CPD-rank F1 (med / IQR) | Verified Detections ($p < 0.01 \land \text{F1} \ge 0.5$) | Boundary Error $\Delta A$ (med / IQR) | Boundary Error $\Delta T$ (med / IQR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random (1.5%)** | 3.0 / 1.2 | 0.038 / 0.145 | 1 / 20 (5%) | 0.200 / 0.231 | 105.0 / 157.5 K |
| **K-Means (1.5%)** | 5.0 / 2.0 | 0.171 / 0.460 | 4 / 20 (20%) | 0.163 / 0.206 | 105.0 / 87.5 K |
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
| Post-heal BW on Domain B | 0.465 | passport blends toward the new domain (0.636 → 0.465); one heal does **not** yet clear the 0.085 alert threshold |
| KRR RMSE on new Domain B | **26.88 K** healed vs **57.17 K** pre-heal | 20/20, $p=1.9\times10^{-6}$ |
| GBM RMSE on new Domain B | **19.38 K** vs **54.57 K** | 20/20, $p=1.9\times10^{-6}$ |
| Old Domain A retention vs naive overwrite | KRR **13.36 K** vs **20.62 K** | 20/20, $p=1.9\times10^{-6}$ |
| Combined A+B vs pre-heal / naive overwrite | **19.10 K** vs 33.86 / 21.58 | 20/20 both |
| Bounded heal (`max_crystal_size=220`) | combined **19.15 K** vs 33.86 | 20/20 — size control costs ≈ nothing |

Headline: on a physically meaningful shift (low-$T_c$ iron-based → cuprate
superconductors) the naive response is either blindness (RMSE 57 K) or catastrophic
forgetting (old domain degrades 13.4 → 20.6 K). Surgical heal captures the new
domain *and* keeps the old one — with a hard cap on crystal size for production
memory budgets.

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

## Known limitations

- **Deliberate background trade-off:** reserving 26% of medoid cells for the
  phase-peak band slightly increases background GP RMSE (`0.546` vs `0.425` on
  the BO bridge) and ties with K-Means on tree-based global RMSE (`p = 0.25` on
  4D reactor GBM). Use SHD when rare / critical regimes matter.
- **Legacy artifact:** `artifacts/superconduct_v37.hologram.npz` predates the
  stored projector; its OOD radar requires `check_domain_shift(X, distiller_ref=...)`.
  Crystals saved by `save_hologram(include_projector=True)` are self-contained.
- **OOD threshold:** the base `BW_ALERT_THRESHOLD = 0.085` (auto-widened for
  small batches as `0.085·sqrt(1024/B)`) is calibrated on the superconductivity
  heal benchmark, not a distribution-free constant. For a new domain, re-check
  the radar's in-distribution false-alarm rate on your own data.

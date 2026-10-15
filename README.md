# Spectral Hologram Distiller (SHD v3.7)

**Zero-GPU streaming dataset distillation with phase-manifold Voronoi medoids, kernel-herding spectral weights, and a 16-bit spectral-passport OOD radar.**

---

## What it does

Compress a streaming dataset into **0.2%–1.8% of its size** — 200–300 *real* medoids
(a `.hologram.npz` crystal, tens of KB) — while (a) keeping rare critical regimes
(phase transitions, high-Tc superconductors) from being absorbed by the background
and (b) shipping a millisecond OOD radar that detects domain shift *without the
original data*.

Honest terminology: the spectral reweighting step is **kernel herding** solved with
FISTA; the cell-mass prior with root tempering `p^0.55` is **importance weighting**.
(Earlier notes called these "Hadamard-Gram quadrature" and "Radon-Nikodym
tempering" — same math.)

## Install

```bash
pip install -e .
# or just point PYTHONPATH at the repo root; dependencies: numpy, scipy, scikit-learn
```

## Quickstart

```python
from shd import SpectralHologramDistiller, HologramCrystalLoader

d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                              sketch_dim=256, tail_quantile=84.0, phase_kappa=3.2)
for i in range(0, len(X_stream), 2500):          # O(1) RAM: peak < 60 MB
    d.partial_fit(X_stream[i:i+2500], y_stream[i:i+2500])
rel_spec_err = d.finalize_crystal()               # kernel-herding weights gamma*
d.save_hologram("my_crystal.hologram.npz")        # stores the RFF projector too

loader = HologramCrystalLoader("my_crystal.hologram.npz")
X_c, y_c, w_c = loader.get_training_data()        # train any model on 1.7% of data
report = loader.check_domain_shift(X_new_stream)  # BW radar, self-contained
print(report)  # {'bw_distance': ..., 'latency_ms': ..., 'is_shifted': False}
```

## Pipeline

1. **RFF sketch (Bochner):** `U = cos(Z·Ω + φ)`, 256 harmonics, row-normalised;
   online accumulators `Ψ = UᵀU`, `Φ = Uᵀy`.
2. **Residual trigger:** ridge predictor on the sketch; top-`q` percentile of
   `residual_ratio + 2·y_z²` marks phase-peak candidates.
3. **Phase-manifold lattice:** one non-overlapping Voronoi tessellation on
   `[Z ‖ κ·y]` (κ = 3.2), medoid-snapped — rare regimes get their own cells.
4. **Kernel-herding weights (FISTA):** `Σ γᵢuᵢuᵢᵀ ≈ Ψ̂`; Voronoi mass prior,
   root-tempered `p^0.55`.
5. **Passport + OOD radar:** top-32 eigenpairs of `Ψ̂` in float16 →
   Bures-Wasserstein distance on a 32-dim subspace (alert threshold 0.085).

## Benchmark results

### 1. UCI Superconductivity (N = 21,263, D = 81) → 300 medoids (1.76%), 97.7 KB, spectral error 4.67%

| Training set | KRR RMSE (all) | KRR RMSE (HTS Tc ≥ 77K) | KRR R² | GBM RMSE (all) | GBM RMSE (HTS Tc ≥ 77K) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Oracle (100%) | 12.86 K | 17.59 K | 0.8562 | 10.29 K | 12.72 K |
| Random (1.76%) | 16.89 K | 22.96 K | 0.7522 | 17.04 K | 23.88 K |
| K-Means (1.76%) | 16.66 K | 21.73 K | 0.7588 | 17.05 K | 25.07 K |
| **Spectral Hologram v3.7 (1.76%)** | **15.69 K** | **17.75 K** | **0.7862** | **16.75 K** | **17.60 K** |

Headline: on the *rare* HTS class, kernel ridge on 1.76% of the data is within
0.16 K of the full-dataset Oracle — while Random and K-Means coresets lose 4–7 K.
The trade-off is deliberate: global RMSE is worse than Oracle by design (the
crystal privileges the rare regime).

### 2. Nonlinear 4D reactor with phase transitions (N = 100,000 → 200 medoids, 0.2%, 21.1 KB)

| Model | Random R²_peak | K-Means R²_peak | **SHD R²_peak** |
| :--- | :---: | :---: | :---: |
| KernelRidge (RBF) | 0.7256 | 0.8329 | **0.9116** |
| HistGradientBoosting | 0.1501 | 0.3800 | **0.7288** |
| MLP (L-BFGS) | 0.6875 | 0.6143 | **0.9186** |
| PyTorch weighted MLP | 0.6875 | 0.6143 | **0.9300** |

Fit-time note: KernelRidge fitting on 200 points takes ~2.4 ms vs ~29 s on the
full dataset — that is a *training-stage* speedup of a kernel method on a small
coreset, not a claim about the distillation itself.

### 3. Self-healing under domain shift
- Shift detected by the 16-bit passport: BW = 0.8508 (ALERT) in ~22.7 ms.
- Injecting 35 new medoids (~39.5 ms): R² on the new anomaly −9.72 → **+0.954**,
  with R² on the old peak retained at 0.911 (no catastrophic forgetting).

## Known limitations (be honest before you build on this)

- All published numbers are **single-seed (seed=42) runs** in Colab; no
  multi-seed statistics yet (step 2 on the roadmap: ≥20 seeds + Wilcoxon).
- The shipped artifact `artifacts/superconduct_v37.hologram.npz` predates the
  stored projector; its OOD radar needs a live distiller reference
  (`check_domain_shift(X, distiller_ref=...)`). New crystals are self-contained.
- The OOD threshold (0.085) is calibrated on one dataset; treat it as a default,
  not a universal constant.

## Artifacts

- `artifacts/superconduct_v37.hologram.npz` — UCI Superconductivity crystal
  (300 × 81, n_seen = 17010), verified loadable.
- `tests/test_smoke.py` — distill → save → load → OOD radar round-trip.

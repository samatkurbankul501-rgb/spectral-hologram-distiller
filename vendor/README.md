# Vendored NEON Lab CPD module

`jarvis_v2_cpd.py` and `jarvis_v2_discovery.py` are vendored from the sibling
JARVIS Lab / NEON Lab repository so the SHD bridge benchmark is reproducible
from this repository alone.

- `jarvis_v2_cpd.py`: discontinuity-aware CPD-Rank detector (quadrant step
  model + permutation test).
- `jarvis_v2_discovery.py`: the `_fit_gp` Matérn-5/2 GP surrogate used by the
  CPD refinement.

The files are unmodified. They are the author's own JARVIS Lab code (same
author as this SHD repository) and are included under the repository's MIT
license. If the original NEON Lab repository is present, `bridge_neon_bo.py`
still falls back to it; the vendored copy is the default.

## Known upstream defect (documented, not patched)

In `jarvis_v2_cpd.py`, `fit_step_discontinuity` computes the OLS step
coefficient with one extra division by `n`:

- line 142: `cov_k = ((M[k_star] @ rv0) / n)` — the covariance is already
  per-point (divided by `n`);
- line 143: `beta_hat = float(cov_k / (var_s[k_star] * n))` — divides by
  `var_s * n` again, so the returned `beta_hat` is understated by a factor
  of `n` (the correct value is `cov_k / var_s[k_star]`).

Consequences, checked against the code as vendored:

- **Detection does not suffer.** The detection statistics are
  scale-invariant in `beta_hat`: `rho^2 = cov^2 / (var_r * var_s)` (line
  124) and the permutation `p_value` (lines 133–139) are computed directly
  from the centered residuals, never from `beta_hat`. F1 / precision /
  recall, the anomaly mask and the boundary estimates `(a0_hat, t0_hat_K)`
  are therefore unaffected.
- **The GP refinement pass degenerates.** The backfitting subtraction at
  line 187 (`_fit_gp(X, y - res['beta_hat'] * step)`) removes almost
  nothing because `beta_hat` is ~n times too small, so the GP still sees
  the step in the data — exactly the situation the two-pass design was
  meant to avoid.
- **The p-value guard masks the degeneration.** Lines 191–192 reject the
  GP pass whenever its `p_value` does not improve, so the final result is
  the poly-pass step. This is why the detector still reports correct
  detections despite the broken coefficient.

Per the unmodified-vendor policy these files are intentionally **not**
patched in this repository. Interpret the `.bridge` benchmark
(`bridge_neon_bo.py` / `bridge_neon_bo_results.json`) accordingly: the
reported CPD metrics are effectively the poly-pass metrics, and the
`beta_hat` values it records should not be read as physical step heights.

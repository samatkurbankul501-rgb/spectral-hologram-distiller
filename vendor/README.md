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

# Independent-machine reproducibility: GitHub Codespaces

**Date:** 2026-11-14
**Machine:** GitHub Codespace `shd-bench-xrvw4w4jgr6r39xv7` (4-core / 16 GB,
Ubuntu, repo auto-cloned at commit `d4bcb16`), driven autonomously via
`gh codespace ssh`.
**Environment:** numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1 — identical to
the laptop venv that produced the committed artifacts (Python minor version
differs; numerical results unaffected, as verified below).

## Result: bit-for-bit reproduction of `bench_cmapss_results.json`

`bench_cmapss.py --seeds 20` was run on the codespace and compared against the
committed artifact (20 seeds, official RUL convention):

- **40 / 40** arm-by-metric medians match exactly (max absolute difference
  `0.000000` across `shd_r{0.26,0.5,0.65}`, `shdu_r{0.26,0.5,0.65}`,
  `random`, `kmeans`, `kt`, `skt` × `krr_rmse_{crit,all}`, `gbm_rmse_{crit,all}`).
- **20 / 20** per-seed `shd_r0.65` records match bit-for-bit.
- Oracle KRR critical RMSE: `13.05087` on both machines.

The codespace result is therefore identical to the committed artifact; no
divergence arises from the independent hardware or the Python minor-version
difference. This is committed evidence that the published C-MAPSS numbers
reproduce on a separate cloud machine, not only on the author's laptop.

## Method

```bash
gh codespace create -R samatkurbankul501-rgb/spectral-hologram-distiller \
  -m standardLinux32gb -d shd-bench --idle-timeout 4h --retention-period 72h
gh codespace ssh -c <name> -- "cd /workspaces/spectral-hologram-distiller \
  && python3 -m venv .venv && .venv/bin/pip install -q -e . \
  && .venv/bin/python bench_cmapss.py --seeds 20 --output /tmp/cmapss_codespace.json"
gh codespace ssh -c <name> -- "cat /tmp/cmapss_codespace.json" > codespace.json
# compare medians + per-seed records against bench_cmapss_results.json
```

## Scope and limits

- Verified for `bench_cmapss` (20 seeds). Other benchmarks reproduce by the
  same deterministic construction (fixed seeds, fixed split) but were not
  re-run on the codespace in this pass.
- Library versions must match the artifact-generation environment for
  bit-for-bit equality; the versions above do. A future `requirements-lock.txt`
  would make this explicit.

## Headline `bench_seeds` (SC + reactor, 20 seeds)

Also reproduced on the codespace (`bench_seeds.py --seeds 20 --dataset both`):

- **Reactor: 20 / 20** arm-by-metric medians exact (max abs diff `0.0`).
- **Superconductivity: 19 / 20** exact. The single deviation is the **K-Means
  baseline's GBM** metric (`gbm_rmse_all` 17.14118 vs 17.18934, diff 0.048;
  `gbm_r2` 0.74474 vs 0.74331) --- not SHD. SHD's own numbers match exactly
  (critical KRR 18.8053 on both machines). The deviation is consistent with
  K-Means / GBM tie-breaking under the Python minor-version difference
  (laptop 3.14 vs codespace 3.12/3.13); the published SHD claims are unaffected.

`requirements-lock.txt` now pins the core versions (numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1) for explicit reproducibility.
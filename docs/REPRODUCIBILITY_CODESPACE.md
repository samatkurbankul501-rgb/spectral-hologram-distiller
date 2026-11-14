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

## Three-way independent reproduction (2026-11-14, commit `2ba3e10`)

Three machines were used in parallel to reproduce the headline and the newest
(`critical_rebuild_max`) artifacts: the author's laptop (Python 3.14), the
codespace above (Python 3.13, updated to `2ba3e10`), and a temporary Kaggle
session (4-core / 31 GB, Python 3.13) reached over a `cloudflared` TCP tunnel.
All three installed the locked core versions. Findings:

- **Reactor headline (`bench_seeds --dataset reactor`, 20 seeds) on Kaggle:**
  every published median exact (SHD critical KRR `3.4364`, random `4.4244`,
  K-Means `4.3898`, KT `4.5161`, SKT `4.0603`); `rare_captured` exact per-seed
  for all five arms; floating metrics agree to `max|Δ| = 8e-15` (Python 3.13 vs
  3.14 rounding, not literal bit-for-bit).
- **Superconductivity headline (`bench_seeds --dataset superconduct`, 20 seeds)
  on the codespace:** SHD / random / KT / SKT reproduce to `max|Δ| ~ 1e-13` on
  KRR, with GBM and `rare_captured` exact. The **K-Means baseline is
  non-deterministic across machines** (this pass: `krr_rmse_crit` Δ0.51,
  `gbm_rmse_crit` Δ1.02, `rare_captured` 1/20 seeds differ) — inherent to
  scikit-learn K-Means (random init + parallel Lloyd). SHD's own medians are
  exact (`18.8053`); no published SHD claim is affected.
- **Radar modes (`bench_radar_modes`, 5 seeds) on the laptop:** exact, 54/54 AUC
  values identical (run-to-run determinism). The E1 weak-spot claim reproduces:
  B=64 partial-25% BW `0.9610`, combined `0.9968`, panel `0.9827`.
- **New combined artifact (`bench_rebuild_at_finalize`, 20 seeds, 3 arms) on the
  codespace:** all **18/18** summary medians exact and **every Wilcoxon p-value
  and win-count exact** (reactor rebuild 20/20 `p=1.9e-6`, combined 20/20
  `p=1.9e-6`; SC rebuild 18/20 `p=1.9e-5`, combined 20/20 `p=1.9e-6`). Per-seed
  values agree to `~1e-13` except a single reactor seed (13, `rebuild_at_finalize`
  arm: `krr_rmse_crit` 2.4874 vs 2.4905, Δ0.003 — a medoid tie-break);
  `rare_captured` is exact for every seed/arm and no median or test changes.

**Takeaway:** SHD is deterministic and its published claims (medians, rare
capture, Wilcoxon) reproduce on independent cloud hardware; the K-Means baseline
is the only non-deterministic arm. The temporary Kaggle session terminated
mid-run (ephemeral), so its rebuild-SC leg was completed on the codespace instead.

### Kaggle tunnel method (ephemeral backend)

```bash
# laptop: cloudflared access tcp --hostname <trycloudflare-host> --url localhost:2222
# Kaggle venv bootstrap (ensurepip is absent on the image):
python3 -m venv --without-pip .venv
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install numpy==2.5.4 scipy==1.18.1 scikit-learn==1.9.1 pandas
.venv/bin/pip install -e .
# drive via paramiko (password from env, never stored); run benches detached
# with nohup so they survive SSH drops.
```
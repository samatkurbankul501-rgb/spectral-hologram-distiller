"""Round-3 probes (2026-10-22): why does FD001 suppress SHD's metrics?

README §10 (bench_cmapss_results.json) left three weak spots on the table:
1. br=0.65 pays a SIGNIFICANT global cost on FD001 (KRR-all 21.40 vs best
   baseline 19.62), steeper than on SC.
2. The KRR oracle is not beaten at crit (8/20) while the GBM crystal beats
   the full-pool GBM oracle 19/20 -- a KRR-specific asymmetry.
3. The operating-point frontier is NON-MONOTONE here: br=0.50 KRR crit
   (15.51) is worse than both br=0.26 (14.94) and br=0.65 (14.52), unlike
   the monotone frontier of SC / reactor in §6.

Three suspects, measured below on the exact bench_cmapss stream:

A. TRIGGER FLOOD -- phase_score = res_ratio + 2*y_z^2. On a weakly-fit
   target (14 noisy raw sensors -> capped RUL) the residual ratio may be
   heavy-tailed and dominate the y-extremeness term, so the band seeds from
   high-residual noise instead of near-failure rows. Measured on the
   warm-up buffer: candidate purity (true-critical among the top-16% scores),
   recall of true-critical rows, and the res/y contribution split -- with
   the UCI Superconductivity trigger as the contrast that WORKS.

B. WEIGHT STARVATION -- gamma weights blend Voronoi volumes (p^0.55) with
   the FISTA sketch match. Band cells carry tiny volumes, so the very rows
   the crystal exists to keep get the SMALLEST weights. Measured: median
   gamma weight of near-failure medoids vs background medoids per br, and
   the same crystals re-evaluated with uniform weights (the §4 knob).

C. BAND COMPOSITION per br -- the purity numbers behind the non-monotone
   middle point.

Run: python tools/hunt/round3_probes.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from bench_cmapss import CRIT_LE, RUL_CAP, evaluate, load_fd001
from bench_seeds import load_superconduct
from shd import SpectralHologramDistiller

BATCH, N_BUF = 2500, 8192


def trigger_autopsy(name, X, y, is_true, br, tail):
    """Feed a pre-build warm-up buffer and dissect the trigger cut.

    The lattice builder consumes (clears) init_*, so the probe stays below
    the build threshold: the real stream builds the lattice once ~4,500+
    rows are buffered (target_buf = max(4500, M*12)), so a 4,000-row
    pre-build slice measures the same score distribution the build-time
    percentile cut will see.
    """
    d = SpectralHologramDistiller(
        crystal_size=300, high_band_ratio=br, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, tail=tail,
        max_init_buffer=N_BUF, random_state=0,
    )
    n = min(4000, len(X))
    d.partial_fit(X[:n], y[:n])
    scores = np.concatenate(d.init_score)
    ys = np.concatenate(d.init_Y).reshape(-1)
    if len(ys) < 500:
        print(f"[{name}] warm-up buffer not retained (len={len(ys)}); abort")
        return
    cut = float(np.percentile(scores, 84.0))
    cand = scores >= cut
    y_norm = (ys - d.y_mean) / d.y_std
    if tail == "low":
        yz = np.maximum(0.0, -y_norm)
    else:
        yz = np.maximum(0.0, y_norm)
    y_part = 2.0 * yz ** 2
    res_part = scores - y_part
    true = is_true(ys)
    purity = 100.0 * np.mean(true[cand]) if cand.any() else 0.0
    recall = 100.0 * np.mean(cand[true]) if true.any() else 0.0
    print(f"[{name}] br={br} tail={tail} buffer={len(ys)} cut={cut:.2f}")
    print(f"  top-16% score candidates: {int(cand.sum())} | true-critical among them: "
          f"{int(true[cand].sum())} ({purity:.0f}% purity)")
    print(f"  true-critical in buffer: {int(true.sum())} | recall into candidates: {recall:.0f}%")
    print(f"  candidate score split (median): res_part={np.median(res_part[cand]):9.2f} "
          f"y_part={np.median(y_part[cand]):9.2f}")
    print(f"  res_part quantiles all rows (50/90/99%): "
          f"{np.percentile(res_part, [50, 90, 99]).round(2)}")
    print(f"  y_part   quantiles all rows (50/90/99%): "
          f"{np.percentile(y_part, [50, 90, 99]).round(2)}")


def eval_autopsy():
    """Same crystals as bench_cmapss; gamma vs uniform weights per br."""
    X_pool, y_pool, X_te, y_te = load_fd001()
    perm = np.random.default_rng(42).permutation(len(X_pool))
    X_tr, y_tr = X_pool[perm], y_pool[perm]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_tr)
    Z_te = scaler.transform(X_te)
    rbf_gamma = 1.0 / (X_tr.shape[1] * 1.1)
    crit = y_te <= CRIT_LE

    print("=" * 78)
    print("PROBE B/C: weights and band composition per br (5 seeds)")
    print("=" * 78)
    rows = []
    for br in (0.26, 0.50, 0.65):
        for seed in range(5):
            d = SpectralHologramDistiller(
                crystal_size=300, high_band_ratio=br, sketch_dim=256,
                tail_quantile=84.0, phase_kappa=3.2, tail="low",
                random_state=seed,
            )
            for i in range(0, len(X_tr), BATCH):
                d.partial_fit(X_tr[i:i + BATCH], y_tr[i:i + BATCH])
            d.finalize_crystal()
            yc = np.asarray(d.crystal_Y, dtype=np.float64)
            wg = np.asarray(d.gamma_weights, dtype=np.float64)
            low = yc <= CRIT_LE
            rows.append({
                "br": br, "seed": seed, "n_low": int(low.sum()),
                "w_low": float(np.median(wg[low])) if low.any() else float("nan"),
                "w_high": float(np.median(wg[~low])),
                "gamma": evaluate(d.crystal_X, yc, wg, scaler, 42, X_te, Z_te, y_te, crit, rbf_gamma),
                "unif": evaluate(d.crystal_X, yc, np.ones(len(yc)), scaler, 42, X_te, Z_te, y_te, crit, rbf_gamma),
            })
    for br in (0.26, 0.50, 0.65):
        sel = [r for r in rows if r["br"] == br]
        n_low = np.median([r["n_low"] for r in sel])
        ratio = np.median([r["w_low"] / r["w_high"] for r in sel])
        line = f"  br={br}: low medoids={n_low:.0f}/300 | median w(low)/w(bg)={ratio:.3f}"
        for tag in ("gamma", "unif"):
            ev = sel[0][tag]
            med = lambda k: float(np.median([r[tag][k] for r in sel]))
            line += (f" | {tag}: krr_all={med('krr_rmse_all'):.2f} "
                     f"krr_crit={med('krr_rmse_crit'):.2f} "
                     f"gbm_all={med('gbm_rmse_all'):.2f} "
                     f"gbm_crit={med('gbm_rmse_crit'):.2f}")
        print(line)


if __name__ == "__main__":
    print("=" * 78)
    print("PROBE A: trigger autopsy (warm-up buffer, top-16% candidates)")
    print("=" * 78)
    X_pool, y_pool, _, _ = load_fd001()
    perm = np.random.default_rng(42).permutation(len(X_pool))
    Xf, yf = X_pool[perm], y_pool[perm]
    for br in (0.26, 0.65):
        trigger_autopsy("FD001", Xf, yf, lambda v: v <= CRIT_LE, br, "low")
    # Contrast: the trigger that works (SC, tail=high, br=0.26).
    Xs, ys, _, _ = load_superconduct()
    permp = np.random.default_rng(42).permutation(len(Xs))
    trigger_autopsy("SC", Xs[permp], ys[permp], lambda v: v >= 77.0, 0.26, "high")
    print()
    eval_autopsy()

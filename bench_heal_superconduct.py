"""Real-data OOD radar & surgical self-healing benchmark on UCI Superconductivity.

Physical domain-shift protocol (20 seeds, paired Wilcoxon):
- Dataset: UCI Superconductivity (N = 21,263 materials, D = 81 chemical features).
- Domain A (In-Distribution, Low-Tc regime: Tc < 40 K): conventional and low-Tc
  superconductors (~13,900 rows). Split 80/20 into stream_A and test_A.
- Domain B (Out-of-Distribution shift, High-Tc regime: Tc >= 40 K): cuprates and
  high-Tc families (~7,300 rows) with distinct oxygen/copper stoichiometry.
  Split 80/20 into stream_B and test_B.

For each seed (0..19):
1. Stream Domain A through SHD v3.7 into an M = 200 crystal and save to a temporary
   .hologram.npz with its 32-mode float16 passport and RFF projector.
2. Check Bures-Wasserstein radar on held-out test_A (must NOT alert: 0% false alarms)
   and on incoming stream_B (must alert: 100% OOD detection in a few ms).
3. Compare downstream KRR and HistGBM regressors across 4 regimes on test_A (old
   domain retention) and test_B (new domain adaptation):
   - Pre-heal crystal (M = 200, Domain A only);
   - Naive overwrite coreset (M = 200 on Domain B only -> catastrophic forgetting);
   - SHD surgical self-healing (`loader.heal_domain_shift(stream_B, n_inject=45)`);
   - SHD bounded self-healing (`loader.heal_domain_shift(..., n_inject=45, max_crystal_size=220)`).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller


def eval_models(X_c, y_c, w_c, X_test_A, y_test_A, X_test_B, y_test_B):
    scaler = StandardScaler().fit(X_c)
    Z_c = scaler.transform(X_c)
    Z_A = scaler.transform(X_test_A)
    Z_B = scaler.transform(X_test_B)

    krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0 / Z_c.shape[1])
    krr.fit(Z_c, y_c, sample_weight=w_c)
    pred_A_krr = krr.predict(Z_A)
    pred_B_krr = krr.predict(Z_B)

    gbm = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
    gbm.fit(Z_c, y_c, sample_weight=w_c)
    pred_A_gbm = gbm.predict(Z_A)
    pred_B_gbm = gbm.predict(Z_B)

    return {
        "krr_rmse_domain_A": float(np.sqrt(mean_squared_error(y_test_A, pred_A_krr))),
        "krr_rmse_domain_B": float(np.sqrt(mean_squared_error(y_test_B, pred_B_krr))),
        "krr_rmse_combined": float(
            np.sqrt(
                mean_squared_error(
                    np.concatenate([y_test_A, y_test_B]),
                    np.concatenate([pred_A_krr, pred_B_krr]),
                )
            )
        ),
        "gbm_rmse_domain_A": float(np.sqrt(mean_squared_error(y_test_A, pred_A_gbm))),
        "gbm_rmse_domain_B": float(np.sqrt(mean_squared_error(y_test_B, pred_B_gbm))),
        "gbm_rmse_combined": float(
            np.sqrt(
                mean_squared_error(
                    np.concatenate([y_test_A, y_test_B]),
                    np.concatenate([pred_A_gbm, pred_B_gbm]),
                )
            )
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--crystal-size", type=int, default=200)
    ap.add_argument("--n-inject", type=int, default=45)
    args = ap.parse_args()

    t0 = time.perf_counter()
    X, y, _, ds_name = load_superconduct()
    mask_A = y < 40.0
    mask_B = y >= 40.0

    X_A, y_A = X[mask_A], y[mask_A]
    X_B, y_B = X[mask_B], y[mask_B]

    X_tr_A, X_te_A, y_tr_A, y_te_A = train_test_split(X_A, y_A, test_size=0.20, random_state=42)
    X_tr_B, X_te_B, y_tr_B, y_te_B = train_test_split(X_B, y_B, test_size=0.20, random_state=42)

    print(
        f"[heal-bench] {ds_name}: Domain A (Tc<40K) train={len(X_tr_A)}, test={len(X_te_A)} | "
        f"Domain B (Tc>=40K) train={len(X_tr_B)}, test={len(X_te_B)}"
    )

    tmp_dir = tempfile.mkdtemp(prefix="shd_heal_bench_")
    records = []

    for seed in range(args.seeds):
        ts = time.perf_counter()
        rng = np.random.default_rng(1000 + seed)
        perm_A = rng.permutation(len(X_tr_A))
        perm_B = rng.permutation(len(X_tr_B))
        Xs_A, ys_A = X_tr_A[perm_A], y_tr_A[perm_A]
        Xs_B, ys_B = X_tr_B[perm_B], y_tr_B[perm_B]

        d = SpectralHologramDistiller(
            crystal_size=args.crystal_size,
            high_band_ratio=0.26,
            sketch_dim=256,
            tail_quantile=84.0,
            phase_kappa=3.2,
            random_state=seed,
        )
        for i in range(0, len(Xs_A), 2500):
            d.partial_fit(Xs_A[i : i + 2500], ys_A[i : i + 2500])
        d.finalize_crystal()

        cryst_path = os.path.join(tmp_dir, f"cryst_s{seed}.hologram.npz")
        d.save_hologram(cryst_path, include_projector=True)

        loader = HologramCrystalLoader(cryst_path)
        radar_in = loader.check_domain_shift(X_te_A)
        radar_ood = loader.check_domain_shift(Xs_B)

        # 1. Pre-heal metrics
        X_pre, y_pre, w_pre = loader.get_training_data()
        m_pre = eval_models(X_pre, y_pre, w_pre, X_te_A, y_te_A, X_te_B, y_te_B)

        # 2. Naive overwrite on new stream B (catastrophic forgetting baseline)
        idx_over = rng.choice(len(Xs_B), size=args.crystal_size, replace=False)
        m_over = eval_models(
            Xs_B[idx_over],
            ys_B[idx_over],
            np.ones(args.crystal_size),
            X_te_A,
            y_te_A,
            X_te_B,
            y_te_B,
        )

        # 3. SHD surgical self-healing (unbounded append: 200 + 45 = 245 medoids)
        heal_rep = loader.heal_domain_shift(Xs_B, ys_B, n_inject=args.n_inject, random_state=seed)
        X_h, y_h, w_h = loader.get_training_data()
        m_heal = eval_models(X_h, y_h, w_h, X_te_A, y_te_A, X_te_B, y_te_B)

        # 4. SHD bounded self-healing (max_crystal_size=220 -> prunes 25 background medoids)
        loader_cap = HologramCrystalLoader(cryst_path)
        heal_cap_rep = loader_cap.heal_domain_shift(
            Xs_B,
            ys_B,
            n_inject=args.n_inject,
            max_crystal_size=args.crystal_size + 20,
            random_state=seed,
        )
        X_hc, y_hc, w_hc = loader_cap.get_training_data()
        m_heal_cap = eval_models(X_hc, y_hc, w_hc, X_te_A, y_te_A, X_te_B, y_te_B)

        rec = {
            "seed": seed,
            "radar_in_bw": float(radar_in["bw_distance"]),
            "radar_in_shifted": bool(radar_in["is_shifted"]),
            "radar_ood_bw": float(radar_ood["bw_distance"]),
            "radar_ood_shifted": bool(radar_ood["is_shifted"]),
            "radar_ood_latency_ms": float(radar_ood["latency_ms"]),
            "bw_after_heal": float(heal_rep["bw_after"]),
            "heal_latency_ms": float(heal_rep["latency_ms"]),
            "pre_heal": m_pre,
            "naive_overwrite": m_over,
            "healed": m_heal,
            "healed_bounded": m_heal_cap,
            "n_pruned_bounded": int(heal_cap_rep["n_pruned"]),
        }
        records.append(rec)
        print(
            f"[heal] seed {seed:2d}: BW in={radar_in['bw_distance']:.3f} (shift={radar_in['is_shifted']}) "
            f"ood={radar_ood['bw_distance']:.3f} (shift={radar_ood['is_shifted']}) -> after={heal_rep['bw_after']:.3f} | "
            f"KRR B: pre={m_pre['krr_rmse_domain_B']:.2f} -> healed={m_heal['krr_rmse_domain_B']:.2f} | "
            f"KRR A: pre={m_pre['krr_rmse_domain_A']:.2f} healed={m_heal['krr_rmse_domain_A']:.2f} "
            f"overwrite={m_over['krr_rmse_domain_A']:.2f} ({time.perf_counter()-ts:.1f}s)"
        )

    bw_in_arr = np.array([r["radar_in_bw"] for r in records])
    bw_ood_arr = np.array([r["radar_ood_bw"] for r in records])
    bw_after_arr = np.array([r["bw_after_heal"] for r in records])

    summary = {
        "n_seeds": args.seeds,
        "dataset": ds_name,
        "crystal_size_base": args.crystal_size,
        "n_inject": args.n_inject,
        "elapsed_sec": time.perf_counter() - t0,
        "radar_false_alarms_on_domain_A": int(sum(1 for r in records if r["radar_in_shifted"])),
        "radar_detections_on_domain_B": int(sum(1 for r in records if r["radar_ood_shifted"])),
        "bw_in_median": float(np.median(bw_in_arr)),
        "bw_in_iqr": float(np.percentile(bw_in_arr, 75) - np.percentile(bw_in_arr, 25)),
        "bw_ood_median": float(np.median(bw_ood_arr)),
        "bw_ood_iqr": float(np.percentile(bw_ood_arr, 75) - np.percentile(bw_ood_arr, 25)),
        "bw_after_heal_median": float(np.median(bw_after_arr)),
        "radar_latency_ms_median": float(np.median([r["radar_ood_latency_ms"] for r in records])),
        "heal_latency_ms_median": float(np.median([r["heal_latency_ms"] for r in records])),
    }

    comparisons = {}
    for metric in [
        "krr_rmse_domain_A",
        "krr_rmse_domain_B",
        "krr_rmse_combined",
        "gbm_rmse_domain_A",
        "gbm_rmse_domain_B",
        "gbm_rmse_combined",
    ]:
        for target_regime in ["healed", "healed_bounded"]:
            for base_regime in ["pre_heal", "naive_overwrite"]:
                a = np.array([r[target_regime][metric] for r in records], dtype=np.float64)
                b = np.array([r[base_regime][metric] for r in records], dtype=np.float64)
                diff = a - b
                stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
                wins = int(np.sum(diff < 0))
                comparisons[f"{target_regime}_vs_{base_regime}_{metric}"] = {
                    f"median_{target_regime}": float(np.median(a)),
                    f"iqr_{target_regime}": float(np.percentile(a, 75) - np.percentile(a, 25)),
                    f"median_{base_regime}": float(np.median(b)),
                    f"iqr_{base_regime}": float(np.percentile(b, 75) - np.percentile(b, 25)),
                    "wins": wins,
                    "p_value": float(p),
                }

    out = {
        "summary": summary,
        "wilcoxon": comparisons,
        "records": records,
    }
    out_path = "bench_heal_superconduct_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {out_path} ({summary['elapsed_sec']:.1f}s)")
    print(
        f"  Radar: false alarms on Domain A = {summary['radar_false_alarms_on_domain_A']}/{args.seeds}, "
        f"OOD detections on Domain B = {summary['radar_detections_on_domain_B']}/{args.seeds} "
        f"(BW in={summary['bw_in_median']:.3f}, OOD={summary['bw_ood_median']:.3f}, post-heal={summary['bw_after_heal_median']:.3f})"
    )
    for k in [
        "healed_vs_pre_heal_krr_rmse_domain_B",
        "healed_vs_naive_overwrite_krr_rmse_domain_A",
        "healed_vs_pre_heal_krr_rmse_combined",
        "healed_vs_naive_overwrite_krr_rmse_combined",
        "healed_bounded_vs_pre_heal_krr_rmse_combined",
        "healed_vs_pre_heal_gbm_rmse_domain_B",
        "healed_vs_naive_overwrite_gbm_rmse_domain_A",
    ]:
        v = comparisons[k]
        keys = [x for x in v if x.startswith("median_")]
        print(
            f"  {k:46s}: {v[keys[0]]:.2f} vs {v[keys[1]]:.2f} | wins={v['wins']}/{args.seeds} | p={v['p_value']:.2e}"
        )


if __name__ == "__main__":
    main()

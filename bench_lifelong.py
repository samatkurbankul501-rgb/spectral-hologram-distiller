"""Lifelong (3-shift) heal benchmark on UCI Superconductivity.

The paper reports a single A -> B heal. This script chains three shifts,
A (<20 K) -> B (20-40 K) -> C (40-77 K) -> D (>=77 K), with a bounded crystal,
and evaluates every domain after every heal. It answers two questions the
single-shift benchmark cannot:

1. Does the passport converge after repeated heals, or does BW stay alarmed?
2. Does old-domain retention degrade cumulatively (true catastrophic forgetting)?

Usage:
    python bench_lifelong.py --seeds 5
Writes bench_lifelong_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller

BANDS = {
    "A_lt20": lambda y: y < 20,
    "B_20_40": lambda y: (y >= 20) & (y < 40),
    "C_40_77": lambda y: (y >= 40) & (y < 77),
    "D_77_plus": lambda y: y >= 77,
}


def fit_eval(loader, tests, d_feat):
    from sklearn.preprocessing import StandardScaler

    Xc, yc, wc = loader.get_training_data()
    scaler = StandardScaler().fit(Xc)
    Zc = scaler.transform(Xc)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (d_feat * 1.1))
    krr.fit(Zc, yc, sample_weight=wc)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42
    )
    gbm.fit(Zc, yc, sample_weight=wc)
    out = {}
    for name, (X_te, y_te) in tests.items():
        Z_te = scaler.transform(X_te)
        out[name] = {
            "krr_rmse": float(np.sqrt(mean_squared_error(y_te, krr.predict(Z_te)))),
            "gbm_rmse": float(np.sqrt(mean_squared_error(y_te, gbm.predict(Z_te)))),
            "n_test": int(len(y_te)),
        }
    return out


def run(seeds, crystal_size=200, n_inject=45, max_crystal_size=260,
        new_weight_boost=1.0, passport_blend_min=0.15, passport_blend_max=0.50,
        protect_old_weight_frac=0.0):
    X, y, _, title = load_superconduct()
    if X.shape[1] != 81:
        raise RuntimeError("UCI fetch failed (fallback dataset); aborting")
    trains, tests = {}, {}
    for name, fn in BANDS.items():
        mask = fn(y)
        Xb, yb = X[mask], y[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(Xb, yb, test_size=0.20, random_state=42)
        trains[name] = (X_tr, y_tr)
        tests[name] = (X_te, y_te)
    print({k: (len(v[0]), len(tests[k][1])) for k, v in trains.items()}, flush=True)

    out = {"dataset": title, "n_seeds": len(seeds), "crystal_size": crystal_size,
           "n_inject": n_inject, "max_crystal_size": max_crystal_size, "records": []}
    tmp = tempfile.mkdtemp(prefix="shd_lifelong_")
    t0 = time.perf_counter()

    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        X_tr, y_tr = trains["A_lt20"]
        perm = rng.permutation(len(X_tr))
        Xs_A, ys_A = X_tr[perm], y_tr[perm]
        d = SpectralHologramDistiller(
            crystal_size=crystal_size, high_band_ratio=0.26, sketch_dim=256,
            tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
        )
        for i in range(0, len(Xs_A), 2500):
            d.partial_fit(Xs_A[i : i + 2500], ys_A[i : i + 2500])
        d.finalize_crystal()
        path = os.path.join(tmp, f"life_{seed}.hologram.npz")
        d.save_hologram(path)
        loader = HologramCrystalLoader(path)

        stages = [{"stage": "A_only", "n_medoids": int(len(loader.X)),
                   "metrics": fit_eval(loader, tests, X.shape[1])}]
        for shift in ("B_20_40", "C_40_77", "D_77_plus"):
            Xs, ys = trains[shift]
            # Force the injection so the sequence is identical across seeds; the
            # radar verdict is still recorded in bw_before / shift flag.
            pre = loader.check_domain_shift(Xs)
            res = loader.heal_domain_shift(
                Xs, ys, n_inject=n_inject, max_crystal_size=max_crystal_size,
                force=True, random_state=seed,
                new_weight_boost=new_weight_boost,
                passport_blend_min=passport_blend_min,
                passport_blend_max=passport_blend_max,
                protect_old_weight_frac=protect_old_weight_frac,
            )
            stages.append({
                "stage": "heal_" + shift,
                "shifted_before": bool(pre["is_shifted"]),
                "bw_before": float(pre["bw_distance"]),
                "bw_after": float(res.get("bw_after", float("nan"))),
                "n_injected": int(res.get("n_injected", 0)),
                "n_pruned": int(res.get("n_pruned", 0)),
                "n_medoids": int(len(loader.X)),
                "metrics": fit_eval(loader, tests, X.shape[1]),
            })
        rec = {"seed": int(seed), "stages": stages}
        out["records"].append(rec)
        last = stages[-1]
        print(f"seed {seed}: after D medoids={last['n_medoids']} "
              f"bw={last['bw_after']:.3f} "
              f"A_krr={last['metrics']['A_lt20']['krr_rmse']:.2f} "
              f"D_krr={last['metrics']['D_77_plus']['krr_rmse']:.2f}", flush=True)

    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--crystal-size", type=int, default=200)
    ap.add_argument("--n-inject", type=int, default=45)
    ap.add_argument("--max-crystal-size", type=int, default=260)
    ap.add_argument("--new-weight-boost", type=float, default=1.0)
    ap.add_argument("--blend-min", type=float, default=0.15)
    ap.add_argument("--blend-max", type=float, default=0.50)
    ap.add_argument("--protect-old-weight-frac", type=float, default=0.0)
    ap.add_argument("--output", default="bench_lifelong_results.json")
    args = ap.parse_args()
    res = run(
        list(range(args.seeds)),
        crystal_size=args.crystal_size,
        n_inject=args.n_inject,
        max_crystal_size=args.max_crystal_size,
        new_weight_boost=args.new_weight_boost,
        passport_blend_min=args.blend_min,
        passport_blend_max=args.blend_max,
        protect_old_weight_frac=args.protect_old_weight_frac,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for stage in ("A_only", "heal_B_20_40", "heal_C_40_77", "heal_D_77_plus"):
        vals = []
        for r in res["records"]:
            s = next(x for x in r["stages"] if x["stage"] == stage)
            vals.append((s["metrics"], s.get("bw_after", float("nan")), s["n_medoids"]))
        print(f"  {stage:20s} "
              f"A={np.median([v[0]['A_lt20']['krr_rmse'] for v in vals]):6.2f} "
              f"B={np.median([v[0]['B_20_40']['krr_rmse'] for v in vals]):6.2f} "
              f"C={np.median([v[0]['C_40_77']['krr_rmse'] for v in vals]):6.2f} "
              f"D={np.median([v[0]['D_77_plus']['krr_rmse'] for v in vals]):6.2f} "
              f"BW={np.nanmedian([v[1] for v in vals]):.3f} M={np.median([v[2] for v in vals]):.0f}")


if __name__ == "__main__":
    main()

"""Domain bank: one SHD crystal per regime + radar router.

The single-crystal heal learns new domains but forgets old ones. A bank keeps
one crystal per domain, so old domains are untouched by construction. The open
question is routing: can the spectral passport pick the right crystal without
labels?

For each seed:
- build A/B/C/D crystals on their own train splits (M=200 each);
- evaluate each test band with its own crystal (oracle upper bound);
- evaluate each test band with the crystal whose passport has the smallest
  BW distance to that band (radar routing);
- compare with a single A-only crystal.

Usage:
    python bench_bank.py --seeds 5
Writes bench_bank_results.json.
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
from sklearn.preprocessing import StandardScaler

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller

BANDS = {
    "A_lt20": lambda y: y < 20,
    "B_20_40": lambda y: (y >= 20) & (y < 40),
    "C_40_77": lambda y: (y >= 40) & (y < 77),
    "D_77_plus": lambda y: y >= 77,
}


def build_crystal(X_tr, y_tr, seed, path, m=200):
    d = SpectralHologramDistiller(crystal_size=m, sketch_dim=256,
                                  tail_quantile=84.0, phase_kappa=3.2,
                                  random_state=seed)
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i : i + 2500], y_tr[i : i + 2500])
    d.finalize_crystal()
    d.save_hologram(path)
    return HologramCrystalLoader(path)


def fit_model(loader, d_feat):
    Xc, yc, wc = loader.get_training_data()
    scaler = StandardScaler().fit(Xc)
    Zc = scaler.transform(Xc)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (d_feat * 1.1))
    krr.fit(Zc, yc, sample_weight=wc)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42
    )
    gbm.fit(Zc, yc, sample_weight=wc)
    return {"scaler": scaler, "krr": krr, "gbm": gbm}


def eval_model(model, X_te, y_te):
    Z = model["scaler"].transform(X_te)
    return {
        "krr_rmse": float(np.sqrt(mean_squared_error(y_te, model["krr"].predict(Z)))),
        "gbm_rmse": float(np.sqrt(mean_squared_error(y_te, model["gbm"].predict(Z)))),
    }


def run(seeds, m=200, window=256, per_window=False):
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

    out = {"dataset": title, "m": m, "n_seeds": len(seeds),
           "window": window, "per_window": per_window, "records": [],
           "summary": {}, "elapsed_sec": 0.0}
    tmp = tempfile.mkdtemp(prefix="shd_bank_")
    t0 = time.perf_counter()
    for seed in seeds:
        loaders, models = {}, {}
        for name in BANDS:
            X_tr, y_tr = trains[name]
            path = os.path.join(tmp, f"{name}_{seed}.hologram.npz")
            loaders[name] = build_crystal(X_tr, y_tr, seed, path, m)
            models[name] = fit_model(loaders[name], X.shape[1])

        rec = {"seed": int(seed), "oracle": {}, "radar": {}, "single_A": {},
               "routing": {}, "bw": {}, "radar_window": {}, "routing_window": {}}
        for band in BANDS:
            X_te, y_te = tests[band]
            rec["oracle"][band] = eval_model(models[band], X_te, y_te)
            rec["single_A"][band] = eval_model(models["A_lt20"], X_te, y_te)
            bw = {}
            for cand in BANDS:
                bw[cand] = float(loaders[cand].check_domain_shift(X_te)["bw_distance"])
            rec["bw"][band] = bw
            chosen = min(bw, key=bw.get)
            rec["routing"][band] = chosen
            rec["radar"][band] = eval_model(models[chosen], X_te, y_te)

            if per_window:
                pred_k, pred_g, choices = [], [], []
                for start in range(0, len(X_te), window):
                    Xw = X_te[start : start + window]
                    if len(Xw) < 32:
                        continue
                    bw_w = {cand: float(loaders[cand].check_domain_shift(Xw)["bw_distance"])
                            for cand in BANDS}
                    ch = min(bw_w, key=bw_w.get)
                    Z = models[ch]["scaler"].transform(Xw)
                    pred_k.append(models[ch]["krr"].predict(Z))
                    pred_g.append(models[ch]["gbm"].predict(Z))
                    choices.append(ch)
                pk = np.concatenate(pred_k)
                pg = np.concatenate(pred_g)
                yy = y_te[: len(pk)]
                rec["radar_window"][band] = {
                    "krr_rmse": float(np.sqrt(mean_squared_error(yy, pk))),
                    "gbm_rmse": float(np.sqrt(mean_squared_error(yy, pg))),
                }
                rec["routing_window"][band] = choices

        out["records"].append(rec)
        print(f"seed {seed}: routing={rec['routing']} "
              f"oracle_D={rec['oracle']['D_77_plus']['krr_rmse']:.1f} "
              f"radar_D={rec['radar']['D_77_plus']['krr_rmse']:.1f}", flush=True)

    arms = ["oracle", "radar", "single_A"] + (["radar_window"] if per_window else [])
    out["summary"] = {}
    for arm in arms:
        out["summary"][arm] = {}
        for band in BANDS:
            out["summary"][arm][band] = {
                "krr_median": float(np.median([r[arm][band]["krr_rmse"] for r in out["records"]])),
                "gbm_median": float(np.median([r[arm][band]["gbm_rmse"] for r in out["records"]])),
            }
    out["routing_accuracy"] = float(np.mean([
        r["routing"][band] == band for r in out["records"] for band in BANDS
    ]))
    if per_window:
        hits = total = 0
        for r in out["records"]:
            for band, choices in r["routing_window"].items():
                hits += sum(c == band for c in choices)
                total += len(choices)
        out["routing_window_accuracy"] = float(hits / max(total, 1))
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-window", action="store_true")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--output", default="bench_bank_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)), window=args.window, per_window=args.per_window)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    print(f"routing accuracy: {res['routing_accuracy']:.2%}")
    if res.get("per_window"):
        print(f"per-window routing accuracy: {res['routing_window_accuracy']:.2%}")
    for arm in res["summary"]:
        print(f"  {arm:12s} " + " ".join(
            f"{b}: {v['krr_median']:.1f}" for b, v in res["summary"][arm].items()))


if __name__ == "__main__":
    main()

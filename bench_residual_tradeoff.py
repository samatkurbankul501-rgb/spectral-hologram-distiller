"""Residual-trigger tradeoff bench.

Tests the published claim (bench_residual_tradeoff.prev): residual trigger
helps global (-0.35 K) but degrades SC crit (+0.80 K, CI excludes 0). This
bench re-runs the ablation with identical splits (SC + Reactor) across
residual_weight in {0, 0.5, 1.0=default, 2.0} and adds the
critical_first_no_residual preset (rw=0, br=0.65).

Reproduces bench_rff/bench_seeds evaluation harness.
Writes bench_residual_tradeoff_results.json
"""
from __future__ import annotations

import argparse, json, time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import eval_models, load_reactor, load_superconduct
from shd import SpectralHologramDistiller

CONFIGS = {
    "rw_0": dict(residual_weight=0.0, high_band_ratio=0.26),
    "rw_0.5": dict(residual_weight=0.5, high_band_ratio=0.26),
    "rw_1.0_default": dict(residual_weight=1.0, high_band_ratio=0.26),
    "rw_2.0": dict(residual_weight=2.0, high_band_ratio=0.26),
    "critical_first_no_residual": dict(residual_weight=0.0, high_band_ratio=0.65),
}

def run_dataset(name, X_raw, y_raw, crit, n_seeds=10, m=300):
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0/(X_raw.shape[1]*1.1)
    out = {"dataset": name, "crit": float(crit), "n_seeds": n_seeds, "crystal_size": m, "configs":{}, "elapsed_sec":0}
    t0 = time.perf_counter()
    for cfg_name, params in CONFIGS.items():
        recs=[]
        for seed in range(n_seeds):
            d = SpectralHologramDistiller(crystal_size=m, sketch_dim=256, tail_quantile=84.0, phase_kappa=3.2, random_state=seed, **params)
            for i in range(0, len(X_tr), 2500):
                d.partial_fit(X_tr[i:i+2500], y_tr[i:i+2500])
            d.finalize_crystal()
            met = eval_models(d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, scaler, 42, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
            met["rare_captured"]=int(np.sum(d.crystal_Y >= crit))
            recs.append(met)
        out["configs"][cfg_name] = {
            "params": params,
            "records": recs,
            "krr_rmse_crit_median": float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_rmse_crit_iqr": float(np.percentile([r["krr_rmse_crit"] for r in recs],75)-np.percentile([r["krr_rmse_crit"] for r in recs],25)),
            "krr_rmse_all_median": float(np.median([r["krr_rmse_all"] for r in recs])),
            "rare_captured_median": float(np.median([r["rare_captured"] for r in recs])),
        }
        print(f"{name} {cfg_name}: crit={out['configs'][cfg_name]['krr_rmse_crit_median']:.3f} all={out['configs'][cfg_name]['krr_rmse_all_median']:.3f} rare={out['configs'][cfg_name]['rare_captured_median']:.0f}", flush=True)
    out["elapsed_sec"]=time.perf_counter()-t0
    # paired wilcoxon vs default
    base = [r["krr_rmse_crit"] for r in out["configs"]["rw_1.0_default"]["records"]]
    for cfg_name in CONFIGS:
        if cfg_name=="rw_1.0_default": continue
        a = np.array(base); b = np.array([r["krr_rmse_crit"] for r in out["configs"][cfg_name]["records"]])
        wins = int(np.sum(b<a))
        try:
            p = float(wilcoxon(a,b).pvalue)
        except Exception:
            p = float("nan")
        out["configs"][cfg_name]["wilcoxon_vs_default"]={"wins":wins,"n":len(a),"p":p, "median_diff":float(np.median(b-a))}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=10)
    ap.add_argument("--output",default="bench_residual_tradeoff_results.json")
    args=ap.parse_args()
    # Reactor first (fast, local)
    X_r, y_r, crit_r, _ = load_reactor()
    res_r = run_dataset("Reactor", X_r, y_r, crit_r, n_seeds=args.seeds)
    # SC (may fetch; fallback ok)
    try:
        X_s, y_s, crit_s, _ = load_superconduct()
        if X_s.shape[1]==81:
            res_s = run_dataset("SC", X_s, y_s, crit_s, n_seeds=args.seeds)
        else:
            res_s = {"skip":"data not 81d"}
    except Exception as e:
        res_s={"error":str(e)}
    out={"reactor":res_r,"SC":res_s}
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=1)
    print(f"Saved -> {args.output}")
    print("Reactor crit medians:", {k:v["krr_rmse_crit_median"] for k,v in res_r["configs"].items()})
    if "configs" in res_s:
        print("SC crit medians:", {k:v["krr_rmse_crit_median"] for k,v in res_s["configs"].items()})

if __name__=="__main__":
    main()

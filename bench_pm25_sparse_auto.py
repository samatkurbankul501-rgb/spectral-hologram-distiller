"""PM2.5 sparse_auto bench.

Compares published default (tail_quantile=84, br=0.26, guard=False) vs
sparse_auto preset (tail_quantile=97, global_guard=True, br=0.26) vs
tail_quantile=97 alone on Beijing PM2.5 3% critical regime.

Reuses bench_pm25_rehab harness shape.
Writes bench_pm25_sparse_auto_results.json
"""
from __future__ import annotations

import argparse, json, time, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

from shd import SpectralHologramDistiller, apply_preset

CONFIGS = {
    "default_q84": dict(tail_quantile=84.0, high_band_ratio=0.26, global_guard=False),
    "tq97_only": dict(tail_quantile=97.0, high_band_ratio=0.26, global_guard=False),
    "sparse_auto": dict(tail_quantile=97.0, high_band_ratio=0.26, global_guard=True),
}

def load_pm25():
    # Reuse bench_pm25 loader if available
    try:
        from bench_pm25 import load_pm25 as lm
        return lm()
    except Exception:
        pass
    try:
        from bench_pm25_hybrid import load_pm25 as lm2
        return lm2()
    except Exception:
        pass
    raise RuntimeError("PM2.5 loader not found (needs KMZ cache)")

def run(seeds, m=300, batch=2500):
    out = {"dataset":"Beijing PM2.5", "n_seeds":len(seeds), "crystal_size":m, "configs":{}} 
    t0=time.perf_counter()
    res = load_pm25()
    if isinstance(res, tuple):
        X_raw, y_raw = res[0], res[1]
    else:
        X_raw, y_raw = res
    # temporal split like bench_pm25 (time-ordered)
    n = len(X_raw); split = int(n*0.8)
    X_tr, y_tr = X_raw[:split], y_raw[:split]
    X_te, y_te = X_raw[split:], y_raw[split:]
    # Critical threshold from the TRAIN split only: the full series (y_raw)
    # would leak the test distribution into the regime definition.
    crit = float(np.percentile(y_tr, 97.0))
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_te, Z_te_crit = scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0/(X_raw.shape[1]*1.1)
    from bench_seeds import eval_models
    for name, params in CONFIGS.items():
        recs=[]
        for seed in seeds:
            d = SpectralHologramDistiller(crystal_size=m, sketch_dim=256, phase_kappa=3.2, random_state=seed, **params)
            for i in range(0,len(X_tr),batch):
                d.partial_fit(X_tr[i:i+batch], y_tr[i:i+batch])
            d.finalize_crystal()
            met = eval_models(d.crystal_X, np.asarray(d.crystal_Y,dtype=np.float64), d.gamma_weights, scaler, 42, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
            met["rare_captured"]=int(np.sum(d.crystal_Y>=crit))
            recs.append(met)
        out["configs"][name]={
            "params":params,
            "records":recs,
            "krr_rmse_crit_median":float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_rmse_all_median":float(np.median([r["krr_rmse_all"] for r in recs])),
            "rare_captured_median":float(np.median([r["rare_captured"] for r in recs])),
        }
        print(f"{name}: crit={out['configs'][name]['krr_rmse_crit_median']:.3f} all={out['configs'][name]['krr_rmse_all_median']:.3f}", flush=True)
    for name in CONFIGS:
        if name=="default_q84": continue
        a=np.array([r["krr_rmse_crit"] for r in out["configs"]["default_q84"]["records"]]); b=np.array([r["krr_rmse_crit"] for r in out["configs"][name]["records"]])
        wins=int(np.sum(b<a))
        try: p=float(wilcoxon(a,b).pvalue)
        except Exception: p=float("nan")
        out["configs"][name]["wilcoxon_vs_default"]={"wins":wins,"n":len(a),"p":p}
    out["elapsed_sec"]=time.perf_counter()-t0
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=10)
    ap.add_argument("--output",default="bench_pm25_sparse_auto_results.json")
    args=ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=1)
    print(f"Saved -> {args.output}")
    for k,v in res["configs"].items():
        print(k, v["krr_rmse_crit_median"], v["krr_rmse_all_median"], v.get("wilcoxon_vs_default"))

if __name__=="__main__":
    main()

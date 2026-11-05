"""Radar hard benchmark: gradual/noisy/partial drift.

AUC=1.0 on the abrupt benchmark (train<20K vs test 40-77/77+K) is an
easy-task characterization. This bench adds three harder regimes:

- gradual: interpolate features between in-distribution and OOD (alpha=0.3/0.5/0.7)
- noisy: add Gaussian noise sigma=0.5/1.0/2.0 to OOD windows
- partial: window contains only 25%/50% OOD rows (rest in-distribution)

Reports AUC + TPR/FPR at published tau + TPR@5%FPR per regime.
Writes bench_radar_hard_results.json
"""
from __future__ import annotations

import argparse, json, os, tempfile, time
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller

def windows(X,B):
    n=(len(X)//B)*B
    if n==0: return []
    return [X[i:i+B] for i in range(0,n,B)]

def score_windows(loader,X,B):
    return np.array([loader.check_domain_shift(w)["bw_distance"] for w in windows(X,B)])

def mix_windows(X_in, X_out, B, frac, rng):
    # partial windows: frac* B rows from OOD, rest in-distribution
    n_out = int(frac*B)
    n_in = B - n_out
    out=[]
    nb = min(len(X_in)//B, len(X_out)//B)
    for i in range(nb):
        w_in = X_in[i*B:(i+1)*B][:n_in]
        w_out = X_out[i*B:(i+1)*B][:n_out]
        w = np.concatenate([w_in, w_out], axis=0)
        rng.shuffle(w)
        out.append(w)
    return out

def run(seeds, crystal_size=200, train_cut=20):
    X,y,_,title = load_superconduct()
    if X.shape[1]!=81:
        raise RuntimeError("UCI fetch failed; aborting")
    in_mask = y < train_cut
    X_in, y_in = X[in_mask], y[in_mask]
    X_tr_in, X_te_in, _, _ = train_test_split(X_in, y_in, test_size=0.20, random_state=42)
    # Graded OOD: use hardest band 20-30 as base OOD source
    X_out = X[(y>=20)&(y<30)]
    if len(X_out)<1000:
        X_out = X[y>=train_cut]
    X_out = X_out[np.random.default_rng(0).permutation(len(X_out))]
    regimes = {
        "abrupt_control": X_out,
        "gradual_alpha0.3": None,
        "gradual_alpha0.5": None,
        "gradual_alpha0.7": None,
        "noisy_sigma0.5": None,
        "noisy_sigma1.0": None,
        "noisy_sigma2.0": None,
        "partial_frac0.25": "partial_0.25",
        "partial_frac0.5": "partial_0.5",
    }
    out={"dataset":title,"n_seeds":len(seeds),"crystal_size":crystal_size,"train_cut":train_cut,"by_regime":{},"by_batch":{}}
    tmp=tempfile.mkdtemp(prefix="shd_hard_")
    t0=time.perf_counter()
    for B in (64,256):
        tau = 0.085*max(1.0, np.sqrt(1024.0/max(B,16)))
        per_seed=[]
        for seed in seeds:
            rng=np.random.default_rng(1000+seed)
            Xs=X_tr_in[rng.permutation(len(X_tr_in))]
            d=SpectralHologramDistiller(crystal_size=crystal_size, high_band_ratio=0.26, sketch_dim=256, tail_quantile=84.0, phase_kappa=3.2, random_state=seed)
            for i in range(0,len(Xs),2500):
                d.partial_fit(Xs[i:i+2500], None)
            d.finalize_crystal()
            path=os.path.join(tmp,f"hard_{seed}.hologram.npz"); d.save_hologram(path); loader=HologramCrystalLoader(path)
            s_in=score_windows(loader, X_te_in, B)
            row={"seed":int(seed),"regimes":{}}
            # gradual
            for alpha in (0.3,0.5,0.7):
                n=min(len(X_te_in),len(X_out)); m=(n//B)*B
                Xg=(1-alpha)*X_te_in[:m]+alpha*X_out[:m]
                s_out=score_windows(loader, Xg, B)
                if len(s_out)==0: continue
                labels=np.concatenate([np.zeros(len(s_in)),np.ones(len(s_out))]); scores=np.concatenate([s_in,s_out])
                auc=float(roc_auc_score(labels,scores)); fpr,tpr,_=roc_curve(labels,scores); idx=max(0,min(int(np.searchsorted(fpr,0.05,side="right")-1),len(tpr)-1))
                row["regimes"][f"gradual_alpha{alpha}"]={"auc":auc,"tpr_at_fpr05":float(tpr[idx]),"fpr_published":float(np.mean(s_in>tau)),"tpr_published":float(np.mean(s_out>tau)),"bw_in_median":float(np.median(s_in)),"bw_out_median":float(np.median(s_out))}
            # noisy
            for sigma in (0.5,1.0,2.0):
                n=min(len(X_te_in),len(X_out)); m=(n//B)*B
                Xn=X_out[:m]+rng.normal(0,sigma,size=(m,X_out.shape[1]))
                s_out=score_windows(loader, Xn, B)
                if len(s_out)==0: continue
                labels=np.concatenate([np.zeros(len(s_in)),np.ones(len(s_out))]); scores=np.concatenate([s_in,s_out])
                auc=float(roc_auc_score(labels,scores)); fpr,tpr,_=roc_curve(labels,scores); idx=max(0,min(int(np.searchsorted(fpr,0.05,side="right")-1),len(tpr)-1))
                row["regimes"][f"noisy_sigma{sigma}"]={"auc":auc,"tpr_at_fpr05":float(tpr[idx]),"fpr_published":float(np.mean(s_in>tau)),"tpr_published":float(np.mean(s_out>tau)),"bw_in_median":float(np.median(s_in)),"bw_out_median":float(np.median(s_out))}
            # partial
            for frac in (0.25,0.5):
                ws=mix_windows(X_te_in, X_out, B, frac, rng)
                s_out=np.array([loader.check_domain_shift(w)["bw_distance"] for w in ws])
                if len(s_out)==0: continue
                labels=np.concatenate([np.zeros(len(s_in)),np.ones(len(s_out))]); scores=np.concatenate([s_in,s_out])
                auc=float(roc_auc_score(labels,scores)); fpr,tpr,_=roc_curve(labels,scores); idx=max(0,min(int(np.searchsorted(fpr,0.05,side="right")-1),len(tpr)-1))
                row["regimes"][f"partial_frac{frac}"]={"auc":auc,"tpr_at_fpr05":float(tpr[idx]),"fpr_published":float(np.mean(s_in>tau)),"tpr_published":float(np.mean(s_out>tau)),"bw_in_median":float(np.median(s_in)),"bw_out_median":float(np.median(s_out))}
            # abrupt control
            s_out=score_windows(loader, X_out, B)
            if len(s_out)>0:
                labels=np.concatenate([np.zeros(len(s_in)),np.ones(len(s_out))]); scores=np.concatenate([s_in,s_out])
                auc=float(roc_auc_score(labels,scores)); fpr,tpr,_=roc_curve(labels,scores); idx=max(0,min(int(np.searchsorted(fpr,0.05,side="right")-1),len(tpr)-1))
                row["regimes"]["abrupt_control"]={"auc":auc,"tpr_at_fpr05":float(tpr[idx]),"fpr_published":float(np.mean(s_in>tau)),"tpr_published":float(np.mean(s_out>tau)),"bw_in_median":float(np.median(s_in)),"bw_out_median":float(np.median(s_out))}
            per_seed.append(row)
            print(f"[B={B}] seed {seed}: "+ " ".join(f"{k}={v['auc']:.3f}" for k,v in row['regimes'].items()), flush=True)
        agg={}
        for name in per_seed[0]["regimes"]:
            vals=[r["regimes"][name] for r in per_seed]
            agg[name]={"auc_median":float(np.median([v["auc"] for v in vals])),"auc_iqr":float(np.percentile([v["auc"] for v in vals],75)-np.percentile([v["auc"] for v in vals],25)),"tpr_at_fpr05_median":float(np.median([v["tpr_at_fpr05"] for v in vals])),"tpr_published_median":float(np.median([v["tpr_published"] for v in vals])),"fpr_published_median":float(np.median([v["fpr_published"] for v in vals]))}
        out["by_batch"][str(B)]={"per_seed":per_seed,"agg":agg}
    out["elapsed_sec"]=time.perf_counter()-t0
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--output",default="bench_radar_hard_results.json")
    args=ap.parse_args()
    res=run(list(range(args.seeds)))
    with open(args.output,"w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=1)
    print(f"Saved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for B,v in res["by_batch"].items():
        print(f"B={B}")
        for k,av in v["agg"].items():
            print(f"  {k:22s} AUC={av['auc_median']:.4f} TPR@5%={av['tpr_at_fpr05_median']:.3f} TPR@tau={av['tpr_published_median']:.3f} FPR@tau={av['fpr_published_median']:.3f}")

if __name__=="__main__":
    main()

"""Suppressor fix: rebuild_at_finalize recovers critical-band allocation.

The default lattice_rebuild_interval=0 builds the lattice on the warm-up
window and only nudges centres online; it cannot create critical cells for
rows arriving after warm-up. A single finalize-time rebuild from the
whole-stream reservoir (rebuild_at_finalize=True) recovers that allocation.

Measures default vs rebuild_at_finalize on reactor + superconductivity
(5 seeds, paired), critical/global KRR RMSE + rare capture, paired Wilcoxon.
Default path is bit-for-bit unchanged (rebuild_at_finalize=False).
Writes bench_rebuild_at_finalize_results.json.

Usage: python bench_rebuild_at_finalize.py [--seeds 5]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from bench_seeds import load_reactor, load_superconduct
from shd import SpectralHologramDistiller


def eval_crit(Xc, yc, wc, scaler, Z_te, y_te, cm, gamma):
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=gamma).fit(
        scaler.transform(Xc), yc, sample_weight=wc)
    p = krr.predict(Z_te)
    return {"krr_rmse_crit": float(np.sqrt(mean_squared_error(y_te[cm], p[cm]))),
            "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, p))),
            "rare_captured": int(np.sum(yc >= cm_threshold))}


cm_threshold = None  # set per dataset


def run(seeds):
    global cm_threshold
    out = {"n_seeds": len(seeds), "datasets": {}}
    loaders = {"reactor": load_reactor, "superconduct": load_superconduct}
    for name, loader in loaders.items():
        X, y, crit, title = loader()
        cm_threshold = crit
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        cm = y_te >= crit
        scaler = StandardScaler().fit(X_tr)
        Z_te = scaler.transform(X_te)
        gamma = 1.0 / (X_tr.shape[1] * 1.1)
        recs = []
        t0 = time.perf_counter()
        for seed in seeds:
            row = {"seed": int(seed)}
            for arm, kw in [("default", {}), ("rebuild_at_finalize", {"rebuild_at_finalize": True})]:
                d = SpectralHologramDistiller(crystal_size=300, high_band_ratio=0.26,
                                              sketch_dim=256, tail_quantile=84.0,
                                              phase_kappa=3.2, random_state=seed, **kw)
                for i in range(0, len(X_tr), 2500):
                    d.partial_fit(X_tr[i:i+2500], y_tr[i:i+2500])
                d.finalize_crystal()
                row[arm] = eval_crit(d.crystal_X, d.crystal_Y, d.gamma_weights,
                                     scaler, Z_te, y_te, cm, gamma)
            recs.append(row)
        summ = {}
        for arm in ["default", "rebuild_at_finalize"]:
            summ[arm] = {m: float(np.median([r[arm][m] for r in recs]))
                         for m in ["krr_rmse_crit", "krr_rmse_all", "rare_captured"]}
        wil = {}
        for m in ["krr_rmse_crit", "krr_rmse_all"]:
            a = np.array([r["rebuild_at_finalize"][m] for r in recs])
            b = np.array([r["default"][m] for r in recs])
            diff = a - b
            stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
            wil[m] = {"wins_rebuild": int(np.sum(diff < 0)), "p_value": float(p),
                      "median_default": float(np.median(b)),
                      "median_rebuild": float(np.median(a))}
        out["datasets"][name] = {"title": title, "crit_threshold": float(crit),
                                 "records": recs, "summary": summ, "wilcoxon": wil,
                                 "elapsed_sec": time.perf_counter() - t0}
        print(f"[{name}] default crit={summ['default']['krr_rmse_crit']:.4f} "
              f"rebuild crit={summ['rebuild_at_finalize']['krr_rmse_crit']:.4f} "
              f"rare {summ['default']['rare_captured']:.0f}->{summ['rebuild_at_finalize']['rare_captured']:.0f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="bench_rebuild_at_finalize_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
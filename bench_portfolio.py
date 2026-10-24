"""Equal-budget portfolio benchmark: SHD-C vs KT vs SKT vs CoresetPortfolio.

Arms per seed (KRR head unless noted):
- shd:        SHD-C critical crystal, m=300 (the distiller's own coreset);
- kt:         unsupervised Kernel Thinning (JMLR 2024), m=300;
- skt:        supervised Kernel Thinning (NeurIPS 2024), m=300;
- portfolio:  CoresetPortfolio on SHD-C(seed) + KT(seed+5000), 300+300;
- hard_gate:  reference routing (pred_shd >= threshold -> SHD else KT), the
              `bench_moe_kt.py` scheme, for the soft-vs-hard comparison.

The portfolio holds 600 coreset points vs 300 for single arms; this
asymmetry is the point of the method (a second expert is bought with cores,
not with new data) and is stated in the README. Every arm shares the same
scaler, KRR hyper-parameters, test split and critical mask.

Usage:
    python bench_portfolio.py --dataset pm25 --seeds 20
    python bench_portfolio.py --dataset superconduct --seeds 20
Writes bench_portfolio_{dataset}_results.json, checkpointed per seed
(resumable: finished seeds are skipped on re-run).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from bench_pm25 import load_pm25
from bench_seeds import build_kt, build_shd, build_skt, load_reactor, load_superconduct
from shd import CoresetPortfolio, SpectralHologramDistiller
from sklearn.model_selection import train_test_split

M = 300
BATCH = 2500


def load_data(name):
    if name == "pm25":
        X, y, title = load_pm25()
        n_tr = int(round(0.80 * len(X)))
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]
        crit = float(np.percentile(y_tr, 97.0))
    elif name == "reactor":
        X, y, crit, title = load_reactor()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        crit = float(crit)
    else:
        X, y, crit, title = load_superconduct()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
        crit = float(crit)
    return X_tr, y_tr, X_te, y_te, crit, title


def _krr_gbm(Xc, yc, wc, scaler, gamma, d_feat, X_te, Z_te, y_te, crit_mask, model_seed):
    Zc = scaler.transform(Xc)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=gamma).fit(
        Zc, yc, sample_weight=wc)
    p = krr.predict(Z_te)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5,
        random_state=model_seed).fit(Xc, yc, sample_weight=wc)
    pg = gbm.predict(X_te)
    return {
        "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, p))),
        "krr_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit_mask], p[crit_mask]))),
        "krr_r2": float(r2_score(y_te, p)),
        "gbm_rmse_all": float(np.sqrt(mean_squared_error(y_te, pg))),
        "gbm_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit_mask], pg[crit_mask]))),
    }


def _load_checkpoint(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def run(dataset, seeds, checkpoint_path):
    X_tr, y_tr, X_te, y_te, crit, title = load_data(dataset)
    crit_mask = y_te >= crit
    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    d_feat = X_tr.shape[1]
    gamma = 1.0 / (d_feat * 1.1)

    out = _load_checkpoint(checkpoint_path) or {
        "dataset": dataset, "title": title, "crit_threshold": crit,
        "m_shd": M, "m_kt": M, "n_test_crit": int(np.sum(crit_mask)),
        "n_seeds": len(seeds), "seeds": list(seeds), "records": [],
        "summary": {}, "wilcoxon": {}, "elapsed_sec": 0.0, "partial": True,
    }
    done = {r["seed"] for r in out["records"]}
    t0 = time.perf_counter()
    for seed in seeds:
        if seed in done:
            print(f"[{dataset}] seed {seed}: cached", flush=True)
            continue
        t_s = time.perf_counter()
        rec = {"seed": int(seed)}

        Xs, ys, ws, n_rare = build_shd(X_tr, y_tr, seed, M, BATCH, crit)
        rec["shd"] = _krr_gbm(Xs, ys, ws, scaler, gamma, d_feat,
                              X_te, scaler.transform(X_te), y_te,
                              crit_mask, 42)
        rec["shd"]["rare_captured"] = n_rare

        Xk, yk, wk = build_kt(X_tr, y_tr, Z_tr, seed, M)
        wk = np.ones(len(yk)) if wk is None else wk
        rec["kt"] = _krr_gbm(Xk, yk, wk, scaler, gamma, d_feat,
                             X_te, scaler.transform(X_te), y_te, crit_mask, 42)
        rec["kt"]["rare_captured"] = int(np.sum(yk >= crit))

        Xw, yw, ww = build_skt(X_tr, y_tr, Z_tr, seed, M)
        ww = np.ones(len(yw)) if ww is None else ww
        rec["skt"] = _krr_gbm(Xw, yw, ww, scaler, gamma, d_feat,
                              X_te, scaler.transform(X_te), y_te, crit_mask, 42)
        rec["skt"]["rare_captured"] = int(np.sum(yw >= crit))

        # Portfolio: critical SHD-C expert + global KT expert (fresh cores).
        pf = CoresetPortfolio(critical_threshold=crit).fit(
            Xs, ys, ws, Xk, yk, wk, scaler, rbf_gamma=gamma, model_seed=42)
        p_pf = pf.predict(X_te)
        gbm_pf = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=20, min_samples_leaf=5,
            random_state=42).fit(Xs, ys, sample_weight=ws)
        pg_pf = gbm_pf.predict(X_te)
        w = pf.gate_weights(X_te)
        rec["portfolio"] = {
            "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, p_pf))),
            "krr_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit_mask], p_pf[crit_mask]))),
            "krr_r2": float(r2_score(y_te, p_pf)),
            "gbm_rmse_all": float(np.sqrt(mean_squared_error(y_te, pg_pf))),
            "gbm_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit_mask], pg_pf[crit_mask]))),
            "gate_mean_all": float(np.mean(w)),
            "gate_mean_crit": float(np.mean(w[crit_mask])),
        }

        # Hard-gate reference: same two experts, argmax routing (no OOF gate).
        Zs, Zk2 = scaler.transform(Xs), scaler.transform(Xk)
        krr_s = KernelRidge(kernel="rbf", alpha=0.08, gamma=gamma).fit(Zs, ys, sample_weight=ws)
        krr_k = KernelRidge(kernel="rbf", alpha=0.08, gamma=gamma).fit(Zk2, yk)
        p_s, p_k = krr_s.predict(scaler.transform(X_te)), krr_k.predict(scaler.transform(X_te))
        g = p_s >= crit
        p_h = np.where(g, p_s, p_k)
        rec["hard_gate"] = {
            "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, p_h))),
            "krr_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit_mask], p_h[crit_mask]))),
            "krr_r2": float(r2_score(y_te, p_h)),
            "gate_fraction": float(np.mean(g)),
        }

        out["records"].append(rec)
        out["elapsed_sec"] = time.perf_counter() - t0 + out.get("elapsed_sec", 0.0)
        _save(checkpoint_path, out)
        print(f"[{dataset}] seed {seed}: shd={rec['shd']['krr_rmse_crit']:.2f} "
              f"kt={rec['kt']['krr_rmse_crit']:.2f} skt={rec['skt']['krr_rmse_crit']:.2f} "
              f"pf={rec['portfolio']['krr_rmse_crit']:.2f} "
              f"(crit/all pf {rec['portfolio']['krr_rmse_all']:.2f} "
              f"shd {rec['shd']['krr_rmse_all']:.2f} kt {rec['kt']['krr_rmse_all']:.2f}) "
              f"({time.perf_counter() - t_s:.1f}s)", flush=True)

    arms = [k for k in out["records"][0] if k != "seed"]
    for arm in arms:
        out["summary"][arm] = {
            key: float(np.median([rec[arm][key] for rec in out["records"]]))
            for key in out["records"][0][arm] if key != "rare_captured"
        }
    for other in ("shd", "kt", "skt", "hard_gate"):
        for metric in ("krr_rmse_all", "krr_rmse_crit"):
            a = np.array([r["portfolio"][metric] for r in out["records"]])
            b = np.array([r[other][metric] for r in out["records"]])
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(a, b)
            out["wilcoxon"][f"portfolio_vs_{other}_{metric}"] = {
                "median_portfolio": float(np.median(a)),
                "median_" + other: float(np.median(b)),
                "p": float(p),
            }
    out["partial"] = False
    out["elapsed_sec"] = time.perf_counter() - t0 + out.get("elapsed_sec", 0.0)
    _save(checkpoint_path, out)
    return out


def _save(path, out):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("pm25", "superconduct", "reactor"), default="pm25")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    path = f"bench_portfolio_{args.dataset}_results.json"
    res = run(args.dataset, list(range(args.seeds)), path)
    print(f"\n{res['title']}  crit_threshold={res['crit_threshold']:.2f}  "
          f"n_test_crit={res['n_test_crit']}  ({res['elapsed_sec']:.0f}s)")
    for arm in ("shd", "kt", "skt", "portfolio", "hard_gate"):
        v = res["summary"][arm]
        gbm = v.get("gbm_rmse_crit")
        gbm_s = f"  gbm_crit={gbm:8.2f}" if gbm is not None else ""
        print(f"  {arm:10s} crit={v['krr_rmse_crit']:8.2f}  all={v['krr_rmse_all']:8.2f}  "
              f"r2={v['krr_r2']:.3f}{gbm_s}")
    for k, v in res["wilcoxon"].items():
        other = k.replace("portfolio_vs_", "").split("_krr_rmse")[0]
        print(f"  {k}: pf {v['median_portfolio']:.2f} vs "
              f"{v['median_' + other]:.2f}  p={v['p']:.4f}")


if __name__ == "__main__":
    main()

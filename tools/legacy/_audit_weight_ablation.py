# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
"""Ablation the audit does not have: SHD crystal with uniform weights vs gamma weights.

Question: is the critical-regime gain driven by the *selection* (phase-band Voronoi
medoids) or by the *importance weights* handed to KRR/GBM?  Same crystal, same
split, only w changes.
"""
import sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(REPO_PATH))
from bench_seeds import load_reactor, load_superconduct, build_shd, build_random, eval_models

def run(key, seeds, m=300, batch=4000):
    if key == "reactor":
        X_raw, y_raw, crit, title = load_reactor()
    else:
        X_raw, y_raw, crit, title = load_superconduct()
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X_raw.shape[1] * 1.1)

    rows = []
    for seed in seeds:
        Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, m, batch, crit)
        uni = np.ones_like(wc)
        mw = eval_models(Xc, yc, wc, scaler, 42, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
        mu = eval_models(Xc, yc, uni, scaler, 42, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
        Xr, yr, wr = build_random(X_tr, y_tr, seed, m)
        mr = eval_models(Xr, yr, None, scaler, 42, X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, gamma)
        rows.append((mw, mu, mr, n_rare, int(np.sum(yr >= crit))))
        print(f"{key} seed {seed}: weighted crit={mw['krr_rmse_crit']:.2f} uniform={mu['krr_rmse_crit']:.2f} random={mr['krr_rmse_crit']:.2f}", flush=True)

    def med(vals):
        return float(np.median(vals))
    print(f"\n== {key} ({len(seeds)} seeds, M={m}) ==")
    print(f"KRR crit: weighted={med([r[0]['krr_rmse_crit'] for r in rows]):.2f} "
          f"uniform={med([r[1]['krr_rmse_crit'] for r in rows]):.2f} "
          f"random={med([r[2]['krr_rmse_crit'] for r in rows]):.2f}")
    print(f"KRR all : weighted={med([r[0]['krr_rmse_all'] for r in rows]):.2f} "
          f"uniform={med([r[1]['krr_rmse_all'] for r in rows]):.2f} "
          f"random={med([r[2]['krr_rmse_all'] for r in rows]):.2f}")
    print(f"GBM crit: weighted={med([r[0]['gbm_rmse_crit'] for r in rows]):.2f} "
          f"uniform={med([r[1]['gbm_rmse_crit'] for r in rows]):.2f} "
          f"random={med([r[2]['gbm_rmse_crit'] for r in rows]):.2f}")
    print(f"GBM all : weighted={med([r[0]['gbm_rmse_all'] for r in rows]):.2f} "
          f"uniform={med([r[1]['gbm_rmse_all'] for r in rows]):.2f} "
          f"random={med([r[2]['gbm_rmse_all'] for r in rows]):.2f}")
    print(f"rare in crystal: shd={med([r[3] for r in rows]):.1f} random={med([r[4] for r in rows]):.1f}")
    # paired deltas
    for name, a, b in [("uniform-vs-weighted", [r[1]["krr_rmse_crit"] for r in rows], [r[0]["krr_rmse_crit"] for r in rows]),
                       ("uniform-vs-random", [r[1]["krr_rmse_crit"] for r in rows], [r[2]["krr_rmse_crit"] for r in rows])]:
        d = np.array(a) - np.array(b)
        print(f"  {name}: median diff={np.median(d):+.2f}, better={int(np.sum(d < 0))}/{len(d)}")

run("reactor", range(5))
run("superconduct", range(3))

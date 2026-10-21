# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

root = REPO_PATH
for name in ["bench_weights_superconduct_results.json", "bench_weights_reactor_results.json"]:
    p = root / name
    d = json.loads(p.read_text(encoding="utf-8"))
    rec = d["records"]
    tests = {}
    for metric in ["krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"]:
        a = np.array([r["uniform"][metric] for r in rec], dtype=float)
        b = np.array([r["gamma"][metric] for r in rec], dtype=float)
        diff = a - b
        stat, pv = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
        higher = metric.endswith("_r2")
        tests[f"uniform_vs_gamma_{metric}"] = {
            "wins_uniform": int(np.sum(diff > 0)) if higher else int(np.sum(diff < 0)),
            "wins_gamma": int(np.sum(diff < 0)) if higher else int(np.sum(diff > 0)),
            "statistic": float(stat),
            "p_value": float(pv),
        }
    d["wilcoxon"] = tests
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(name, "updated")

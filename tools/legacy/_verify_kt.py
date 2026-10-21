# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
"""Post-KT verification: strict README-vs-JSON (reactor), regression vs backup,
rare-capture comparison across arms."""
import json

import numpy as np

ROOT = str(REPO_PATH)
cur = json.load(open(ROOT + r"\bench_seeds_results.json", encoding="utf-8"))
bak = json.load(open(ROOT + r"\bench_seeds_results.v38_pre_kt.json", encoding="utf-8"))

ok = True

# ---- A. reactor: strict README row check vs JSON wilcoxon ----
print("== A. reactor README rows vs JSON ==")
W = cur["results"]["reactor"]["wilcoxon"]
# README table 2 claims (median / IQR) per arm per metric
CLAIMS = {
    "random": {"krr_rmse_all": (1.134, 0.104), "krr_rmse_crit": (4.424, 0.814), "krr_r2": (0.621, 0.070),
               "gbm_rmse_all": (0.711, 0.081), "gbm_rmse_crit": (2.409, 0.549)},
    "kmeans": {"krr_rmse_all": (1.069, 0.073), "krr_rmse_crit": (4.390, 0.722), "krr_r2": (0.663, 0.046),
               "gbm_rmse_all": (0.596, 0.079), "gbm_rmse_crit": (2.033, 0.572)},
    "kt": {"krr_rmse_all": (1.140, 0.056), "krr_rmse_crit": (4.534, 0.606), "krr_r2": (0.617, 0.038),
           "gbm_rmse_all": (0.661, 0.106), "gbm_rmse_crit": (2.329, 0.706)},
    "skt": {"krr_rmse_all": (1.047, 0.036), "krr_rmse_crit": (4.119, 0.423), "krr_r2": (0.677, 0.022),
            "gbm_rmse_all": (0.688, 0.125), "gbm_rmse_crit": (2.555, 0.777)},
    "shd": {"krr_rmse_all": (0.976, 0.021), "krr_rmse_crit": (3.436, 0.225), "krr_r2": (0.720, 0.012),
            "gbm_rmse_all": (0.624, 0.092), "gbm_rmse_crit": (1.732, 0.579)},
}
def check_cell(label, readme, actual, tol_rel=0.002):
    global ok
    good = abs(readme - actual) <= tol_rel * max(1.0, abs(actual))
    if not good:
        ok = False
        print(f"  BAD {label}: readme={readme} json={actual:.6g}")
    return good

for arm, claims in CLAIMS.items():
    for metric, (m_rd, i_rd) in claims.items():
        if arm == "shd":
            # shd side lives in every wilcoxon entry; use the random pair entry
            e = W["shd_vs_random_" + metric]
            m_js, i_js = e["median_shd"], e["iqr_shd"]
        else:
            e = W["shd_vs_" + arm + "_" + metric]
            m_js, i_js = e[f"median_{arm}"], e[f"iqr_{arm}"]
        check_cell(f"reactor {arm} {metric} med", m_rd, m_js)
        check_cell(f"reactor {arm} {metric} iqr", i_rd, i_js, tol_rel=0.02)

# wilcoxon rows
WCLAIMS = {
    ("kt", "krr_rmse_all"): (20, 1.9e-6), ("kt", "krr_rmse_crit"): (20, 1.9e-6), ("kt", "krr_r2"): (20, 1.9e-6),
    ("kt", "gbm_rmse_all"): (13, 8.97e-2), ("kt", "gbm_rmse_crit"): (15, 5.58e-3),
    ("skt", "krr_rmse_all"): (19, 3.81e-6), ("skt", "krr_rmse_crit"): (20, 1.9e-6), ("skt", "krr_r2"): (19, 3.81e-6),
    ("skt", "gbm_rmse_all"): (16, 8.51e-4), ("skt", "gbm_rmse_crit"): (18, 1.34e-4),
}
for (arm, metric), (wins_rd, p_rd) in WCLAIMS.items():
    e = W[f"shd_vs_{arm}_{metric}"]
    if e["wins_shd"] != wins_rd:
        ok = False
        print(f"  BAD reactor {arm} {metric} wins: readme={wins_rd} json={e['wins_shd']}")
    if abs(e["p_value"] - p_rd) > 0.05 * p_rd:
        ok = False
        print(f"  BAD reactor {arm} {metric} p: readme={p_rd} json={e['p_value']:.3e}")
print("  reactor cells:", "OK" if ok else "MISMATCHES ABOVE")

# ---- B. reactor regression: new shd/random/kmeans vs backup ----
print("== B. reactor regression vs pre-KT backup ==")
ok_b = True
for arm in ("shd", "random", "kmeans"):
    for metric in ("krr_rmse_all", "krr_rmse_crit", "krr_r2", "gbm_rmse_all", "gbm_rmse_crit", "gbm_r2"):
        a = np.array([r[arm][metric] for r in cur["results"]["reactor"]["records"]])
        b = np.array([r[arm][metric] for r in bak["results"]["reactor"]["records"]])
        if not np.array_equal(a, b):
            ok_b = False
            ok = False
            print(f"  BAD reactor {arm} {metric}: records differ (max|d|={np.max(np.abs(a-b)):.3e})")
print("  reactor old arms:", "BITWISE-IDENTICAL to backup" if ok_b else "REGRESSION!")

# ---- C. superconduct current-vs-backup identity (pre-re-run snapshot) ----
print("== C. superconduct snapshot identity ==")
ok_c = True
sc_cur, sc_bak = cur["results"]["superconduct"], bak["results"]["superconduct"]
for arm in ("shd", "random", "kmeans"):
    for metric in ("krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all", "gbm_rmse_crit"):
        a = np.array([r[arm][metric] for r in sc_cur["records"]])
        b = np.array([r[arm][metric] for r in sc_bak["records"]])
        if not np.array_equal(a, b):
            ok_c = False
            print(f"  BAD superconduct {arm} {metric} differs from backup")
print("  superconduct snapshot:", "IDENTICAL to backup" if ok_c else "DIFFERS")
print(f"  superconduct arms present: {sorted(sc_cur['records'][0].keys())}")
print(f"  reactor arms present: {sorted(cur['results']['reactor']['records'][0].keys())}")

# ---- D. rare-regime capture across arms (reactor) ----
print("== D. rare regime capture (reactor, med/IQR) ==")
for arm in ("shd", "random", "kmeans", "kt", "skt"):
    v = np.array([r[arm]["rare_captured"] for r in cur["results"]["reactor"]["records"]])
    print(f"  {arm:7s}: {np.median(v):.1f} / {np.percentile(v,75)-np.percentile(v,25):.1f}  (of 300)")

print()
print("VERDICT:", "ALL CHECKS PASS" if ok and ok_b else "ISSUES FOUND")

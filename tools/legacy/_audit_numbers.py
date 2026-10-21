# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json, os, statistics as st
p = str(REPO_PATH)

def load(n):
    return json.load(open(os.path.join(p, n), encoding="utf-8"))

def med(xs):
    return float(st.median(xs))

print("############ BENCH_SEEDS ############")
d = load("bench_seeds_results.json")
for ds, blk in d["results"].items():
    print("\n==", ds, "M=", blk["crystal_size"], "seeds=", blk["n_seeds"])
    arms = ["shd", "random", "kmeans", "kt", "skt"]
    for arm in arms:
        rows = [r[arm] for r in blk["records"]]
        print(f"{arm:7s} crit={med([r['krr_rmse_crit'] for r in rows]):7.2f} "
              f"all={med([r['krr_rmse_all'] for r in rows]):7.2f} "
              f"gbm_all={med([r['gbm_rmse_all'] for r in rows]):7.2f} "
              f"gbm_crit={med([r['gbm_rmse_crit'] for r in rows]):7.2f} "
              f"rare={med([r['rare_captured'] for r in rows]):6.1f} "
              f"r2={med([r['krr_r2'] for r in rows]):.3f}")
    for key in ["krr_rmse_crit", "krr_rmse_all", "gbm_rmse_all", "gbm_rmse_crit"]:
        for base in ["random", "kmeans", "kt", "skt"]:
            w = blk["wilcoxon"][f"shd_vs_{base}_{key}"]
            print(f"  {key:14s} vs {base:7s} wins={w['wins_shd']:2d}/20 p={w['p_value']:.2e}")

print("\n############ HEAL ############")
h = load("bench_heal_superconduct_results.json")
for k in ["healed_vs_pre_heal_krr_rmse_domain_B", "healed_vs_naive_overwrite_krr_rmse_domain_A",
          "healed_vs_pre_heal_krr_rmse_domain_A", "healed_vs_naive_overwrite_krr_rmse_domain_B",
          "healed_vs_pre_heal_krr_rmse_combined", "healed_vs_naive_overwrite_krr_rmse_combined",
          "healed_bounded_vs_pre_heal_krr_rmse_combined", "healed_bounded_vs_naive_overwrite_krr_rmse_combined",
          "healed_vs_pre_heal_gbm_rmse_domain_B", "healed_vs_naive_overwrite_gbm_rmse_domain_A",
          "healed_vs_pre_heal_gbm_rmse_combined", "healed_vs_naive_overwrite_gbm_rmse_combined"]:
    v = h["wilcoxon"][k]
    ks = list(v.keys())
    print(f"{k:52s} {v[ks[0]]:8.2f} vs {v[ks[2]]:8.2f} wins={v['wins']}/20 p={v['p_value']:.2e}")
print("summary:", {k: h['summary'][k] for k in ['radar_false_alarms_on_domain_A','radar_detections_on_domain_B','bw_in_median','bw_ood_median','bw_after_heal_median','radar_latency_ms_median','heal_latency_ms_median']})

print("\n############ PM2.5 ############")
pm = load("bench_pm25_results.json")
print("title", pm["title"], "n_train", pm["n_train"], "crit", pm["crit_threshold"], "n_test_crit", pm["n_test_crit"])
for key, v in pm["wilcoxon"].items():
    print(f"{key:46s} {v}")

print("\n############ BRIDGE ############")
b = load("bridge_neon_bo_results.json")
for key, v in b["wilcoxon"].items():
    if isinstance(v, dict):
        print(f"{key:46s} {v}")
    else:
        print(f"{key:46s} {v}")

print("\n############ CRITICAL GUARD ############")
cg = load("bench_critical_guard_results.json")
print("summary:", json.dumps(cg["summary"], indent=1)[:1500])

print("\n############ BANK ############")
bk = load("bench_bank_results.json")
print("routing_accuracy:", bk.get("routing_accuracy"))
print("summary arms:", list(bk["summary"].keys()))
for arm, dct in bk["summary"].items():
    print(arm, {k: round(v["krr_median"], 2) if isinstance(v, dict) else v for k, v in dct.items()})

print("\n############ LIFELONG ############")
for fn in ["bench_lifelong_results.json", "bench_lifelong_retention_results.json", "bench_lifelong_boost_results.json"]:
    lf = load(fn)
    print("--", fn, "M", lf["crystal_size"], "inject", lf["n_inject"], "cap", lf["max_crystal_size"])
    for stage_idx, label in [(1, "heal_B"), (2, "heal_C"), (3, "heal_D")]:
        vals = []
        for r in lf["records"]:
            stg = r["stages"][stage_idx]
            vals.append((stg["metrics"]["A_lt20"]["krr_rmse"], stg["metrics"]["D_77_plus"]["krr_rmse"], stg["n_medoids"]))
        print(f"   after {label}: A={med([v[0] for v in vals]):.2f} D={med([v[1] for v in vals]):.2f} M={med([v[2] for v in vals]):.0f}")
    lastA = med([r["stages"][0]["metrics"]["A_lt20"]["krr_rmse"] for r in lf["records"]])
    print(f"   stage0 A baseline={lastA:.2f}")

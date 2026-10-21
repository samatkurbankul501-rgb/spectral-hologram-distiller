# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json, os
p = str(REPO_PATH)
def L(n): return json.load(open(os.path.join(p,n),encoding="utf-8"))

d=L("bench_seeds_results.json")
for ds in ("superconduct","reactor"):
    print("==",ds)
    print("wilcoxon keys:", list(d["results"][ds]["wilcoxon"].keys()))

print("\nheal summary keys:", list(L("bench_heal_superconduct_results.json")["summary"].keys()))
print("heal wilcoxon keys:", list(L("bench_heal_superconduct_results.json")["wilcoxon"].keys()))
print("\nbridge wilcoxon keys:", list(L("bridge_neon_bo_results.json")["wilcoxon"].keys()))
print("\npm25 wilcoxon keys:", list(L("bench_pm25_results.json")["wilcoxon"].keys()))
print("\ngaps pooled tests keys:", list(L("bench_gaps_results.json")["pooled_summary"]["tests"].keys()))
print("\nweights summaries:", {k: list(L(f"bench_weights_{k}_results.json")["summary"].keys()) for k in ("superconduct","reactor")})
print("\npassport summary keys:", list(L("bench_passport_results.json")["summary"].keys()))
print("\nshdc pm25 keys:", L("bench_shdc_pm25_results.json")["summary"].keys())
print("guard keys:", L("bench_critical_guard300_results.json")["summary"].keys())
print("stats_holm keys:", list(L("stats_holm_results.json").keys()))

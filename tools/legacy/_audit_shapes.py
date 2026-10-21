# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json, os
p = str(REPO_PATH)

def load(n):
    return json.load(open(os.path.join(p, n), encoding="utf-8"))

d = load("bench_seeds_results.json")
print("=== bench_seeds config ===")
print(json.dumps(d["config"], indent=1)[:800])
print("datasets:", list(d["results"].keys()))
for ds, blk in d["results"].items():
    print("--", ds, "keys:", list(blk.keys()))
    arms = blk.get("arms", {})
    print("   arms:", list(arms.keys())[:12])
    first = next(iter(arms.values())) if arms else None
    if isinstance(first, dict):
        print("   arm keys:", list(first.keys())[:12])
    break

print("\n=== heal summary ===")
h = load("bench_heal_superconduct_results.json")
print(json.dumps(h["summary"], indent=1))

print("\n=== bridge structure ===")
b = load("bridge_neon_bo_results.json")
print("record keys:", list(b["records"][0].keys()))
print("wilcoxon keys:", list(b["wilcoxon"].keys())[:20])
print("sample wilcoxon:", json.dumps(next(iter(b["wilcoxon"].items())), indent=1)[:600])

print("\n=== pm25 structure ===")
pm = load("bench_pm25_results.json")
print({k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in pm.items()})
print("summary sample:", json.dumps(pm.get("summary", {}), indent=1)[:800])

print("\n=== gaps structure ===")
g = load("bench_gaps_results.json")
print("keys:", list(g.keys()))
print("pooled_summary:", json.dumps(g["pooled_summary"], indent=1)[:1200])

print("\n=== bank structure ===")
bk = load("bench_bank_results.json")
print("summary:", json.dumps(bk["summary"], indent=1)[:1000])

print("\n=== lifelong record keys ===")
lf = load("bench_lifelong_results.json")
print(list(lf["records"][0].keys()))
print(json.dumps(lf["records"][0], indent=1)[:900])

print("\n=== radar roc ===")
r = load("bench_radar_roc_results.json")
print("by_cut keys:", list(r["by_cut"].keys()))
print(json.dumps(next(iter(r["by_cut"].items())), indent=1)[:700])

print("\n=== residual ablation ===")
ra = load("bench_residual_ablation.json")
print(json.dumps(ra, indent=1)[:900])

print("\n=== stats_holm ===")
sh = load("stats_holm_results.json")
print(json.dumps(sh, indent=1)[:2000])

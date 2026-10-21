# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json, os, numpy as np
p = str(REPO_PATH)
def L(n): return json.load(open(os.path.join(p, n), encoding="utf-8"))

for fn in ["bench_shdc_superconduct_results.json", "bench_shdc_pm25_results.json"]:
    d = L(fn)
    print("==", fn)
    print("summary:", json.dumps(d.get("summary"), indent=1)[:700])

for fn in ["bench_critical_guard100_results.json", "bench_critical_guard300_results.json"]:
    d = L(fn)
    print("==", fn, "summary:")
    print(json.dumps(d.get("summary"), indent=1)[:700])

d = L("bench_moe_kt_results.json")
print("== moe_kt summary:", json.dumps(d.get("summary"), indent=1)[:900])

d = L("bench_bank_results.json")
print("== bank routing:", d.get("routing_accuracy"))

d = L("bench_lifelong_results.json")
print("== lifelong stages:")
for i, st in enumerate(d["records"][0]["stages"]):
    print(i, st["stage"], "M=", st["n_medoids"])

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

d = load("bench_seeds_results.json")
for ds, blk in d["results"].items():
    print("=" * 20, ds)
    print("title:", blk.get("title"), "| M:", blk.get("crystal_size"), "| seeds:", blk.get("n_seeds"))
    print("oracle:", json.dumps(blk.get("oracle"), indent=1))
    print("wilcoxon keys:")
    print(list(blk["wilcoxon"].keys()))
    rec = blk["records"][0]
    print("record keys:", list(rec.keys()))
    print("sample record arm keys:", {k: list(v.keys()) for k, v in rec.items() if isinstance(v, dict)})

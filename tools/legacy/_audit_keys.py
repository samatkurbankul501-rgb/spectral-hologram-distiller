# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import json, os, glob
p = str(REPO_PATH)
for f in sorted(glob.glob(os.path.join(p, "*.json"))):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print(os.path.basename(f), "ERR", e)
        continue
    name = os.path.basename(f)
    if isinstance(d, dict):
        print(f"{name:46s} {os.path.getsize(f)//1024:5d} KB keys={list(d.keys())[:9]}")
    else:
        print(name, type(d), len(d))

# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
import re
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")
begins = re.findall(r"\\begin\{([^}]+)\}", t)
ends = re.findall(r"\\end\{([^}]+)\}", t)
print("begin", len(begins), "end", len(ends))
from collections import Counter
print("begin", Counter(begins))
print("end", Counter(ends))
print("braces", t.count("{"), t.count("}"), "dollars", t.count("$"))
for env in set(begins) | set(ends):
    if begins.count(env) != ends.count(env):
        print("UNBALANCED", env, begins.count(env), ends.count(env))

# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")
reps = [
    (r"the phase-manifold geometry, not stratification alone, is doing the work.",
     r"the phase-manifold geometry, not stratification alone, is doing the work (artifact: \texttt{bench\_gaps\_results.json})."),
    (r"for a critical-only deployment, uniform weights are preferable on the reactor and neutral on Superconductivity.",
     r"for a critical-only deployment, uniform weights are preferable on the reactor and neutral on Superconductivity (artifacts: \texttt{bench\_weights\_superconduct\_results.json}, \texttt{bench\_weights\_reactor\_results.json})."),
]
for old, new in reps:
    n = t.count(old)
    assert n == 1, (n, old)
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ok")
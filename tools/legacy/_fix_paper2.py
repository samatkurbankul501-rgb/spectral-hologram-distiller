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
    (r"\textbf{Closed-Form Spectral Passport \& Sub-Millisecond Radar:}", r"\textbf{Closed-Form Spectral Passport \& Millisecond-Scale Radar:}"),
    (r"while pruning only lowest-weight background cells, preserving the protected phase-peak band.", r"while pruning only lowest-weight non-protected cells, preserving the protected phase-peak band."),
    (r"removing the phase coordinate is a large degradation, but a narrow band is worse", r"removing the phase coordinate costs 0.85 K"),
]
for old, new in reps:
    n = t.count(old)
    assert n == 1, (n, old)
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ok")
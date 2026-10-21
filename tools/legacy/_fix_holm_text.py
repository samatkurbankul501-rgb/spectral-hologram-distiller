# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")
old = r"SKT remains the strongest baseline on global KRR ($15.78\,\mathrm{K}$; SHD wins $17/20$, $p=6.4\times10^{-3}$)"
new = r"SKT remains the strongest baseline on global KRR ($15.78\,\mathrm{K}$; SHD wins $17/20$, raw $p=6.4\times10^{-3}$, but Holm $p=0.051$, n.s.)"
print("count", t.count(old))
assert t.count(old) == 1
p.write_text(t.replace(old, new), encoding="utf-8")
print("ok")

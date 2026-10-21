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
labels = set(re.findall(r"\\label\{([^}]+)\}", t))
refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
print("labels", sorted(labels))
print("refs", sorted(refs))
print("missing", sorted(refs - labels))

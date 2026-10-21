"""Check: are the 3 syntax errors pre-existing in the ORIGINALS? + list leftovers."""
import ast
from pathlib import Path

ROOT = Path(r"C:\Users\User\Downloads\New folder")
REPO = ROOT / "_release" / "spectral-hologram-distiller"

for name, orig in [
    ("_tmp_lat3.py", ROOT / "_tmp_lat3.py"),
    ("cell_03.py", ROOT / "_cells" / "cell_03.py"),
    ("cell_05.py", ROOT / "_cells" / "cell_05.py"),
]:
    try:
        ast.parse(orig.read_text(encoding="utf-8"))
        print(f"{name}: ORIGINAL parses OK -> migration introduced the error")
    except SyntaxError as exc:
        print(f"{name}: ORIGINAL already invalid ({exc.msg} line {exc.lineno}) -> pre-existing")

print("\n--- leftover absolute literals in migrated files ---")
import re
for f in sorted((REPO / "tools").rglob("*.py")):
    t = f.read_text(encoding="utf-8")
    for m in re.findall(r"[\"']([A-Za-z]:\\[^\"'\n]*)[\"']", t):
        print(f"  {f.relative_to(REPO)}: {m}")

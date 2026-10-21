# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
root = REPO_PATH
p = root / "README.md"
t = p.read_text(encoding="utf-8")
old_title = "# Spectral Hologram Distiller (SHD v3.9.2)"
new_title = "# Spectral Hologram Distiller (SHD v3.10.0)"
assert t.count(old_title) == 1
t = t.replace(old_title, new_title)
old_note = "> **v3.9.2 patch:** hard `max_crystal_size` contract (raises instead of silently\n> overshooting), NPZ `--target-col` support, golden regression test, and an MIT LICENSE."
new_note = "> **v3.10.0:** `DomainBank` (per-domain crystals + nearest-passport routing),\n> `passport_recalibrate=True` for single-crystal heals, `weighting=\"uniform\"` on\n> the loader, PM2.5 section; plus the v3.9.2 hard cap, NPZ `--target-col`,\n> golden regression test, and MIT LICENSE."
assert t.count(old_note) == 1
t = t.replace(old_note, new_note)
p.write_text(t, encoding="utf-8")
for rel, old, new in [
    ("pyproject.toml", 'version = "3.9.2"', 'version = "3.10.0"'),
    ("shd/__init__.py", '__version__ = "3.9.2"', '__version__ = "3.10.0"'),
    ("tests/test_smoke.py", 'assert __version__ == "3.9.2"', 'assert __version__ == "3.10.0"'),
]:
    q = root / rel
    s = q.read_text(encoding="utf-8")
    assert s.count(old) == 1, (rel, s.count(old))
    q.write_text(s.replace(old, new), encoding="utf-8")
print("ok")
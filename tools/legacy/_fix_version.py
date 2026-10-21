# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
root = REPO_PATH
# README
p = root / "README.md"
t = p.read_text(encoding="utf-8")
reps = [
    ("# Spectral Hologram Distiller (SHD v3.9.1)", "# Spectral Hologram Distiller (SHD v3.9.2)"),
    ("| `phase_kappa` | 0.0 / 1.6 / **3.2** / 6.4 | 4.199 / 3.236 / **3.346** / 3.404 | $\\kappa=0$ (no phase manifold) is the worst by a wide margin — the phase geometry is the load-bearing idea, not decoration |",
     "| `phase_kappa` | 0.0 / 1.6 / **3.2** / 6.4 | 4.199 / 3.236 / **3.346** / 3.404 | $\\kappa=0$ (no phase manifold) is a large degradation, but a too-narrow band (0.10) is worse (4.433); the phase geometry is load-bearing, not decoration |"),
]
for old, new in reps:
    n = t.count(old)
    assert n == 1, (n, old[:60])
    t = t.replace(old, new)
# Add a patch note after the title line
needle = "# Spectral Hologram Distiller (SHD v3.9.2)\n"
note = needle + "\n> **v3.9.2 patch:** hard `max_crystal_size` contract (raises instead of silently\n> overshooting), NPZ `--target-col` support, golden regression test, and an MIT LICENSE.\n"
assert t.startswith(needle)
t = t.replace(needle, note, 1)
p.write_text(t, encoding="utf-8")
# Version files
for rel, old, new in [
    ("pyproject.toml", 'version = "3.9.1"', 'version = "3.9.2"'),
    ("shd/__init__.py", '__version__ = "3.9.1"', '__version__ = "3.9.2"'),
    ("tests/test_smoke.py", 'assert __version__ == "3.9.1"', 'assert __version__ == "3.9.2"'),
]:
    q = root / rel
    s = q.read_text(encoding="utf-8")
    assert s.count(old) == 1, (rel, s.count(old))
    q.write_text(s.replace(old, new), encoding="utf-8")
print("ok")
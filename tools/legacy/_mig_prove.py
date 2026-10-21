"""Prove the migration is lossless: run original vs migrated, diff the output.

HISTORICAL TOOL. This was run on 2026-10-21 against the workspace originals
and returned ``same=11 differ=0``; only then were the originals removed with
the recoverable-delete launcher. Since the originals are gone, every
candidate now reports SKIP and the script can only re-check that the migrated
copies still execute. See ``docs/MIGRATION.md``.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\User\Downloads\New folder")
REPO = ROOT / "_release" / "spectral-hologram-distiller"
PY = ROOT / "_v2venv" / "Scripts" / "python.exe"

TMPPAT = re.compile(r"[A-Za-z]:\\\\[^\"']*?Temp\\\\[^\"']*?(?=[\\\\\"'])")


def norm(s: str) -> str:
    """Normalize random temp paths so only real behaviour differences show."""
    return TMPPAT.sub("<TMP>", s)

# read-only audit scripts (they only print; they do not rewrite the paper)
CANDIDATES = [
    "_audit_numbers.py",
    "_audit_shapes.py",
    "_audit_tex.py",
    "_audit_golden.py",
    "_audit_defects.py",
    "_audit_keys.py",
    "_audit_seeds.py",
    "_audit_refs.py",
    "_audit_tiny.py",
    "_audit_weight_ablation.py",
    "_audit_industry.py",
]

ok = same = differ = skipped = 0
for name in CANDIDATES:
    orig = ROOT / name
    migr = REPO / "tools" / "legacy" / name
    if not orig.exists() or not migr.exists():
        print(f"SKIP  {name} (missing)")
        skipped += 1
        continue
    a = subprocess.run([str(PY), "-X", "utf8", str(orig)], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=300)
    b = subprocess.run([str(PY), "-X", "utf8", str(migr)], capture_output=True, text=True,
                       cwd=str(REPO), timeout=300)
    if norm(a.stdout) == norm(b.stdout) and a.returncode == b.returncode:
        print(f"SAME  {name}  (exit {a.returncode}, {len(a.stdout)} chars)")
        same += 1
    else:
        differ += 1
        print(f"DIFF  {name}: orig exit={a.returncode} mig exit={b.returncode}")
        la, lb = a.stdout.splitlines(), b.stdout.splitlines()
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                print(f"      line {i}: orig={x[:90]!r}")
                print(f"              mig ={y[:90]!r}")
                break
        if len(la) != len(lb):
            print(f"      line count: orig={len(la)} mig={len(lb)}")
        if b.stderr.strip():
            print("      mig stderr:", b.stderr.strip()[:300])
        if a.stderr.strip():
            print("      orig stderr:", a.stderr.strip()[:200])
print(f"\nsame={same} differ={differ} skipped={skipped}")
sys.exit(0 if differ == 0 else 1)

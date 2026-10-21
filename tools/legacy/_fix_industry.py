# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "INDUSTRY_ITERATION.md"
t = p.read_text(encoding="utf-8")
reps = [
    ("### Domain bank (SHD-Bank): zero forgetting by construction",
     "### Domain bank (SHD-Bank): zero forgetting by construction\n\nРеализовано в `shd.DomainBank` (`shd/bank.py`): nearest-passport routing поверх\nнескольких `.hologram.npz`; банк не перезаписывает старые кристаллы. Плюс\n`heal_domain_shift(..., passport_recalibrate=True)` для случая, когда старый домен\nбольше не вернётся."),
    ("- `bench_bank.py` → `bench_bank_results.json` (domain bank)",
     "- `bench_bank.py` → `bench_bank_results.json` (domain bank)\n- `bench_passport.py` → `bench_passport_results.json` (blend / recalibrate / bank)\n- `bench_weights.py` → `bench_weights_*_results.json` (gamma vs uniform)"),
]
for old, new in reps:
    n = t.count(old)
    assert n == 1, (n, old[:50])
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ok")
# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "README.md"
t = p.read_text(encoding="utf-8")
reps = [
    ("# Surgical self-healing when a new regime arrives (zero catastrophic forgetting):",
     "# Surgical self-healing when a new regime arrives (bounded forgetting):"),
    ("| **Oracle (full train)** | 12.86 K | 17.59 K | 0.8562 | 10.27 K | 12.72 K |",
     "| **Oracle (KRR 9,000; GBM full train)** | 12.86 K | 17.59 K | 0.8562 | 10.27 K | 12.72 K |"),
    ("captures $64.5$ critical superconductors per crystal vs $56.0$ (KT) and $55.5$\n(SKT).",
     "captures $64.5$ critical superconductors per crystal vs $59.0$ (KT) and $55.0$\n(SKT)."),
    ("Headline: on a physically meaningful shift (low-$T_c$ iron-based → cuprate\nsuperconductors) the naive response is either blindness (RMSE 57 K) or catastrophic\nforgetting (old domain degrades 13.4 → 20.6 K). Surgical heal captures the new\ndomain *and* keeps the old one — with a hard cap on crystal size for production\nmemory budgets.",
     "Headline: on a physically meaningful shift (low-$T_c$ iron-based → cuprate\nsuperconductors) the naive response is either blindness (RMSE 57 K) or catastrophic\nforgetting (old domain degrades 13.4 → 20.6 K). Surgical heal captures the new\ndomain *and* keeps the old one better than overwrite (13.36 vs 20.62 K), but not\nfor free: pre-heal Domain A is 7.32 K, so the heal trades ≈6 K of old-domain\naccuracy for ≈30 K on the new domain. The crystal can be hard-capped for\nproduction memory budgets."),
]
for old, new in reps:
    n = t.count(old)
    if n != 1:
        print("SKIP count", n, repr(old[:50]))
        continue
    t = t.replace(old, new)

marker = "- **OOD threshold:**"
addition = "- **Weighting is a trade-off, not a free win:** on 20 seeds with identical crystals,\n  gamma weights improve global KRR (SC 15.57 vs 15.99; reactor 0.975 vs 0.992;\n  both 20/20) but on the reactor they cost critical KRR (3.426 vs 3.335; uniform\n  wins 20/20). See `bench_weights_*_results.json`.\n"
assert t.count(marker) == 1
t = t.replace(marker, addition + marker)
p.write_text(t, encoding="utf-8")
print("ok")
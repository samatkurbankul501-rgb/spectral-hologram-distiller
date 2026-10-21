# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "README.md"
t = p.read_text(encoding="utf-8")

old = "# Surgical self-healing when a new regime arrives (bounded forgetting):\nheal_info = loader.heal_domain_shift(X_drifted, y_drifted, n_inject=35)\nloader.save(\"my_crystal_healed.hologram.npz\")\n```"
new = "# Surgical self-healing when a new regime arrives (bounded forgetting):\nheal_info = loader.heal_domain_shift(X_drifted, y_drifted, n_inject=35)\nloader.save(\"my_crystal_healed.hologram.npz\")\n\n# Accept the new domain explicitly (the old domain will be flagged):\n# loader.heal_domain_shift(..., passport_recalibrate=True)\n# Or keep both domains in-distribution with a bank:\n# from shd import DomainBank\n# bank = DomainBank([\"a.hologram.npz\", \"b.hologram.npz\"])\n\n# Per-deployment weighting (see bench_weights_*):\nX_g, y_g, w_g = loader.get_training_data(weighting=\"gamma\")    # default\nX_u, y_u, w_u = loader.get_training_data(weighting=\"uniform\")  # critical-only\n```"
assert t.count(old) == 1
t = t.replace(old, new)

old = "blend the spectral passport without touching old medoids."
new = "blend the spectral passport (old medoids are kept unless a `max_crystal_size` cap prunes low-weight cells)."
assert t.count(old) == 1
t = t.replace(old, new)

old = "| Post-heal BW on Domain B | 0.465 | passport blends toward the new domain (0.636 → 0.465); one heal does **not** yet clear the 0.085 alert threshold |"
new = "| Post-heal BW on Domain B | 0.465 | default blend leaves both A (0.203) and B (0.458) above 0.085; `passport_recalibrate=True` gives B 0.038 / A 0.497; `DomainBank` accepts both |"
assert t.count(old) == 1
t = t.replace(old, new)

marker = "Headline: on a physically meaningful shift"
addition = """Passport modes (5 seeds, `bench_passport_results.json`): pre-heal A 0.035 / B 0.629;
blend A 0.203 / B 0.458 (both OOD); recalibrate A 0.497 / B 0.038 (new in, old
flagged); bank routes 5/5 and accepts both 5/5. Use recalibration only if the old
domain will not return; use a bank when both regimes remain live.

"""
assert t.count(marker) == 1
t = t.replace(marker, addition + marker)

marker = "## Known limitations"
bullet = """## Known limitations

- **Single-passport healing is a three-way trade-off:** the default blend can
  leave *both* domains uncalibrated on a strong shift; `passport_recalibrate=True`
  accepts the new domain but flags the old; `DomainBank` accepts both at the cost
  of one crystal per domain. See `bench_passport_results.json`.
"""
assert t.count(marker) == 1
t = t.replace(marker, bullet)

p.write_text(t, encoding="utf-8")
print("ok")
# Workspace migration (2026-10-21)

Everything that used to live loose in the workspace root
(`C:\Users\User\Downloads\New folder\`) is now inside this repository and under
version control. The reason is not tidiness: the claim audit
(`tools/verify_claims.py`) lived outside git, went stale against the v3.11
artifacts, and reported 9 false failures on a 217-check suite. Anything that
guards a published number must live next to the code it validates.

## What moved where

| Source in the workspace root | Now in the repository | What it is |
| :--- | :--- | :--- |
| `_audit_*.py`, `_fix_*.py`, `_tmp_*.py`, `_keys2.py`, `_bump_310.py`, `_add_*.py` | `tools/legacy/` | The paper/README audit chain: scripts that re-derived the numbers in `paper/main.tex` and re-applied the text fixes. They are the reason the paper's figures can be re-verified. |
| `_inspect_nb.py`, `_inspect_npz.py`, `_verify_kt.py` | `tools/legacy/` | One-off inspection helpers (notebook cells, NPZ contents, the kernel-thinning port check). |
| `_hunt/*.py` | `tools/hunt/` | The 2026-10-21 defect hunt: fuzzers, reproducers and probes for the core (`fuzz_core.py`, `hunt_core.py`, `repro_*.py`, `check_*.py`). |
| `_cells/cell_NN.py` | `docs/notebook_cells/` | The Colab notebook's cells extracted as plain Python, with their captured outputs (`cell_NN_out.txt`). |
| `Untitled22.ipynb` | `docs/notebook/` | The origin notebook: the v3.2 → v3.7 evolution in Colab, including the superconductivity capture (91 → 36 → 69). |
| `ИИ-проекты на стыке науки и старта.md` | `docs/development_history_gemini_export.md` | The development log (the SHD v3.3 synthesis and the Colab port). |
| `ОТЧЁТ_РАЗБОР_SHD_v3.11.md` | `docs/review_report_shd_v3.11.md` | The v3.11 review report. |
| `ОХОТА_НА_КОСЯКИ_SHD_v3.11.md` | `docs/hunt_report_shd_v3.11.md` | The v3.11 defect-hunt report. |
| `_verify_all.py` | `tools/verify_claims.py` | The claim audit itself (the root file is now a thin shim that calls the in-repo copy). |

## How the migration was done and proved

`tools/legacy/_migrate_root.py` performed the move: it rewrote every absolute
workspace path into two self-locating resolvers, so the scripts now work from
any directory and from any clone:

```python
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
```

`tools/legacy/_mig_prove.py` proved the move lossless by running the original
and the migrated copy side by side and diffing their output: **11/11 audit
scripts produce byte-identical output** (the only difference was a random
temporary-directory name, which the comparison normalizes).

The originals were removed with the recoverable-delete launcher after the
proof, so they are still recoverable from the trash.

## Notes for future readers

- The scripts in `tools/legacy/` and `tools/hunt/` are **historical**: they
  produced the artifacts cited in the paper, and several of them were
  one-shot. They are kept because the audit chain is the provenance of the
  published numbers, not because they are a maintained toolchain.
- `tools/verify_claims.py` and `tools/hunt_probes.py` are the maintained
  entry points: the first is the gate (217 checks at migration time; it has
  since grown with each release), the second reproduces the
  v3.11.1 defect fixes on demand.

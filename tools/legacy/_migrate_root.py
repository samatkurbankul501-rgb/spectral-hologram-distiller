"""Migrate the workspace's loose scripts into the repository (v3.11.1).

Step 1 of the tidy-up: the audit chain, the hunt probes and the extracted
notebook cells lived in the workspace root, outside version control, which
is how the claim audit went stale. They are copied into the repository with
their absolute path literals rewritten to self-locating resolvers.

The originals are NOT deleted here; the caller removes them with the
recoverable-delete launcher after this script verifies the copies.
"""
from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\User\Downloads\New folder")
REPO = ROOT / "_release" / "spectral-hologram-distiller"
REPO_STR = str(REPO)
WS_STR = str(ROOT)

HEADER = (
    "# Migrated into the repository on 2026-10-21: the absolute workspace\n"
    "# paths below were replaced by self-locating resolvers so the script\n"
    "# keeps working from any directory and from any clone.\n"
    "from pathlib import Path as _Path\n"
    "REPO_PATH = _Path(__file__).resolve().parents[2]\n"
    "WORKSPACE_PATH = REPO_PATH.parents[1]\n"
)

# longest first: the repo literal is a prefix of the file literals
LITERALS = [
    REPO_STR + r"\paper\main.tex",
    REPO_STR + r"\README.md",
    REPO_STR + r"\INDUSTRY_ITERATION.md",
    REPO_STR + r"\tests",
    REPO_STR,
    WS_STR,
]


def to_parts(lit: str) -> tuple[str, list[str]]:
    if lit.startswith(REPO_STR):
        rest = lit[len(REPO_STR):].strip("\\/")
        return "REPO_PATH", [p for p in rest.split("\\") if p]
    if lit.startswith(WS_STR):
        rest = lit[len(WS_STR):].strip("\\/")
        return "WORKSPACE_PATH", [p for p in rest.split("\\") if p]
    raise ValueError("literal outside the known roots: " + lit)


def rewrite(text: str) -> str:
    # 1) Path(r"<abs>") -> REPO_PATH / "a" / "b"
    def repl_path(m: re.Match) -> str:
        base, parts = to_parts(m.group(1))
        expr = base
        for p in parts:
            expr += f' / "{p}"'
        return expr

    for q in ('"', "'"):
        text = re.sub(r'Path\(\s*r' + q + r'(' + re.escape(REPO_STR) + r'[^' + q + r']*)' + q + r'\s*\)',
                      repl_path, text)
        text = re.sub(r'Path\(\s*r' + q + r'(' + re.escape(WS_STR) + r'[^' + q + r']*)' + q + r'\s*\)',
                      repl_path, text)

    # 2) bare r"<abs>" -> str(REPO_PATH) / str(WORKSPACE_PATH)
    def repl_bare(m: re.Match) -> str:
        base, parts = to_parts(m.group(1))
        if parts:
            expr = base
            for p in parts:
                expr += f' / "{p}"'
            # bare use: sys.path / os.path.join / open() all need a str
            return f"str({expr})"
        return f"str({base})"

    for q in ('"', "'"):
        text = re.sub(r'r' + q + r'(' + re.escape(REPO_STR) + r'[^' + q + r']*)' + q,
                      repl_bare, text)
        text = re.sub(r'r' + q + r'(' + re.escape(WS_STR) + r'[^' + q + r']*)' + q,
                      repl_bare, text)
    return text


def migrate(src: Path, dest_dir: Path) -> tuple[Path, bool]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = src.read_text(encoding="utf-8-sig", errors="replace")
    new = rewrite(raw)
    if src.suffix == ".py" and ("REPO_PATH" in new or "WORKSPACE_PATH" in new):
        # the rewrite introduced a resolver reference: define it
        if "REPO_PATH = " not in new:
            new = HEADER + new
    dest = dest_dir / src.name
    dest.write_text(new, encoding="utf-8")
    return dest, new != raw


def main() -> None:
    moved: list[tuple[Path, Path]] = []
    rewritten = 0

    for f in sorted(ROOT.glob("*.py")):
        if f.name in ("_verify_all.py", "_mig_inventory.py"):
            continue
        d, ch = migrate(f, REPO / "tools" / "legacy")
        moved.append((f, d))
        rewritten += bool(ch)

    for f in sorted((ROOT / "_hunt").glob("*.py")):
        d, ch = migrate(f, REPO / "tools" / "hunt")
        moved.append((f, d))
        rewritten += bool(ch)

    for f in sorted((ROOT / "_hunt").glob("*.log")):
        dest = REPO / "tools" / "hunt" / f.name
        shutil.copy2(f, dest)
        moved.append((f, dest))

    cells = REPO / "docs" / "notebook_cells"
    cells.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "_cells").glob("*")):
        dest = cells / f.name
        shutil.copy2(f, dest)
        moved.append((f, dest))

    docs = REPO / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    doc_map = {
        "ИИ-проекты на стыке науки и старта.md": "development_history_gemini_export.md",
        "ОТЧЁТ_РАЗБОР_SHD_v3.11.md": "review_report_shd_v3.11.md",
        "ОХОТА_НА_КОСЯКИ_SHD_v3.11.md": "hunt_report_shd_v3.11.md",
    }
    for src_name, dest_name in doc_map.items():
        src = ROOT / src_name
        if src.exists():
            shutil.copy2(src, docs / dest_name)
            moved.append((src, docs / dest_name))

    legacy_txt = ROOT / "_commit_msg2.txt"
    if legacy_txt.exists():
        shutil.copy2(legacy_txt, REPO / "tools" / "legacy" / "_commit_msg2.txt")
        moved.append((legacy_txt, REPO / "tools" / "legacy" / "_commit_msg2.txt"))

    print(f"copied {len(moved)} files, {rewritten} with rewritten paths")
    bad = []
    for src, dest in moved:
        if dest.suffix == ".py":
            try:
                ast.parse(dest.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                bad.append((dest, exc))
    print("syntax errors:", len(bad))
    for dest, exc in bad:
        print("  ", dest, exc)
    # leftover absolute literals
    leftovers = []
    for src, dest in moved:
        if dest.suffix == ".py":
            t = dest.read_text(encoding="utf-8")
            for m in re.findall(r"[\"']([A-Za-z]:\\[^\"'\n]*)[\"']", t):
                leftovers.append((dest.name, m))
    print("leftover absolute literals:", len(leftovers))
    for n, m in leftovers[:10]:
        print("  ", n, m)

    with open(ROOT / "_mig_sources.txt", "w", encoding="utf-8") as fh:
        for src, dest in moved:
            fh.write(f"{src}\t{dest}\n")
    print("manifest -> _mig_sources.txt")


if __name__ == "__main__":
    main()

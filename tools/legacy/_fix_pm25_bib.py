# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
root = REPO_PATH
# Paper citation
p = root / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")
assert t.count(r"\citep{chen2015pm25}") == 1
t = t.replace(r"\citep{chen2015pm25}", r"\citep{liang2015pm25}")
# numeric cells in passport table
t = t.replace(r"Domain bank (nearest) & below threshold & below threshold & both in", r"Domain bank (nearest) & $<0.085$ & $<0.085$ & both in")
p.write_text(t, encoding="utf-8")
# Bib entry
b = root / "paper" / "references.bib"
s = b.read_text(encoding="utf-8")
entry = r"""@article{liang2015pm25,
  title   = {Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating},
  author  = {Liang, Xuan and Zou, Tao and Guo, Bin and Li, Shuo and Zhang, Haozhe and Zhang, Shuyi and Huang, Hui and Chen, S. X.},
  journal = {Proceedings of the Royal Society A},
  volume  = {471},
  number  = {2182},
  pages   = {20150257},
  year    = {2015}
}

"""
assert "liang2015pm25" not in s
s = entry + s
b.write_text(s, encoding="utf-8")
print("ok")
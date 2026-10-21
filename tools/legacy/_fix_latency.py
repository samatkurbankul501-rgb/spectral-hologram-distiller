# Migrated into the repository on 2026-10-21: the absolute workspace
# paths below were replaced by self-locating resolvers so the script
# keeps working from any directory and from any clone.
from pathlib import Path as _Path
REPO_PATH = _Path(__file__).resolve().parents[2]
WORKSPACE_PATH = REPO_PATH.parents[1]
from pathlib import Path
p = REPO_PATH / "paper" / "main.tex"
t = p.read_text(encoding="utf-8")
reps = [
    (r"a millisecond-scale Bures--Wasserstein radar (4--84\,ms depending on window size)",
     r"a millisecond-scale Bures--Wasserstein radar (5--25\,ms for 64--5{,}780-row windows; 50\,ms for the 7{,}300-row heal window)"),
    (r"detects out-of-distribution drift in 4--84\,ms on CPU (window-size dependent)",
     r"detects out-of-distribution drift in 5--25\,ms on CPU (median for 64--5,780-row windows; 50\,ms for the 7,300-row heal window)"),
    (r"detection latency is 4--84\,ms depending on window size ($B = 64\ldots 5{,}780$) on this CPU, not sub-millisecond.",
     r"detection latency is 4.8--24.9\,ms (medians for $B = 64\ldots 5{,}780$) on this CPU, not sub-millisecond; the artifact is \texttt{bench\_latency\_results.json}."),
    (r"Median latency for the 7,300-row Domain B window is $50.2\,\mathrm{ms}$ ($4$--$84\,\mathrm{ms}$ across window sizes); it is not sub-millisecond.",
     r"Median latency for the 7,300-row Domain B window is $50.2\,\mathrm{ms}$ ($4.8$--$24.9\,\mathrm{ms}$ for 64--5,780 rows); it is not sub-millisecond."),
]
for old, new in reps:
    n = t.count(old)
    assert n == 1, (n, old[:60])
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ok")
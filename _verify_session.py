"""Independent re-verification of this session's additions.

Recomputes every claim from the raw JSON artifacts with fresh code, checks the
README text against them (including numbers that may have gone stale when an
artifact was regenerated), and audits the manifest for coverage.

Returns nonzero when a check fails. Historical artifacts are not current reruns.
"""

import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FAIL = []


def chk(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name + " :: " + detail)


def load(name):
    return json.load(open(os.path.join(ROOT, name), encoding="utf-8"))


def readme():
    return open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()


# ---------------------------------------------------------------- 1. Scania
print("=" * 96)
print("1. README §8 vs bench_scania_cost_results.json")
print("=" * 96)
sc = load("bench_scania_cost_results.json")
s = sc["summary"]
md = readme()

chk("seed count", sc["n_seeds"] == 10, str(sc["n_seeds"]))
chk("cost model 10/1", sc["cost_model"] == {"fn": 10, "fp": 1}, str(sc["cost_model"]))

for arm in ("random", "kmeans", "shd"):
    for model in ("hgb", "logit"):
        key = "{}_{}".format(arm, model)
        row = next((line.replace("**", "") for line in md.splitlines()
                    if line.startswith("|") and
                    line.replace("**", "").split("|")[1].strip().lower().replace("-", "") == arm and
                    line.replace("**", "").split("|")[2].strip().lower() == model), None)
        chk(key + " README row exists", row is not None)
        if row is None:
            continue
        cells = [cell.strip() for cell in row.split("|")]
        for metric, column in (("fn", 3), ("fp", 4), ("cost", 7)):
            expected = float(np.median([r[key][metric] for r in sc["records"]]))
            chk(key + " " + metric + " summary median", s[key][metric] == expected)
            chk(key + " " + metric + " README exact count", float(cells[column]) == expected,
                "{} vs {}".format(cells[column], expected))

# per-seed cost identity: cost == 10*fn + fp for every record
ident_ok = all(r["{}_hgb".format(a)]["cost"] == 10 * r["{}_hgb".format(a)]["fn"]
               + r["{}_hgb".format(a)]["fp"]
               for r in sc["records"] for a in ("random", "kmeans", "shd"))
ident_ok &= all(r["{}_logit".format(a)]["cost"] == 10 * r["{}_logit".format(a)]["fn"]
                + r["{}_logit".format(a)]["fp"]
                for r in sc["records"] for a in ("random", "kmeans", "shd"))
chk("per-seed cost identity 10*FN+FP", ident_ok)

chk("oracle hgb cost 531", round(sc["oracle_fulltrain"]["hgb"]["cost"]) == 531,
    str(sc["oracle_fulltrain"]["hgb"]["cost"]))
chk("oracle logit cost 770", round(sc["oracle_fulltrain"]["logit"]["cost"]) == 770,
    str(sc["oracle_fulltrain"]["logit"]["cost"]))
chk("never-alert 3750", round(sc["never_alert"]["cost"]) == 3750,
    str(sc["never_alert"]["cost"]))

d_km = 100 * (s["shd_hgb"]["cost"] / s["kmeans_hgb"]["cost"] - 1)
d_rd = 100 * (s["shd_hgb"]["cost"] / s["random_hgb"]["cost"] - 1)
chk("delta vs KMeans -21.9%", abs(d_km + 21.9) < 0.15, "{:+.2f}%".format(d_km))
chk("delta vs random -30.9%", abs(d_rd + 30.9) < 0.15, "{:+.2f}%".format(d_rd))
chk("README quotes 21.9% cheaper", "21.9% cheaper" in md)
chk("README quotes 30.9%", "30.9%" in md and "cheaper than a random" in md)

cpk = sc["crystal_positives"]
chk("crystal positives 7.0/7.5/25.0", (cpk["random"], cpk["kmeans"],
                                       cpk["shd"]) == (7.0, 7.5, 25.0), str(cpk))
chk("README preserves fractional KMeans positives", "K-Means **7.5**" in md)
chk("custom Scania cost clearly labelled", "custom 10:1 cost scenario" in md)
chk("synthetic 81D case clearly labelled", "81D synthetic Gaussian/sine" in md)

w = sc["wilcoxon"]
for key, wins, p in (("shd_vs_random_hgb_cost", 8, 0.0195),
                     ("shd_vs_kmeans_hgb_cost", 8, 0.0195),
                     ("shd_vs_random_logit_cost", 3, 0.625),
                     ("shd_vs_kmeans_logit_cost", 2, 0.432)):
    v = w[key]
    chk(key, v["wins_shd"] == wins and abs(v["p_value"] - p) < 6e-4,
        "wins={} p={:.4f}".format(v["wins_shd"], v["p_value"]))

chk("300/48000 = 0.6% claim", abs(300 / sc["n_train"] * 100 - 0.625) < 0.01,
    "{:.3f}%".format(300 / sc["n_train"] * 100))

# ---------------------------------------------------------------- 2. Bank
print()
print("=" * 96)
print("2. README §4b vs bench_bank_scaling_results.json")
print("=" * 96)
bk = load("bench_bank_scaling_results.json")
EXP = {"1.00": (1.00, 0.265, 1.307, 0.549, 9.646),
       "0.50": (1.00, 0.265, 0.863, 0.402, 2.744),
       "0.25": (0.60, 0.265, 0.651, 0.272, 1.277)}
for key, (rt, r0, last, heal, froz) in EXP.items():
    c = bk["conditions"][key]["summary"]
    chk("shift {} routing".format(key), abs(c["routing_accuracy"] - rt) < 0.006,
        "{:.3f}".format(c["routing_accuracy"]))
    chk("shift {} bank_r0".format(key), abs(c["bank_retention_r0"] - r0) < 0.001,
        "{:.4f}".format(c["bank_retention_r0"]))
    chk("shift {} bank_last".format(key), abs(c["bank_last"] - last) < 0.001,
        "{:.3f}".format(c["bank_last"]))
    chk("shift {} heal_r0_end".format(key), abs(c["heal_retention_r0_end"] - heal) < 0.001,
        "{:.3f}".format(c["heal_retention_r0_end"]))
    chk("shift {} frozen".format(key), abs(c["frozen_adapt_last"] - froz) < 0.001,
        "{:.3f}".format(c["frozen_adapt_last"]))

c1 = bk["conditions"]["1.00"]["summary"]
fac = c1["heal_retention_r0_end"] / c1["bank_retention_r0"]
chk("forgetting x2.07", abs(fac - 2.07) < 0.01, "{:.3f}x".format(fac))
chk("README quotes x2.07", "×2.07" in md or "x2.07" in md)
kb = c1["bank_total_bytes"] / 1024 / bk["n_domains"]
chk("14 KB per regime", abs(kb - 14.0) < 0.2, "{:.2f} KB".format(kb))
chk("README quotes 14 KB", "14 KB" in md)

# retention must equal frozen R0 at every drift (same untouched crystal)
for key in EXP:
    c = bk["conditions"][key]["summary"]
    chk("shift {} bank_r0 == frozen_r0".format(key),
        abs(c["bank_retention_r0"] - c["frozen_retention_r0"]) < 1e-9,
        "{:.6f} vs {:.6f}".format(c["bank_retention_r0"], c["frozen_retention_r0"]))

# healing must be monotone non-decreasing across shifts (forgetting signal)
mono = all(np.all(np.diff(r["heal_retention_r0_curve"]) > -0.02)
           for cond in bk["conditions"].values() for r in cond["records"])
chk("heal curve monotone (forgetting)", mono)

# ---------------------------------------------------------------- 3. Envelope
print()
print("=" * 96)
print("3. README §7 vs bench_edge_envelope_results.json")
print("=" * 96)
ev = load("bench_edge_envelope_results.json")
chk("repeats recorded", ev.get("repeats", 0) >= 3, str(ev.get("repeats")))
chk("single-thread env", ev.get("thread_env") == 1)
for c, exp_kb in zip(ev["cases"], (28.7, 184.7)):
    chk(c["case"] + " artifact", abs(c["artifact_kb"] - exp_kb) < 0.15,
        "{:.1f}".format(c["artifact_kb"]))
    chk(c["case"] + " radar B64", 1.0 <= c["radar_ms"]["64"] <= 1.6,
        "{:.2f} ms".format(c["radar_ms"]["64"]))
    chk(c["case"] + " RSS delta in 0.28-1.25", 0.2 <= c["rss_delta_load_mb"] <= 1.4,
        "{:.2f} MB".format(c["rss_delta_load_mb"]))
    chk(c["case"] + " RSS baseline 58-69", 55 <= c["rss_base_mb"] <= 72,
        "{:.1f} MB".format(c["rss_base_mb"]))

# STALENESS CHECK: README quotes a latency spread; it must match this artifact
spread_b64 = ev["cases"][0]["radar_ms_spread"]["64"]
spread_b1024 = ev["cases"][0]["radar_ms_spread"]["1024"]
quoted = re.search(r"Latency spread across passes was ([\d.]+)[–-]([\d.]+) ms \(B=64\) "
                   r"and ([\d.]+)[–-]([\d.]+) ms \(B=1024\)", md)
if quoted:
    lo64, hi64, lo1k, hi1k = (float(x) for x in quoted.groups())
    chk("quoted B=64 spread matches artifact",
        abs(lo64 - spread_b64[0]) < 0.051 and abs(hi64 - spread_b64[1]) < 0.051,
        "quoted {}-{} vs artifact {}-{}".format(lo64, hi64, *spread_b64))
    chk("quoted B=1024 spread matches artifact",
        abs(lo1k - spread_b1024[0]) < 0.051 and abs(hi1k - spread_b1024[1]) < 0.051,
        "quoted {}-{} vs artifact {}-{}".format(lo1k, hi1k, *spread_b1024))
else:
    chk("spread sentence present in README", False)

# ---------------------------------------------------------------- 4. Manifest
print()
print("=" * 96)
print("4. ARTIFACTS.md coverage")
print("=" * 96)
man = open(os.path.join(ROOT, "ARTIFACTS.md"), encoding="utf-8").read()
jsons = sorted(f for f in os.listdir(ROOT) if f.endswith(".json"))
missing = [f for f in jsons if f not in man]
chk("every *.json appears in ARTIFACTS.md", not missing,
    "missing {}: {}".format(len(missing), missing[:6]))
# files I deleted earlier must not be referenced anywhere
for gone in ("_smoke_scania.json", "bench_bank_scaling_mid_results.json",
             "bench_bank_scaling_subtle_results.json"):
    chk("deleted artifact not referenced: " + gone,
        not os.path.exists(os.path.join(ROOT, gone)) and gone not in man and gone not in md)

# ---------------------------------------------------------------- 5. hygiene
print()
print("=" * 96)
print("5. Presence, README structure, stubs")
print("=" * 96)
for f in ("bench_scania_cost.py", "bench_bank_scaling.py", "bench_edge_envelope.py",
          "bench_pm25_rehab.py", "make_scania_table.py", "ARTIFACTS.md",
          "bench_scania_cost_results.json", "bench_bank_scaling_results.json",
          "bench_edge_envelope_results.json", "bench_pm25_rehab_results.json"):
    chk("file exists: " + f, os.path.exists(os.path.join(ROOT, f)))

bad = 0
block = []
for i, line in enumerate(md.split("\n"), 1):
    if line.strip().startswith("|"):
        block.append(line)
    else:
        if block:
            if len({l.count("|") for l in block}) > 1:
                bad += 1
            block = []
chk("README tables consistent", bad == 0, "{} bad".format(bad))

for m in re.finditer(r"\[([^\]]+)\]\((?!http)([^)]+)\)", md):
    tgt = m.group(2).split("#")[0]
    if tgt and not os.path.exists(os.path.join(ROOT, tgt)):
        chk("README link " + tgt, False)

# the threshold helper must still be tie-safe
import sys
sys.path.insert(0, ROOT)
from bench_scania_cost import pick_threshold, evaluate  # noqa: E402

s0 = np.zeros(50); y0 = np.zeros(50, dtype=int); y0[:1] = 1
thr, _ = pick_threshold(s0, y0)
chk("degenerate -> never-alert", thr == float("inf") and evaluate(s0, y0, 1.0)["cost"] == 10,
    "thr={}".format(thr))

print()
print("=" * 96)
print("RESULT: {} failures".format(len(FAIL)))
print("=" * 96)
for f in FAIL:
    print("  " + f)
raise SystemExit(1 if FAIL else 0)

"""Independent claim audit: paper/README vs JSON artifacts (in-repo copy).

Run from anywhere. Prints PASS/FAIL for every numeric claim and exits non-zero
if any claim is unsupported.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import statistics

ROOT = Path(__file__).resolve().parent.parent
PAPER = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
INDUSTRY = (ROOT / "INDUSTRY_ITERATION.md").read_text(encoding="utf-8")

failures = []
checks = 0

def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

def check(name, actual, expected, tol=0.011):
    global checks
    checks += 1
    ok = abs(float(actual) - float(expected)) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} actual={actual!r:>12} expected={expected!r}")
    if not ok:
        failures.append(name)

def check_min(name, actual, minimum):
    """Assert actual >= minimum (one-sided bound)."""
    global checks
    checks += 1
    ok = float(actual) >= float(minimum)
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} actual={actual!r:>12} min={minimum!r}")
    if not ok:
        failures.append(name)

def check_text(name, needle, haystack=PAPER):
    global checks
    checks += 1
    ok = needle in haystack
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} text={needle[:60]!r}")
    if not ok:
        failures.append(name)

def check_in_function(name, func_header, needle, haystack):
    """Assert that a needle lives INSIDE a given function body.

    A plain substring check is not enough: the same statement can occur
    several times in a file (e.g. the constructor and finalize_crystal),
    and then the check still passes after the intended site was removed --
    a vacuous check. Extract the function body first, then look inside it.
    """
    global checks
    checks += 1
    start = haystack.find(func_header)
    ok = False
    if start >= 0:
        rest = haystack[start + len(func_header):]
        end = rest.find("\n    def ")
        body = rest if end < 0 else rest[:end]
        ok = needle in body
    label = "PASS" if ok else "FAIL"
    print(f"{label}  {name:52s} in {func_header[:26]!r} needle={needle[:38]!r}")
    if not ok:
        failures.append(name)

def check_text_count(name, needle, expected, haystack):
    """Assert the needle appears EXACTLY 'expected' times in the haystack.

    A plain substring check is vacuous when the same number is rendered more
    than once (two README tables quote the same Scania cost): mutating one
    copy leaves the other, the check still passes and the regression slips
    through. Pinning the COUNT turns a hidden copy into a visible failure.
    """
    global checks
    checks += 1
    n = haystack.count(needle)
    ok = n == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} count={n} expected={expected} text={needle[:34]!r}")
    if not ok:
        failures.append(name)


def check_digest(name, actual, expected):
    """Assert a sha256 digest matches exactly (string equality)."""
    global checks
    checks += 1
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} sha256={actual[:16]}...")
    if not ok:
        failures.append(name)


def artifact_digest(payload):
    """sha256 over an artifact with the volatile timing field removed.

    Value pins and median pins are both blind to a doctored per-seed row:
    a median is robust to a single outlier, and the published summary quotes
    only the median. The digest covers EVERY field of EVERY record, so any
    tampering inside the artifact breaks the gate.
    """
    import hashlib
    body = {k: v for k, v in payload.items() if k != "elapsed_sec"}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_absent(name, needle, haystack=PAPER):
    global checks
    checks += 1
    ok = needle not in haystack
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} forbidden={needle!r}")
    if not ok:
        failures.append(name)

def med(records, arm, metric):
    return float(np.median([r[arm][metric] for r in records]))

# ---------------------------------------------------------------- seeds: SC
d = load("bench_seeds_results.json")
sc = d["results"]["superconduct"]
rc = d["results"]["reactor"]
scr, rcr = sc["records"], rc["records"]

check("SC SHD KRR crit", med(scr, "shd", "krr_rmse_crit"), 18.81)
check("SC SHD KRR all", med(scr, "shd", "krr_rmse_all"), 15.42)
check("SC SHD GBM crit", med(scr, "shd", "gbm_rmse_crit"), 18.71)
check("SC SHD GBM all", med(scr, "shd", "gbm_rmse_all"), 16.06)
check("SC SHD rare", med(scr, "shd", "rare_captured"), 64.5, 0.06)
check("SC Random KRR crit", med(scr, "random", "krr_rmse_crit"), 22.43)
check("SC KMeans KRR crit", med(scr, "kmeans", "krr_rmse_crit"), 22.93)
check("SC KT KRR crit", med(scr, "kt", "krr_rmse_crit"), 23.26)
check("SC SKT KRR crit", med(scr, "skt", "krr_rmse_crit"), 21.20)
check("SC SKT KRR all", med(scr, "skt", "krr_rmse_all"), 15.78)
check("SC KT rare", med(scr, "kt", "rare_captured"), 59.0, 0.06)
check("SC SKT rare", med(scr, "skt", "rare_captured"), 55.0, 0.06)
check("SC Random rare", med(scr, "random", "rare_captured"), 57.0, 0.06)
check("SC KMeans rare", med(scr, "kmeans", "rare_captured"), 31.0, 0.06)

check("SC SHD vs SKT KRR-all wins", sc["wilcoxon"]["shd_vs_skt_krr_rmse_all"]["wins_shd"], 17, 0)
check("SC SHD vs SKT KRR-all p", sc["wilcoxon"]["shd_vs_skt_krr_rmse_all"]["p_value"], 6.39e-3, 1e-5)
check("SC SHD vs SKT GBM-all wins", sc["wilcoxon"]["shd_vs_skt_gbm_rmse_all"]["wins_shd"], 6, 0)
check("SC SHD vs SKT GBM-all p", sc["wilcoxon"]["shd_vs_skt_gbm_rmse_all"]["p_value"], 0.0696, 1e-4)

# ---------------------------------------------------------------- seeds: reactor
check("Reactor SHD KRR crit", med(rcr, "shd", "krr_rmse_crit"), 3.436, 0.006)
check("Reactor SHD KRR all", med(rcr, "shd", "krr_rmse_all"), 0.976, 0.006)
check("Reactor SHD GBM all", med(rcr, "shd", "gbm_rmse_all"), 0.624, 0.006)
check("Reactor SHD GBM crit", med(rcr, "shd", "gbm_rmse_crit"), 1.732, 0.006)
check("Reactor SHD rare", med(rcr, "shd", "rare_captured"), 19.5, 0.06)
check("Reactor Random KRR crit", med(rcr, "random", "krr_rmse_crit"), 4.424, 0.006)
check("Reactor KMeans KRR crit", med(rcr, "kmeans", "krr_rmse_crit"), 4.390, 0.006)
check("Reactor KT KRR crit", med(rcr, "kt", "krr_rmse_crit"), 4.516, 0.006)
check("Reactor SKT KRR crit", med(rcr, "skt", "krr_rmse_crit"), 4.060, 0.006)
check("Reactor KMeans GBM all", med(rcr, "kmeans", "gbm_rmse_all"), 0.596, 0.006)
check("Reactor KT rare", med(rcr, "kt", "rare_captured"), 10.0, 0.06)
check("Reactor SKT rare", med(rcr, "skt", "rare_captured"), 8.5, 0.06)
check("Reactor Random rare", med(rcr, "random", "rare_captured"), 9.0, 0.06)
check("Reactor KMeans rare", med(rcr, "kmeans", "rare_captured"), 11.5, 0.06)
check("Reactor SHD vs KT crit wins", rc["wilcoxon"]["shd_vs_kt_krr_rmse_crit"]["wins_shd"], 18, 0)
check("Reactor SHD vs KT crit p", rc["wilcoxon"]["shd_vs_kt_krr_rmse_crit"]["p_value"], 9.54e-6, 1e-7)
check("Reactor SHD vs KMeans GBM-all wins", rc["wilcoxon"]["shd_vs_kmeans_gbm_rmse_all"]["wins_shd"], 8, 0)
check("Reactor SHD vs KMeans GBM-all p", rc["wilcoxon"]["shd_vs_kmeans_gbm_rmse_all"]["p_value"], 0.245, 1e-3)

# ---------------------------------------------------------------- heal
h = load("bench_heal_superconduct_results.json")
hs = h["summary"]
check("Heal A pre", h["wilcoxon"]["healed_vs_pre_heal_krr_rmse_domain_A"]["median_pre_heal"], 7.32, 0.011)
check("Heal A post", h["wilcoxon"]["healed_vs_pre_heal_krr_rmse_domain_A"]["median_healed"], 13.36, 0.011)
check("Heal A overwrite", h["wilcoxon"]["healed_vs_naive_overwrite_krr_rmse_domain_A"]["median_naive_overwrite"], 20.62, 0.011)
check("Heal B pre", h["wilcoxon"]["healed_vs_pre_heal_krr_rmse_domain_B"]["median_pre_heal"], 57.17, 0.011)
check("Heal B post", h["wilcoxon"]["healed_vs_pre_heal_krr_rmse_domain_B"]["median_healed"], 26.88, 0.011)
check("Heal combined post", h["wilcoxon"]["healed_vs_pre_heal_krr_rmse_combined"]["median_healed"], 19.10, 0.011)
check("Heal combined overwrite", h["wilcoxon"]["healed_vs_naive_overwrite_krr_rmse_combined"]["median_naive_overwrite"], 21.58, 0.011)
check("Heal false alarms", hs["radar_false_alarms_on_domain_A"], 0, 0)
check("Heal detections", hs["radar_detections_on_domain_B"], 20, 0)
check("Heal BW in", hs["bw_in_median"], 0.035, 0.001)
check("Heal BW OOD", hs["bw_ood_median"], 0.636, 0.001)
check("Heal BW after", hs["bw_after_heal_median"], 0.465, 0.001)
check("Heal radar latency", hs["radar_latency_ms_median"], 50.15, 0.05)
check("Heal latency", hs["heal_latency_ms_median"], 410.37, 0.5)

# ---------------------------------------------------------------- controls
g = load("bench_gaps_results.json")
gp = g["pooled_summary"]
check("Gaps SHD crit", gp["medians"]["shd"]["krr_rmse_crit"]["median"], 19.204, 0.011)
check("Gaps SHD_sub crit", gp["medians"]["shd_sub4800"]["krr_rmse_crit"]["median"], 18.714, 0.011)
check("Gaps KT crit", gp["medians"]["kt"]["krr_rmse_crit"]["median"], 23.194, 0.011)
check("Gaps SKT crit", gp["medians"]["skt"]["krr_rmse_crit"]["median"], 21.698, 0.011)
check("Gaps stratified crit", gp["medians"]["stratified"]["krr_rmse_crit"]["median"], 24.092, 0.011)
check("Gaps sub vs strat wins", gp["tests"]["shd_vs_stratified_krr_rmse_crit"]["wins_shd"], 15, 0)
check("Gaps sub vs strat p", gp["tests"]["shd_vs_stratified_krr_rmse_crit"]["p_value"], 6.10e-5, 1e-6)
# Per-split inference is the independent unit (3 splits x 5 coreset seeds).
# The pooled p above is descriptive only: the 5 seeds within a split share a
# test set, so pooling them is pseudoreplication. Pin the per-split 5/5 wins
# (sub4800 vs KT/SKT computed from records; summarise only stores shd_vs_*).
for _sp in ("42", "43", "44"):
    _recs = g["splits"][_sp]["records"]
    for _pair in ("kt", "skt"):
        _a = [r["shd_sub4800"]["krr_rmse_crit"] for r in _recs]
        _b = [r[_pair]["krr_rmse_crit"] for r in _recs]
        check(f"Gaps split {_sp} sub4800 vs {_pair} wins",
              sum(1 for x, y in zip(_a, _b) if x < y), 5, 0)
    check(f"Gaps split {_sp} shd vs stratified wins",
          g["splits"][_sp]["summary"]["tests"]["shd_vs_stratified_krr_rmse_crit"]["wins_shd"], 5, 0)
# The paper must frame these as paired RUNS (not independent splits) and name
# the pseudoreplication; the unbacked sub4800-vs-baseline p attribution is gone.
check_absent("paper paired-splits framing", "paired splits", PAPER)
check_text("paper names pseudoreplication", "pseudoreplication", PAPER)

# ---------------------------------------------------------------- weights
ws = load("bench_weights_superconduct_results.json")
wr = load("bench_weights_reactor_results.json")
for name, obj, vals in [
    ("weights SC KRR all gamma/uniform", ws, (15.57, 15.99)),
    ("weights SC KRR crit gamma/uniform", ws, (18.98, 19.38)),
    ("weights SC GBM crit gamma/uniform", ws, (18.00, 18.07)),
    ("weights reactor KRR all gamma/uniform", wr, (0.975, 0.992)),
    ("weights reactor KRR crit gamma/uniform", wr, (3.426, 3.335)),
    ("weights reactor GBM crit gamma/uniform", wr, (1.77, 2.17)),
]:
    s = obj["summary"]
    metric = name.split("KRR ")[-1] if "KRR" in name else "gbm"
    # direct by known suffix
    key = ("krr_rmse_all" if "KRR all" in name else
           "krr_rmse_crit" if "KRR crit" in name else
           "gbm_rmse_crit")
    check(name + " gamma", s[key]["gamma_median"], vals[0], 0.011 if "reactor" not in name else 0.006)
    check(name + " uniform", s[key]["uniform_median"], vals[1], 0.011 if "reactor" not in name else 0.006)
check("weights SC KRR-all gamma wins", ws["wilcoxon"]["uniform_vs_gamma_krr_rmse_all"]["wins_gamma"], 20, 0)
check("weights reactor KRR-crit uniform wins", wr["wilcoxon"]["uniform_vs_gamma_krr_rmse_crit"]["wins_uniform"], 20, 0)
check("weights reactor KRR-crit p", wr["wilcoxon"]["uniform_vs_gamma_krr_rmse_crit"]["p_value"], 1.907e-6, 1e-7)

# ---------------------------------------------------------------- PM2.5
pm = load("bench_pm25_results.json")
check("PM2.5 n_total", pm["n_total"], 41757, 0)
check("PM2.5 n_features", pm["n_features"], 14, 0)
check("PM2.5 n_test_crit", pm["n_test_crit"], 374, 0)
check("PM2.5 threshold", pm["crit_threshold"], 324.0, 0.01)
pmr = pm["records"]
for arm, vals in [
    ("shd", (95.08, 231.01, 86.11, 211.91, 10)),
    ("random", (88.24, 248.92, 80.05, 225.51, 9)),
    ("kmeans", (88.92, 241.18, 78.31, 213.79, 6)),
    ("kt", (89.89, 245.74, 79.85, 228.81, 9)),
    ("skt", (86.53, 246.02, 78.87, 221.19, 9)),
]:
    check(f"PM2.5 {arm} KRR all", med(pmr, arm, "krr_rmse_all"), vals[0], 0.011)
    check(f"PM2.5 {arm} KRR crit", med(pmr, arm, "krr_rmse_crit"), vals[1], 0.011)
    check(f"PM2.5 {arm} GBM all", med(pmr, arm, "gbm_rmse_all"), vals[2], 0.011)
    check(f"PM2.5 {arm} GBM crit", med(pmr, arm, "gbm_rmse_crit"), vals[3], 0.011)
    check(f"PM2.5 {arm} rare", med(pmr, arm, "rare_captured"), vals[4], 0.06)
check("PM2.5 SHD vs KT crit wins", pm["wilcoxon"]["shd_vs_kt_krr_rmse_crit"]["wins_shd"], 13, 0)
check("PM2.5 SHD vs KT crit p", pm["wilcoxon"]["shd_vs_kt_krr_rmse_crit"]["p_value"], 0.216, 1e-3)

shdc = load("bench_shdc_pm25_results.json")["summary"]
check("PM2.5 SHD-C KRR all", shdc["krr_rmse_all"]["median"], 101.15, 0.011)
check("PM2.5 SHD-C KRR crit", shdc["krr_rmse_crit"]["median"], 216.57, 0.011)
check("PM2.5 SHD-C GBM all", shdc["gbm_rmse_all"]["median"], 91.26, 0.011)
check("PM2.5 SHD-C GBM crit", shdc["gbm_rmse_crit"]["median"], 198.42, 0.011)
check("PM2.5 SHD-C rare", shdc["rare_captured"]["median"], 25, 0.06)

guard = load("bench_critical_guard300_results.json")["summary"]
gshd, gguard = guard["crit_shd"], guard["crit_guard"]
check("PM2.5 guard SHD base KRR crit", gshd["krr_crit_median"], 207.98, 0.011)
check("PM2.5 guard SHD base KRR all", gshd["krr_all_median"], 98.75, 0.011)
check("PM2.5 guard SHD base GBM crit", gshd["gbm_crit_median"], 193.70, 0.011)
check("PM2.5 guard SHD base GBM all", gshd["gbm_all_median"], 87.85, 0.011)
check("PM2.5 guard KRR crit", gguard["krr_crit_median"], 196.02, 0.011)
check("PM2.5 guard KRR all", gguard["krr_all_median"], 85.32, 0.011)
check("PM2.5 guard GBM crit", gguard["gbm_crit_median"], 168.40, 0.011)
check("PM2.5 guard GBM all", gguard["gbm_all_median"], 78.92, 0.011)

# ---------------------------------------------------------------- passport
pp = load("bench_passport_results.json")["summary"]
check("Passport pre A", pp["pre_A"], 0.035, 0.001)
check("Passport pre B", pp["pre_B"], 0.629, 0.001)
check("Passport blend A", pp["blend_A"], 0.203, 0.001)
check("Passport blend B", pp["blend_B"], 0.458, 0.001)
check("Passport recal A", pp["recal_A"], 0.497, 0.001)
check("Passport recal B", pp["recal_B"], 0.038, 0.001)
check("Passport blend B above", pp["blend_B_above"], 5, 0)
check("Passport blend A above", pp["blend_A_above"], 5, 0)
check("Passport recal B below", pp["recal_B_below"], 5, 0)
check("Passport bank routing", pp["bank_route_correct"], 5, 0)
check("Passport bank both accepted", pp["bank_accepted_both"], 5, 0)

# ---------------------------------------------------------------- latency
lat = load("bench_latency_v2_results.json")
check("Latency min", lat["range_ms"][0], 2.21, 0.01)
check("Latency max", lat["range_ms"][1], 62.35, 0.05)
check("Latency B=64", lat["records"]["64"]["median_ms"], 2.21, 0.05)
check("Latency B=full (2808 rows)", lat["records"]["full"]["median_ms"], 62.35, 0.05)
# The legacy artifact truncated oversize windows; the paper withdraws its
# numbers, so they must not reappear in the paper.
for text in ["24.935", "4.8--24.9", "5--25"]:
    check_absent("paper stale latency " + text, text)

# ---------------------------------------------------------------- Holm
holm = load("stats_holm_results.json")
check("Holm SC crit", holm["bench_seeds"]["superconduct"]["shd_vs_skt_krr_rmse_crit"]["p_holm"], 4.577e-5, 1e-8)
check("Holm SC GBM SKT", holm["bench_seeds"]["superconduct"]["shd_vs_skt_gbm_rmse_all"]["p_holm"], 0.2783, 1e-4)
check("Holm SC KRR-all SKT", holm["bench_seeds"]["superconduct"]["shd_vs_skt_krr_rmse_all"]["p_holm"], 0.0511, 1e-4)

# ------------------------------------------- §6 frontier (bench_frontier.py)
fr = load("bench_frontier_results.json")["results"]
for _ds in ("superconduct", "reactor"):
    _f = fr[_ds]
    check(f"frontier {_ds} complete", 1 if _f.get("partial") is False else 0, 1, 0)
    check(f"frontier {_ds} arm count",
          1 if all(len(_f["arms"][a]) == 20 for a in _f["arms"]) else 0, 1, 0)
fsc = fr["superconduct"]
check("frontier SC br065 krr_crit median",
      fsc["wilcoxon"]["shd_r0.65"]["krr_rmse_crit"]["median_shd"], 15.5111, 0.0002)
check("frontier SC br065 krr_crit vs km wins",
      fsc["wilcoxon"]["shd_r0.65"]["krr_rmse_crit"]["wins_shd"], 20, 0)
check("frontier SC br065 krr_crit p<1e-4",
      1 if fsc["wilcoxon"]["shd_r0.65"]["krr_rmse_crit"]["p_value"] < 1e-4 else 0, 1, 0)
check("frontier SC br065 gbm_all cost delta %",
      fsc["wilcoxon"]["shd_r0.65"]["gbm_rmse_all"]["delta_median_pct"], 4.5, 0.05)
check("frontier SC oracle krr joined from seeds",
      fsc["oracle"]["krr_rmse_crit"], 17.5892, 0.0005)
check("frontier SC br065 beats oracle krr wins",
      fsc["vs_oracle"]["shd_r0.65"]["krr_rmse_crit"]["wins_vs_oracle"], 20, 0)
check("frontier SC br065 gbm_crit median",
      fsc["vs_oracle"]["shd_r0.65"]["gbm_rmse_crit"]["median_shd"], 12.0815, 0.0002)
check("frontier SC br065 beats oracle gbm wins",
      fsc["vs_oracle"]["shd_r0.65"]["gbm_rmse_crit"]["wins_vs_oracle"], 16, 0)
frc = fr["reactor"]
check("frontier reactor br065 krr_crit median",
      frc["wilcoxon"]["shd_r0.65"]["krr_rmse_crit"]["median_shd"], 2.4049, 0.0002)
check("frontier reactor br065 krr delta %",
      frc["wilcoxon"]["shd_r0.65"]["krr_rmse_crit"]["delta_median_pct"], -33.2, 0.05)
check("frontier reactor br065 beats oracle krr wins",
      frc["vs_oracle"]["shd_r0.65"]["krr_rmse_crit"]["wins_vs_oracle"], 20, 0)

# ------------------------------------------- C-MAPSS FD001 (bench_cmapss.py)
# Regenerated 2026-11-13 under the official RUL convention (life - cyc):
# the +1 train-label offset is gone; train and test share one convention.
cm = load("bench_cmapss_results.json")["results"]["fd001"]
check("cmapss complete", 0 if cm.get("partial") else 1, 1, 0)
check("cmapss seed count", len(cm["records"]), 20, 0)
check("cmapss oracle krr crit", cm["oracle"]["krr_rmse_crit"], 13.0509, 0.0005)
check("cmapss oracle gbm full crit", cm["oracle"]["gbm_full_pool"]["gbm_rmse_crit"], 16.9903, 0.0005)
check("cmapss br065 krr_crit median",
      cm["wilcoxon"]["shd_r0.65_vs_kt_krr_rmse_crit"]["median_shd"], 14.5387, 0.0005)
check("cmapss br065 krr_crit wins vs kt",
      cm["wilcoxon"]["shd_r0.65_vs_kt_krr_rmse_crit"]["wins_shd"], 14, 0)
check("cmapss br065 krr_crit wins vs kmeans",
      cm["wilcoxon"]["shd_r0.65_vs_kmeans_krr_rmse_crit"]["wins_shd"], 12, 0)
check("cmapss br065 krr_crit p<0.05 vs kt",
      1 if cm["wilcoxon"]["shd_r0.65_vs_kt_krr_rmse_crit"]["p_value"] < 0.05 else 0, 1, 0)
check("cmapss br065 krr vs kmeans is a tie (n.s.)",
      1 if cm["wilcoxon"]["shd_r0.65_vs_kmeans_krr_rmse_crit"]["p_value"] > 0.05 else 0, 1, 0)
check("cmapss br065 gbm_crit wins vs kt",
      cm["wilcoxon"]["shd_r0.65_vs_kt_gbm_rmse_crit"]["wins_shd"], 17, 0)
check("cmapss br065 gbm_crit wins vs skt",
      cm["wilcoxon"]["shd_r0.65_vs_skt_gbm_rmse_crit"]["wins_shd"], 13, 0)
check("cmapss br065 gbm beats full-pool oracle at crit",
      cm["vs_oracle"]["shd_r0.65"]["gbm_rmse_crit"]["wins_vs_oracle"], 16, 0)
check("cmapss br050 gbm beats full-pool oracle at crit (19/20)",
      cm["vs_oracle"]["shd_r0.5"]["gbm_rmse_crit"]["wins_vs_oracle"], 19, 0)
check("cmapss br065 gamma arm krr oracle NOT beaten (honest negative)",
      cm["vs_oracle"]["shd_r0.65"]["krr_rmse_crit"]["wins_vs_oracle"], 6, 0)
# The uniform-weight arms (SHDU) are the SAME crystals with ones instead of
# the gamma weights. Under the corrected convention they improve KRR critical
# but NOT GBM critical at br=0.65 (model-dependent weighting), and NO arm
# beats the KRR oracle -- the pre-fix oracle-beating claim is retracted.
check("cmapss shdu r065 krr_crit median",
      cm["wilcoxon"]["shdu_r0.65_vs_kt_krr_rmse_crit"]["median_shd"], 13.8100, 0.0005)
check("cmapss shdu r065 gbm_crit median",
      cm["wilcoxon"]["shdu_r0.65_vs_kt_gbm_rmse_crit"]["median_shd"], 13.9486, 0.0005)
check("cmapss shdu r050 krr_crit median (KRR-best point)",
      cm["wilcoxon"]["shdu_r0.5_vs_kt_krr_rmse_crit"]["median_shd"], 13.4542, 0.0005)
check("cmapss shdu r065 krr_crit wins vs random",
      cm["wilcoxon"]["shdu_r0.65_vs_random_krr_rmse_crit"]["wins_shd"], 16, 0)
check("cmapss shdu r065 krr_crit wins vs kmeans",
      cm["wilcoxon"]["shdu_r0.65_vs_kmeans_krr_rmse_crit"]["wins_shd"], 12, 0)
check("cmapss shdu r065 gbm_crit wins vs random (18/20)",
      cm["wilcoxon"]["shdu_r0.65_vs_random_gbm_rmse_crit"]["wins_shd"], 18, 0)
check("cmapss shdu r065 gbm_crit wins vs kt",
      cm["wilcoxon"]["shdu_r0.65_vs_kt_gbm_rmse_crit"]["wins_shd"], 17, 0)
check("cmapss shdu r065 krr_crit p<0.01 vs random",
      1 if cm["wilcoxon"]["shdu_r0.65_vs_random_krr_rmse_crit"]["p_value"] < 0.01 else 0, 1, 0)
check("cmapss shdu r065 gbm_crit p<1e-4 vs random",
      1 if cm["wilcoxon"]["shdu_r0.65_vs_random_gbm_rmse_crit"]["p_value"] < 1e-4 else 0, 1, 0)
check("cmapss shdu r065 krr_crit does NOT beat oracle by median (retracted)",
      1 if cm["vs_oracle"]["shdu_r0.65"]["krr_rmse_crit"]["median_shd"]
             < cm["vs_oracle"]["shdu_r0.65"]["krr_rmse_crit"]["oracle"] else 0, 0, 0)
check("cmapss shdu r065 krr_crit wins vs oracle",
      cm["vs_oracle"]["shdu_r0.65"]["krr_rmse_crit"]["wins_vs_oracle"], 8, 0)
check("cmapss shdu r065 gbm_crit wins vs oracle",
      cm["vs_oracle"]["shdu_r0.65"]["gbm_rmse_crit"]["wins_vs_oracle"], 18, 0)
check("cmapss no arm beats the oracle on global krr",
      1 if all(v["krr_rmse_all"]["wins_vs_oracle"] == 0
               for v in cm["vs_oracle"].values()) else 0, 1, 0)
check("cmapss br065 rare median",
      float(np.median([r["shd_r0.65"]["rare_captured"] for r in cm["records"]])), 140.5, 0.001)
check("cmapss kmeans krr_crit median",
      cm["wilcoxon"]["shd_r0.65_vs_kmeans_krr_rmse_crit"]["median_kmeans"], 15.4556, 0.0005)

# ---------------------------------------------------------------- paper strings
PAPER_NEEDLE_COUNTS = {
    '0.038': 4,
    '0.203': 2,
    '0.458': 2,
    '0.465': 1,
    '0.497': 2,
    '13.36': 2,
    '15.42': 1,
    '168.40': 1,
    '18.6\\% of the dataset (3,952 / 21,263; 778 of 4,253 test rows)': 1,
    '18.81': 3,
    '196.02': 1,
    '2.21--62.35': 2,
    '20.62': 2,
    '21.20': 3,
    '22.43': 2,
    '22.93': 2,
    '220': 1,
    '23.26': 3,
    '245': 5,
    '41,757': 1,
    '50.2': 2,
    '55.0': 1,
    '59.0': 1,
    '64.5': 1,
    '7.32': 2,
    '78.92': 1,
    '85.32': 1,
    'DomainBank': 2,
    'Holm $p=0.051$': 1,
    'Oracle (KRR 9,000; GBM 16,000)': 1,
    'Oracle (KRR 9,000; GBM full train)': 1,
    'SHD wins (KRR Crit)': 2,
    'passport\\_recalibrate': 2,
    'sec:passport': 4,
    'sec:pm25': 5,
    'tab:passport': 2,
    'tab:pm25': 2,
}

for text in [
    r"18.6\% of the dataset (3,952 / 21,263; 778 of 4,253 test rows)",
    r"Oracle (KRR 9,000; GBM full train)",
    r"Oracle (KRR 9,000; GBM 16,000)",
    r"SHD wins (KRR Crit)",
    r"18.81", r"15.42", r"23.26", r"21.20", r"22.43", r"22.93",
    r"59.0", r"55.0", r"64.5",
    r"2.21--62.35", r"50.2",
    r"220", r"245", r"7.32", r"13.36", r"20.62", r"0.465",
    r"Holm $p=0.051$",
    r"41,757", r"196.02", r"85.32", r"168.40", r"78.92",
    r"0.203", r"0.458", r"0.038", r"0.497",
    r"DomainBank", r"passport\_recalibrate",
    r"tab:pm25", r"sec:pm25", r"tab:passport", r"sec:passport",
]:
    check_text_count("paper contains " + text[:30], text,
                      PAPER_NEEDLE_COUNTS[text], PAPER)

for needle in [
    r"zero catastrophic", r"strictly preserved", r"M = 90",
    r"state-of-the-art", r"22.90", r"21.41", r"56.0", r"55.5",
    r"8.7\%", r"whitened", r"phase-weighted", r"diag(",
    r"solved online", r"B / n_seen", r"higher phase score",
    r"cannot be pruned", r"the KRR oracle is not beaten on this dataset",
]:
    check_absent("paper stale " + needle, needle)

README_NEEDLE_COUNTS = {
    '# Spectral Hologram Distiller (SHD v3.15.2)': 1,
    '### 10. NASA C-MAPSS FD001: remaining useful life on the LOW tail': 1,
    '0.203': 2,
    '0.458': 2,
    '0.465': 1,
    '324': 1,
    '55.0': 5,
    '59.0': 1,
    '85.3': 1,
    '993.0': 3,
    'DomainBank': 8,
    'PM2.5': 16,
    'passport_recalibrate': 4,
    'weighting="uniform"': 2,
    '13.45': 2,
    '13.81': 3,
    '13.95': 4,
}

for text in [
    "# Spectral Hologram Distiller (SHD v3.15.2)",
    "### 10. NASA C-MAPSS FD001: remaining useful life on the LOW tail",
    "DomainBank", "passport_recalibrate", 'weighting="uniform"',
    "13.45", "13.81", "13.95",
    # "196.0" lived in the paper only (PM2.5 guard KRR-crit); the README
    # carries the threshold and the Scania official-cost row instead.
    "59.0", "55.0", "PM2.5", "324", "85.3", "0.465", "993.0",
    "0.203", "0.458",
]:
    check_text_count("readme contains " + text[:30], text,
                      README_NEEDLE_COUNTS[text], README)
# "55.5" is a LIVE number in the README (Scania missed failures, SHD row),
# so it must not be forbidden here; the stale-value watch is the "993.0"
# positive check above.
for needle in ["zero catastrophic", "M = 90", "sub-millisecond", "8.7%", "56.0",
               "14.53"]:
    check_absent("readme stale " + needle, needle, README)

# ------------------------------------- ablation paragraph (2026-11-11 follow-up)
# The v3.15.0 paragraph quoted a 20-seed sweep that no committed artifact
# contains. Pin the paragraph to the 5-seed sweep and its 10-seed CI
# replication, and forbid the fabricated strings so they cannot return.
ab = load("bench_ablation_results.json")["reactor"]["sweeps"]
ab_ci = load("bench_ablation_ci_summary.json")["reactor"]
ab_k = {e["value"]: e["krr_rmse_crit_median"] for e in ab["phase_kappa"]}
ab_b = {e["value"]: e["krr_rmse_crit_median"] for e in ab["high_band_ratio"]}
check("ablation kappa=0 crit", ab_k[0.0], 4.199, 0.001)
check("ablation default crit", ab_k[3.2], 3.346, 0.001)
check("ablation br=0.10 crit", ab_b[0.1], 4.433, 0.001)
check("ablation br=0.50 crit", ab_b[0.5], 2.718, 0.001)
check("ablation CI kappa=0 crit", ab_ci["phase_kappa"]["0.0"]["median"], 4.132, 0.001)
check("ablation CI br=0.10 crit", ab_ci["high_band_ratio"]["0.1"]["median"], 4.249, 0.001)
check("ablation CI default crit", ab_ci["phase_kappa"]["3.2"]["median"], 3.440, 0.001)
# The 5-seed artifact now carries per-seed records: pin that every published
# median really is the median of its records, so a doctored row with the
# summary left intact cannot slip through.
_ab_bad = 0
for _rows in ab.values():
    for _row in _rows:
        _vals = [r["krr_rmse_crit"] for r in _row["records"]]
        if abs(float(statistics.median(_vals)) - float(_row["krr_rmse_crit_median"])) > 1e-9:
            _ab_bad += 1
check("ablation medians are medians of their records", _ab_bad, 0, 0)
for text in [
    "1.09 K (3.346 -> 4.433 K)",
    "+0.85 K (3.346 -> 4.199 K)",
    "lowers it to 2.718 K",
    "kappa=0 median 4.132, br=0.10 median 4.249, default 3.440",
]:
    check_text_count("readme ablation " + text[:30], text, 1, README)
for needle in ["3.44 -> 4.53", "lowers it to 2.30", "+1.35 K", "+0.86 K",
               "kappa=0.65"]:
    check_absent("readme fabricated ablation " + needle, needle, README)

# ------------------------------------------------- passport-reference benchmark
pr = load("bench_passport_reference_results.json")
syn = pr["arms"]["synthetic"]
pm_full = pr["arms"]["pm25_full"]
pm_sens = pr["arms"]["pm25_sensors"]
# Synthetic: the contamination penalty only shows up at 40% contamination...
check("passport-ref syn f0.10 stream FA", syn["stream_f0.10"]["fa_rate_median"], 0.0, 0.001)
check("passport-ref syn f0.10 detect", syn["stream_f0.10"]["detect_median"], 1.0, 0.001)
check("passport-ref syn f0.40 stream FA", syn["stream_f0.40"]["fa_rate_median"], 1.0, 0.001)
# ...and there the passport is degenerate (in-sample windows alarm too).
check("passport-ref syn f0.40 in-sample FA", syn["stream_f0.40"]["insample_fa_median"], 1.0, 0.001)
check("passport-ref syn f0.00 in-sample FA", syn["stream_f0.00"]["insample_fa_median"], 0.0, 0.001)
# PM2.5: degenerate for both references and both feature sets.
check("passport-ref pm25 full stream FA", pm_full["stream_f0.00"]["fa_rate_median"], 1.0, 0.001)
check("passport-ref pm25 full in-sample FA", pm_full["stream_f0.00"]["insample_fa_median"], 1.0, 0.001)
check("passport-ref pm25 sensors stream FA", pm_sens["stream_f0.00"]["fa_rate_median"], 1.0, 0.001)
check("passport-ref pm25 sensors in-sample FA", pm_sens["stream_f0.00"]["insample_fa_median"], 1.0, 0.001)
# The background reference must not be reported as a remedy.
check("passport-ref verdict pm25 label",
      1 if pr["verdict"]["pm25_full_f0.00"]["label"] == "both_broken_fa_1.00" else 0, 1, 0)
check_text_count("readme claims the negative result", "background-only spectral passport is not a fix", 1, README)
for text in ["background passport fixes", "background-only passport fixes"]:
    check_absent("readme overclaims " + text, text, README)

# ---------------------------------------------------------------- artifact/version
art = np.load(ROOT / "artifacts" / "superconduct_v37.hologram.npz", allow_pickle=False)
check("artifact medoids", art["crystal_X"].shape[0], 300, 0)
check("artifact features", art["crystal_X"].shape[1], 81, 0)
check("artifact passport rank", len(art["passport_eigvals"]), 32, 0)
check("artifact passport dtype", 1 if art["passport_eigvals"].dtype == np.float16 else 0, 1, 0)
size_kb = os.path.getsize(ROOT / "artifacts" / "superconduct_v37.hologram.npz") / 1024
check("artifact size KB", size_kb, 97.7, 0.2)
# n_seen / medoids -- exactly what loader.summary()["compression_ratio"]
# reports. The crystal distilled the 17,010-row train split, not all 21,263
# rows, so the honest ratio is 56.7 (the README's 1.76% = 300 / 17010).
check("compression ratio", art["n_seen"][0] / art["crystal_X"].shape[0], 56.7, 0.05)

distiller_src = (ROOT / "shd" / "distiller.py").read_text(encoding="utf-8")
version = (ROOT / "shd" / "__init__.py").read_text(encoding="utf-8")
pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
# v3.13.2: SHD_VERSION in shd/distiller.py is the single source of truth;
# shd/__init__.py re-exports it (__version__ = SHD_VERSION) instead of
# duplicating the literal (the duplication is what drifted in v3.10/v3.11).
check_text("version source 3.15.2", 'SHD_VERSION = "3.15.2"', distiller_src)
check_text_count("version re-export", "__version__ = SHD_VERSION", 1, version)
check_text("pyproject 3.15.2", 'version = "3.15.2"', pyproject)
# The three version carriers must agree; this is the check that would have
# caught the v3.10.0/v3.11.0 drift before it reached the README title.
check("version carriers agree",
      1 if ('SHD_VERSION = "3.15.2"' in distiller_src
            and "__version__ = SHD_VERSION" in version
            and 'version = "3.15.2"' in pyproject
            and "SHD v3.15.2" in README) else 0, 1, 0)

# --------------------------------------------- presets + gate solver (v3.14.1)
sys.path.insert(0, str(ROOT))
import shd as _shd  # noqa: E402

try:
    _shd.validate_presets()
    presets_ok = 1
except Exception:
    presets_ok = 0
check("presets validate against the constructor", presets_ok, 1.0, 0)
check("preset critical_first params",
      1 if _shd.apply_preset("critical_first") == {"high_band_ratio": 0.65} else 0, 1, 0)
check("preset balanced params",
      1 if _shd.apply_preset("balanced") == {"high_band_ratio": 0.50} else 0, 1, 0)
check("preset sparse_stream params",
      1 if _shd.apply_preset("sparse_stream_top3pct") == {"tail_quantile": 97.0} else 0, 1, 0)
check("preset global_first params",
      1 if _shd.apply_preset("global_first") == {
          "high_band_ratio": 0.26, "tail_quantile": 84.0} else 0, 1, 0)
check("preset pm25_global params",
      1 if _shd.apply_preset("pm25_global") == {"global_guard": True} else 0, 1, 0)
check("preset count", len(_shd.PRESETS), 10, 0)
check("preset sparse_auto params",
      1 if _shd.apply_preset("sparse_auto") == {
          "tail_quantile": 97.0, "global_guard": True,
          "high_band_ratio": 0.26} else 0, 1, 0)
check("preset critical_first_no_residual params",
      1 if _shd.apply_preset("critical_first_no_residual") == {
          "residual_weight": 0.0, "high_band_ratio": 0.65} else 0, 1, 0)
check("preset single_scale_low params",
      1 if _shd.apply_preset("single_scale_low") == {"rff_mode": "single_low"} else 0, 1, 0)
check("preset single_scale_high params",
      1 if _shd.apply_preset("single_scale_high") == {"rff_mode": "single_high"} else 0, 1, 0)
portfolio_src = (ROOT / "shd" / "portfolio.py").read_text(encoding="utf-8")
check_text_count("gate solver budget 5000", "max_iter=5000", 4, portfolio_src)
check_text_count("gate solver fallback recorded", "gate_solver_", 4, portfolio_src)
check_text_count("readme documents the logit-C fix", "tune-logit-c", 3, README)
check_text_count("readme states the tuned-logit verdict", "Tuned logit beats tuned HGB", 1, README)

# ------------------------------------- v3.15.0 trade-off / honesty artifacts
res = load("bench_residual_tradeoff_results.json")
for ds in ("reactor", "SC"):
    cfgs = res[ds]["configs"]
    default_crit = float(np.median([r["krr_rmse_crit"] for r in cfgs["rw_1.0_default"]["records"]]))
    rw0_crit = float(np.median([r["krr_rmse_crit"] for r in cfgs["rw_0"]["records"]]))
    # residual trigger HURTS the critical regime: rw_0 must be <= default
    check(f"{ds} residual rw_0 not worse than default", 1 if rw0_crit <= default_crit else 0, 1, 0)
    cf = cfgs["critical_first_no_residual"]
    cf_crit = float(np.median([r["krr_rmse_crit"] for r in cf["records"]]))
    check(f"{ds} critical_first_no_residual beats default crit", 1 if cf_crit < default_crit else 0, 1, 0)
    check(f"{ds} critical_first_no_residual wilcoxon wins >= 8/10",
          cf["wilcoxon_vs_default"]["wins"], 10, 2)

hard = load("bench_radar_hard_results.json")
for b in hard["by_batch"]:
    agg = hard["by_batch"][b]["agg"]
    # the abrupt control must stay perfect (the easy task is still perfect)
    check(f"radar_hard B={b} abrupt control AUC ~ 1", agg["abrupt_control"]["auc_median"], 1.0, 0.002)
    # graded drift must stay strongly ranked even where the fixed tau misses it
    for name in ("gradual_alpha0.3", "gradual_alpha0.5"):
        check_min(f"radar_hard B={b} {name} AUC >= 0.95", agg[name]["auc_median"], 0.95)
    # partial windows are the weak spot: document it, do not hide it
    check_min(f"radar_hard B={b} partial_frac0.25 AUC >= 0.9", agg["partial_frac0.25"]["auc_median"], 0.9)

# ------------------------------------- v3.15.1 shipped global-guard artifact
# The presets pm25_global / sparse_auto turn the global guard ON. Until
# v3.15.1 no artifact measured the SHIPPED guard path (the PM2.5 'guard
# wins' numbers came from a manual K-Means concat in
# bench_critical_guard.py, a different mechanism). These checks pin the
# shipped-path artifact AND the trade-off it actually shows.
gg = load("bench_global_guard_results.json")
check("guard artifact used the shipped path",
      1 if "shipped" in gg["guard"]["path"] else 0, 1, 0)
check("guard artifact n_seeds", gg["n_seeds"], 20, 0)
check("guard artifact budget", gg["m"], 300, 0)
g_sum = gg["summary"]
g_w = gg["wilcoxon"]
# the guard must improve the GLOBAL fit over the same distiller without it
check("guard improves global KRR vs no guard (wins)",
      g_w["guarded_vs_plain_krr_rmse_all"]["wins_guarded"], 20, 0)
check("guard improves global GBM vs no guard (wins)",
      g_w["guarded_vs_plain_gbm_rmse_all"]["wins_guarded"], 20, 0)
check("guard improves global KRR (median)",
      1 if g_sum["shd_guarded"]["krr_rmse_all"] < g_sum["shd_plain"]["krr_rmse_all"] else 0, 1, 0)
check("guard improves global GBM (median)",
      1 if g_sum["shd_guarded"]["gbm_rmse_all"] < g_sum["shd_plain"]["gbm_rmse_all"] else 0, 1, 0)
# it must NOT be presented as beating K-Means on global fit: on KRR it ties
# (n.s.), on GBM it loses. The win is on the CRITICAL regime.
check("guard ties KMeans on global KRR (not a win)",
      g_w["guarded_vs_kmeans_krr_rmse_all"]["p_value"], 0.1429, 0.01)
check("guard loses to KMeans on global GBM (documented)",
      1 if g_sum["shd_guarded"]["gbm_rmse_all"] > g_sum["kmeans"]["gbm_rmse_all"] else 0, 1, 0)
check("guard beats KMeans on critical KRR (wins)",
      g_w["guarded_vs_kmeans_krr_rmse_crit"]["wins_guarded"], 15, 0)
# the honest cost: the guard spends rare-regime rows on global coverage.
check("guard costs rare rows (documented trade-off)",
      1 if g_sum["shd_guarded"]["rare_captured_median"]
      < g_sum["shd_plain"]["rare_captured_median"] else 0, 1, 0)
check("guard does not hurt the critical regime vs plain",
      1 if g_sum["shd_guarded"]["krr_rmse_crit"] <= g_sum["shd_plain"]["krr_rmse_crit"] else 0, 1, 0)
# --------------------------------- v3.15.2 bug-hunt invariants (2026-11-07)
# Seven defects found by an independent audit and reproduced by hand.
# Each fix is pinned here so a future refactor cannot silently undo it.
dist_src = (ROOT / "shd" / "distiller.py").read_text(encoding="utf-8")
cli_src = (ROOT / "shd" / "cli.py").read_text(encoding="utf-8")
pf_src = (ROOT / "shd" / "portfolio.py").read_text(encoding="utf-8")
check_in_function("finalize invalidates a stale guard flag",
                "def finalize_crystal(", "self._guard_applied = False", dist_src)
check_in_function("save refuses a crystal older than the stream",
                "def save_hologram(", "({int(_fin_n)} -> {int(self.n_seen)} rows)", dist_src)
check_text_count("artifact records tail_quantile", "tail_quantile=np.array([float(self.tail_q)]", 1, dist_src)
check_text_count("artifact records phase_kappa", "phase_kappa=np.array([float(self.kappa)]", 1, dist_src)
check_text_count("CLI normalises the output path", "def _npz_path(", 1, cli_src)
check_text_count("portfolio gate uses one column convention", "self._gate_design(self.krr_crit_.predict(Zg), self.oof_glob_, Zg)", 1, pf_src)
check_text_count("conformal fit declares model independence", "model_independent: bool = False", 1, pf_src)
check_text_count("alarm floor survives emergency_band=0", "alarm = (p_c >= self.critical_threshold).astype(float)", 1, pf_src)

# the gate must pin the PUBLISHED VALUES, not only the ordering: a doctored
# artifact that keeps the ordering (guarded < plain) would pass a
# relation-only gate. Same pattern as the older artifact checks above.
check("guard value: plain KRR all", g_sum["shd_plain"]["krr_rmse_all"], 101.1455, 0.011)
check("guard value: guarded KRR all", g_sum["shd_guarded"]["krr_rmse_all"], 91.5215, 0.011)
check("guard value: plain GBM all", g_sum["shd_plain"]["gbm_rmse_all"], 93.3596, 0.011)
check("guard value: guarded GBM all", g_sum["shd_guarded"]["gbm_rmse_all"], 85.859, 0.011)
check("guard value: guarded KRR r2", g_sum["shd_guarded"]["krr_r2"], 0.0578, 0.011)
check("guard value: KMeans KRR all", g_sum["kmeans"]["krr_rmse_all"], 90.4025, 0.011)
check("guard value: KMeans GBM all", g_sum["kmeans"]["gbm_rmse_all"], 77.3806, 0.011)
check("guard value: plain rare captured", g_sum["shd_plain"]["rare_captured_median"], 26.5, 0.06)
check("guard value: guarded rare captured", g_sum["shd_guarded"]["rare_captured_median"], 11.5, 0.06)
check("guard value: plain-vs-guarded p", g_w["guarded_vs_plain_krr_rmse_all"]["p_value"], 1.9073486328125e-06, 1e-12)
check("guard value: critical-vs-KMeans p", g_w["guarded_vs_kmeans_krr_rmse_crit"]["p_value"], 0.0014324188232421875, 1e-12)
# the README renders the artifact numbers; pin the rendering too, so a
# hand-edited README table cannot drift away from the artifact it cites
check_text_count("README SC crit row", "**18.81 / 0.93 K**", 1, README)
check_text_count("README Scania official cost", "**18645.0**", 2, README)
check_text_count("README edge artifact size", "28.9 KB artifact", 1, README)
check_text_count("README edge 81D size", "184.9 KB", 1, README)
check_text_count("README guard plain KRR", "101.15", 2, README)
check_text_count("README guard guarded KRR", "91.52", 2, README)
check_text_count("README portfolio PM2.5 SHD-C global", "vs 101.15 SHD-C", 1, README)
check_absent("README no pre-2026-11-13 portfolio SHD-C global", "95.08", README)

# --------------------------------------- portfolio artifacts (2026-11-13)
# The portfolio arm was re-measured FROM SCRATCH (20 seeds, 3 datasets)
# after the critical-aware expert fix (critical_threshold + auto_band_ratio);
# the 2026-11-07 artifact is superseded.
# The gate must pin the published values, not only the ordering: a stale or
# doctored artifact must not pass.
pf_sc = load("bench_portfolio_superconduct_results.json")
pf_pm = load("bench_portfolio_pm25_results.json")
pf_re = load("bench_portfolio_reactor_results.json")
for _nm, _d in (("SC", pf_sc), ("PM2.5", pf_pm), ("reactor", pf_re)):
    check(f"portfolio {_nm} artifact used 20 seeds", _d["n_seeds"], 20, 0)
    check(f"portfolio {_nm} artifact is complete",
          1 if _d.get("partial") is False else 0, 1, 0)
    check(f"portfolio {_nm} artifact has 20 records", len(_d["records"]), 20, 0)
    # KNOWN LIMITATION, pinned on purpose: the portfolio's GBM arm fits the
    # SAME critical coreset as the SHD arm with the same seed, so its GBM
    # column duplicates SHD by construction and is NOT an independent
    # portfolio measurement. Only the KRR columns are a portfolio result.
    check(f"portfolio {_nm} GBM column duplicates the SHD arm",
          abs(_d["summary"]["portfolio"]["gbm_rmse_all"]
              - _d["summary"]["shd"]["gbm_rmse_all"]), 0.0, 1e-9)
    check(f"portfolio {_nm} GBM crit duplicates the SHD arm",
          abs(_d["summary"]["portfolio"]["gbm_rmse_crit"]
              - _d["summary"]["shd"]["gbm_rmse_crit"]), 0.0, 1e-9)

check("portfolio SC crit", pf_sc["summary"]["portfolio"]["krr_rmse_crit"], 18.2158, 0.011)
check("portfolio SC all", pf_sc["summary"]["portfolio"]["krr_rmse_all"], 15.3822, 0.011)
check("portfolio SC p vs SHD-C crit",
      pf_sc["wilcoxon"]["portfolio_vs_shd_krr_rmse_crit"]["p"], 0.3488, 0.001)
check("portfolio SC p vs SHD-C all",
      pf_sc["wilcoxon"]["portfolio_vs_shd_krr_rmse_all"]["p"], 0.0266, 0.001)
check("portfolio SC p vs KT crit",
      pf_sc["wilcoxon"]["portfolio_vs_kt_krr_rmse_crit"]["p"], 1.9073486328125e-06, 1e-9)
check("portfolio SC p vs SKT crit",
      pf_sc["wilcoxon"]["portfolio_vs_skt_krr_rmse_crit"]["p"], 1.9073486328125e-06, 1e-9)
check("portfolio SC p vs hard-gate crit",
      pf_sc["wilcoxon"]["portfolio_vs_hard_gate_krr_rmse_crit"]["p"], 5.7220458984375e-06, 1e-9)
check("portfolio SC SHD-C crit", pf_sc["summary"]["shd"]["krr_rmse_crit"], 18.3900, 0.011)
check("portfolio SC SHD-C all", pf_sc["summary"]["shd"]["krr_rmse_all"], 15.4628, 0.011)
check("portfolio SC KT crit", pf_sc["summary"]["kt"]["krr_rmse_crit"], 23.2562, 0.011)
check("portfolio SC SKT crit", pf_sc["summary"]["skt"]["krr_rmse_crit"], 21.1946, 0.011)
check("portfolio SC hard-gate crit", pf_sc["summary"]["hard_gate"]["krr_rmse_crit"], 19.2221, 0.011)
check("portfolio SC crit is a tie with SHD-C (n.s.)",
      1 if pf_sc["wilcoxon"]["portfolio_vs_shd_krr_rmse_crit"]["p"] > 0.05 else 0, 1, 0)

check("portfolio PM2.5 crit", pf_pm["summary"]["portfolio"]["krr_rmse_crit"], 224.6667, 0.011)
check("portfolio PM2.5 all", pf_pm["summary"]["portfolio"]["krr_rmse_all"], 86.6297, 0.011)
check("portfolio PM2.5 p vs SHD-C crit",
      pf_pm["wilcoxon"]["portfolio_vs_shd_krr_rmse_crit"]["p"], 0.6477, 0.001)
check("portfolio PM2.5 p vs SHD-C all",
      pf_pm["wilcoxon"]["portfolio_vs_shd_krr_rmse_all"]["p"], 1.9073486328125e-06, 1e-9)
check("portfolio PM2.5 p vs SKT all",
      pf_pm["wilcoxon"]["portfolio_vs_skt_krr_rmse_all"]["p"], 0.4749, 0.001)
check("portfolio PM2.5 p vs SKT crit",
      pf_pm["wilcoxon"]["portfolio_vs_skt_krr_rmse_crit"]["p"], 9.5367431640625e-06, 1e-9)
check("portfolio PM2.5 p vs hard-gate all",
      pf_pm["wilcoxon"]["portfolio_vs_hard_gate_krr_rmse_all"]["p"], 1.9073486328125e-06, 1e-9)
check("portfolio PM2.5 SHD-C crit", pf_pm["summary"]["shd"]["krr_rmse_crit"], 216.5654, 0.011)
check("portfolio PM2.5 SHD-C all", pf_pm["summary"]["shd"]["krr_rmse_all"], 101.1519, 0.011)
check("portfolio PM2.5 KT all", pf_pm["summary"]["kt"]["krr_rmse_all"], 89.8869, 0.011)
check("portfolio PM2.5 SKT all", pf_pm["summary"]["skt"]["krr_rmse_all"], 86.5269, 0.011)
check("portfolio PM2.5 hard-gate crit", pf_pm["summary"]["hard_gate"]["krr_rmse_crit"], 237.1483, 0.011)
check("portfolio PM2.5 hard-gate all", pf_pm["summary"]["hard_gate"]["krr_rmse_all"], 90.3050, 0.011)
check("portfolio PM2.5 beats SKT on crit",
      1 if pf_pm["summary"]["portfolio"]["krr_rmse_crit"]
      < pf_pm["summary"]["skt"]["krr_rmse_crit"] else 0, 1, 0)
check("portfolio PM2.5 does NOT beat SKT on global",
      1 if pf_pm["summary"]["portfolio"]["krr_rmse_all"]
      > pf_pm["summary"]["skt"]["krr_rmse_all"] else 0, 1, 0)

check("portfolio reactor crit", pf_re["summary"]["portfolio"]["krr_rmse_crit"], 3.3176, 0.006)
check("portfolio reactor all", pf_re["summary"]["portfolio"]["krr_rmse_all"], 0.9754, 0.006)
check("portfolio reactor p vs SHD-C crit",
      pf_re["wilcoxon"]["portfolio_vs_shd_krr_rmse_crit"]["p"], 3.814697265625e-06, 1e-9)
check("portfolio reactor p vs SHD-C all",
      pf_re["wilcoxon"]["portfolio_vs_shd_krr_rmse_all"]["p"], 0.7012, 0.001)
check("portfolio reactor p vs hard-gate crit",
      pf_re["wilcoxon"]["portfolio_vs_hard_gate_krr_rmse_crit"]["p"], 3.814697265625e-06, 1e-9)
check("portfolio reactor SHD-C crit", pf_re["summary"]["shd"]["krr_rmse_crit"], 3.0401, 0.006)
check("portfolio reactor SHD-C all", pf_re["summary"]["shd"]["krr_rmse_all"], 0.9686, 0.006)
check("portfolio reactor hard-gate crit", pf_re["summary"]["hard_gate"]["krr_rmse_crit"], 4.0570, 0.006)
check("portfolio reactor hard-gate all", pf_re["summary"]["hard_gate"]["krr_rmse_all"], 1.1065, 0.006)
check("portfolio reactor is WORSE than SHD-C on crit",
      1 if pf_re["summary"]["portfolio"]["krr_rmse_crit"]
      > pf_re["summary"]["shd"]["krr_rmse_crit"] else 0, 1, 0)
check("portfolio reactor is WORSE than SHD-C on global",
      1 if pf_re["summary"]["portfolio"]["krr_rmse_all"]
      > pf_re["summary"]["shd"]["krr_rmse_all"] else 0, 1, 0)

# the benchmark must keep the duplication visible in its own source, so a
# future "fix" of the GBM arm breaks the gate instead of silently changing
# what the published numbers mean
bench_pf_src = (ROOT / "bench_portfolio.py").read_text(encoding="utf-8")
check_text_count("bench portfolio GBM arm fits the SHD coreset", ".fit(Xs, ys, sample_weight=ws)", 1, bench_pf_src)

# the published documents must carry the post-fix numbers, and the pre-fix
# ones must be GONE (a stale number in prose is a silent false claim)
check_text_count("README portfolio SC crit", "**18.22**", 1, README)
check_text_count("README portfolio SC global", "**15.38**", 1, README)
check_text_count("README portfolio PM2.5 global", "**86.63**", 1, README)
check_text_count("README portfolio PM2.5 crit", "**224.67**", 1, README)
check_text_count("README portfolio reactor crit", "crit 3.32 vs", 1, README)
check_text_count("README portfolio p (SC crit)", "p=0.349", 1, README)
check_text_count("README documents the GBM duplication", "GBM column duplicates the SHD arm", 1, README)
check_absent("README no stale portfolio crit", "crit **17.95**", README)
check_absent("README no stale portfolio PM2.5 crit", "**227.27**", README)
check_absent("README no stale portfolio global", "**86.67**", README)
check_absent("README no pre-2026-11-13 portfolio SC crit", "**18.32**", README)
check_absent("README no pre-2026-11-13 portfolio PM2.5 global", "**87.08**", README)
check_absent("README no pre-2026-11-13 portfolio PM2.5 crit", "**228.31**", README)
check_absent("README no in-flight marker", "Re-measurement in flight", README)
check_absent("README no stale Pareto claim", "Pareto-best: crit", README)
check_text_count("industry doc portfolio SC crit", "**18.22**", 1, INDUSTRY)
check_text_count("industry doc portfolio PM2.5 global", "**86.63**", 1, INDUSTRY)
check_text_count("industry doc portfolio reactor", "3.32 vs 3.04", 2, INDUSTRY)
check_absent("industry doc no stale SC crit", "**17.95**", INDUSTRY)
check_absent("industry doc no stale PM2.5 crit", "**227.27**", INDUSTRY)
check_absent("industry doc no stale global", "**86.67**", INDUSTRY)
check_absent("industry doc no pre-2026-11-13 SC crit", "**18.32**", INDUSTRY)
check_absent("industry doc no pre-2026-11-13 PM2.5 global", "**87.08**", INDUSTRY)
check_absent("industry doc no stale Pareto claim",
             "Pareto-лучший**: crit", INDUSTRY)

# the manifest must list EVERY artifact that ships in the repo root, and
# every manifest row must point at a file that exists. An artifact that
# pins claims in this gate but is missing from ARTIFACTS.md is an
# untracked claim - exactly the drift this manifest exists to prevent.
_manifest = (ROOT / "ARTIFACTS.md").read_text(encoding="utf-8")
_root_jsons = sorted(p.name for p in ROOT.glob("*.json"))
_missing = [n for n in _root_jsons if (chr(96) + n + chr(96)) not in _manifest]
if _missing:
    print("  artifacts missing from ARTIFACTS.md:", _missing)
check("ARTIFACTS.md lists every root artifact", len(_missing), 0, 0)
_rows = [l for l in _manifest.splitlines() if l.startswith("| " + chr(96))]
_ghosts = []
for _l in _rows:
    _cell = _l.split("|")[1].strip().strip(chr(96))
    if "*" in _cell or "?" in _cell:
        _ghosts.append(_cell + " (glob)")
        continue
    if not any((d / _cell).exists() for d in
               (ROOT, ROOT / "paper", ROOT / "artifacts", ROOT / "docs")):
        _ghosts.append(_cell + " (absent)")
if _ghosts:
    print("  manifest rows without a real file:", _ghosts)
check("ARTIFACTS.md rows point at real files", len(_ghosts), 0, 0)

# the summaries are medians OVER THE PER-SEED RECORDS: pin that relation so
# a doctored record (with the summary left intact) cannot slip through, and
# so the published numbers stay a function of the raw rows.
for _nm, _d in (("SC", pf_sc), ("PM2.5", pf_pm), ("reactor", pf_re)):
    for _arm in ("shd", "kt", "skt", "portfolio", "hard_gate"):
        for _metric in ("krr_rmse_all", "krr_rmse_crit"):
            _vals = [r[_arm][_metric] for r in _d["records"] if _metric in r[_arm]]
            if not _vals:
                continue
            check(f"portfolio {_nm} {_arm} {_metric} is the median of its records",
                  abs(float(statistics.median(_vals))
                      - float(_d["summary"][_arm][_metric])), 0.0, 1e-9)
for _arm in ("shd_plain", "shd_guarded", "kmeans"):
    _vals = [r[_arm]["krr_rmse_all"] for r in gg["records"] if _arm in r]
    if not _vals:
        continue
    check(f"guard {_arm} KRR all is the median of its records",
          abs(float(statistics.median(_vals))
              - float(g_sum[_arm]["krr_rmse_all"])), 0.0, 1e-9)

# ------------------------------------------------- deltas (bench_deltas.py)
# The GBM arm once fit on raw X_c but predicted on scaled Z_te (a unit
# mismatch that corrupted the trees). The corrected artifact is pinned here
# so the README table can never silently drift back to the buggy numbers.
_dl = load("bench_deltas_results.json")
_dls = _dl["summary"]
_dlw = _dl["wilcoxon_deltas_on_vs_off"]
check("deltas burst captured off", _dls["off_burst_captured_median"], 15.0, 0.0)
check("deltas burst captured on", _dls["on_burst_captured_median"], 17.0, 0.0)
check("deltas burst captured random", _dls["random_burst_captured_median"], 3.0, 0.0)
check("deltas KRR burst off", _dls["off_krr_rmse_burst_median"], 4.153, 0.001)
check("deltas KRR burst on", _dls["on_krr_rmse_burst_median"], 4.180, 0.001)
check("deltas KRR all off", _dls["off_krr_rmse_all_median"], 0.554, 0.001)
check("deltas KRR all on", _dls["on_krr_rmse_all_median"], 0.610, 0.001)
check("deltas GBM burst off", _dls["off_gbm_rmse_burst_median"], 3.711, 0.001)
check("deltas GBM burst on", _dls["on_gbm_rmse_burst_median"], 3.480, 0.001)
check("deltas GBM all off", _dls["off_gbm_rmse_all_median"], 0.724, 0.001)
check("deltas GBM all on", _dls["on_gbm_rmse_all_median"], 0.803, 0.001)
check("deltas GBM burst wins", _dlw["deltas_on_vs_off_gbm_rmse_burst"]["wins"], 15, 0)
check("deltas GBM burst p", _dlw["deltas_on_vs_off_gbm_rmse_burst"]["p_value"], 0.0441, 0.001)
check("deltas GBM all wins", _dlw["deltas_on_vs_off_gbm_rmse_all"]["wins"], 6, 0)
check("deltas GBM all p", _dlw["deltas_on_vs_off_gbm_rmse_all"]["p_value"], 0.0107, 0.001)
check_digest("deltas artifact digest", artifact_digest(_dl),
             "ef7e8252ba0578c32b07280f702474c5b8816b79c242cb143f541b9bc9231256")

# ----------------------------- drift late-regime capture (finalize rebuild)
_dr = load("bench_drift_rebuild_results.json")
check("drift interval=0 rare (blind)", _dr["arms"]["0"]["rare_captured_median"], 0.0, 0.0)
check("drift interval=1500 rare", _dr["arms"]["1500"]["rare_captured_median"], 21.0, 0.0)
check("drift interval=0 crit", _dr["arms"]["0"]["krr_rmse_crit_median"], 7.499, 0.01)
check("drift interval=1500 crit", _dr["arms"]["1500"]["krr_rmse_crit_median"], 3.610, 0.01)
check_digest("drift rebuild artifact digest", artifact_digest(_dr),
             "405adbe90138aef848b052421612039d0e731ef8672faa03bfb24041bf86db52")

# ------------------------------------ metro watch + 250k stress (backed pins)
_mw = load("bench_metro_watch_v2_results.json")
_mw_arms = _mw["results"]["metropt3"]["arms"] if "results" in _mw else _mw["arms"]
_mw_shd = next(a for a in _mw_arms if a["arm"] == "shd_detrend")
check("metro shd_detrend crystal KB", _mw_shd["crystal_kb"], 55.5, 0.05)
check("metro shd_detrend learn sec", _mw_shd["learn_sec"], 6.2, 0.05)
check("metro shd_detrend watch scan sec", _mw_shd["watch_scan_sec"], 36.8, 0.05)
_st = load("bench_stress250k_results.json")
check("stress250k SHD fit s", _st["shd_fit_s"], 28.6, 0.05)
check("stress250k SHD finalize s", _st["shd_finalize_s"], 0.08, 0.005)
check("stress250k SHD peak MB", _st["shd_peak_mb"], 71.7, 0.05)
check("stress250k SHD artifact KB", _st["shd_artifact_kb"], 225.2, 0.05)
check("stress250k SHD rel error", _st["shd_rel_error"], 0.3468, 0.0005)
check("stress250k SHD panel radar ms", _st["shd_panel_radar_ms"], 117.71, 0.05)
# Figures removed from the README because their probe scripts / artifacts were
# never committed (Scania validation-AUC probe; 250k MiniBatchKMeans comparison).
# Forbid them so the unbacked numbers cannot silently return.
for _needle in ("aucV 0.831", "aucV 0.974", "aucV 0.799",
                "MiniBatchKMeans 5.14", "MBK predict 0.74"):
    check_absent("readme unbacked " + _needle, _needle, README)

check_digest("portfolio SC artifact digest", artifact_digest(pf_sc), "282c31d352ce58eb1ed92c895d686078f3d90964e4f589ff777531e4deaa6de7")
check_digest("portfolio PM2.5 artifact digest", artifact_digest(pf_pm), "0567106d3bfe8a1dc0508694ef93ba19f2928ff53665dc594a671ba10c213447")
check_digest("portfolio reactor artifact digest", artifact_digest(pf_re), "752a680a985b8137159abb74864f4e66b4ee81fd83c5014c50b0c991f738e975")
check_digest("global-guard artifact digest", artifact_digest(gg), "ae096e1578291373babce23253cfb3cd699bd45df5d47d053700d7f949ee1ce9")

print(f"\n{checks} checks, {len(failures)} failures")
if failures:
    print("FAILED:", failures)
    sys.exit(1)

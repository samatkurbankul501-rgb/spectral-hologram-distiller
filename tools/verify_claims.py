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

ROOT = Path(__file__).resolve().parent.parent
PAPER = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

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

def check_text(name, needle, haystack=PAPER):
    global checks
    checks += 1
    ok = needle in haystack
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} text={needle[:60]!r}")
    if not ok:
        failures.append(name)

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

# ---------------------------------------------------------------- paper strings
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
    check_text("paper contains " + text[:30], text)

for needle in [
    r"zero catastrophic", r"strictly preserved", r"M = 90",
    r"state-of-the-art", r"22.90", r"21.41", r"56.0", r"55.5",
    r"8.7\%", r"whitened", r"phase-weighted", r"diag(",
    r"solved online", r"B / n_seen", r"higher phase score",
    r"cannot be pruned",
]:
    check_absent("paper stale " + needle, needle)

for text in [
    "# Spectral Hologram Distiller (SHD v3.12.0)",
    "DomainBank", "passport_recalibrate", 'weighting="uniform"',
    # "196.0" lived in the paper only (PM2.5 guard KRR-crit); the README
    # carries the threshold and the Scania official-cost row instead.
    "59.0", "55.0", "PM2.5", "324", "85.3", "0.465", "993.0",
    "0.203", "0.458",
]:
    check_text("readme contains " + text[:30], text, README)
# "55.5" is a LIVE number in the README (Scania missed failures, SHD row),
# so it must not be forbidden here; the stale-value watch is the "993.0"
# positive check above.
for needle in ["zero catastrophic", "M = 90", "sub-millisecond", "8.7%", "56.0"]:
    check_absent("readme stale " + needle, needle, README)

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
check_text("readme claims the negative result",
           "background-only spectral passport is not a fix", README)
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
check("compression ratio", 21263 / 300, 70.88, 0.02)

version = (ROOT / "shd" / "__init__.py").read_text(encoding="utf-8")
check_text("version 3.12.0", '__version__ = "3.12.0"', version)
pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
check_text("pyproject 3.12.0", 'version = "3.12.0"', pyproject)
# The three version carriers must agree; this is the check that would have
# caught the v3.10.0/v3.11.0 drift before it reached the README title.
check("version carriers agree",
      1 if ('__version__ = "3.12.0"' in version and 'version = "3.12.0"' in pyproject
            and "SHD v3.12.0" in README) else 0, 1, 0)

print(f"\n{checks} checks, {len(failures)} failures")
if failures:
    print("FAILED:", failures)
    sys.exit(1)

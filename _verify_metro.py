"""Check historical Metro artifacts separately from current protocol-v2 replay.

A-C check preserved v1 results/cache, NOT the validity of the old calibration.
D checks current timestamped windows and exact frozen-baseline calibration.
Old detection counts are not imposed on the corrected implementation.
"""

import csv
import io
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
NPZ = r"C:\Users\User\Downloads\New folder\_datasets\metro_minute_cache.npz"
ZIP = r"C:\Users\User\Downloads\New folder\_datasets\metropt3.zip"
FAIL = []


def chk(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name + " :: " + detail)


md = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
pitch = open(os.path.join(ROOT, "PITCH.md"), encoding="utf-8").read()
email = open(os.path.join(ROOT, "EMAIL_ENGINEER.md"), encoding="utf-8").read()
res = json.load(open(os.path.join(ROOT, "bench_metro_watch_results.json"),
                     encoding="utf-8"))

print("=" * 96)
print("A. Docs vs artifact (warm-baseline run)")
print("=" * 96)
arms = {a["arm"]: a for a in res["arms"]}
shd, z, ifo, raw = (arms[k] for k in ("shd_detrend", "z_detrend",
                                      "iforest_detrend", "shd_raw"))

chk("shd watch eps 0", shd["watch_episodes"] == 0, str(shd["watch_episodes"]))
chk("shd missed all four", all(r["detected"] is False for r in shd["incidents"]))
chk("z eps 27", z["watch_episodes"] == 27)
chk("z FA 3.19", abs(z["unmatched_fa_per_month"] - 3.19) < 0.005)
inc_z = {r["name"]: r for r in z["incidents"]}
chk("z #1 lead 131.6", abs(inc_z["Air leak #1"]["lead_hours"] - 131.6) < 0.15)
chk("z #2 lead 393.3", abs(inc_z["Air leak #2"]["lead_hours"] - 393.3) < 0.15)
chk("z #3 lead 415.4", abs(inc_z["Air leak #3"]["lead_hours"] - 415.4) < 0.15)
chk("z #4 lead 164.2", abs(inc_z["Air leak #4"]["lead_hours"] - 164.2) < 0.15)
inc_ifo = {r["name"]: r for r in ifo["incidents"]}
chk("iforest eps 1", ifo["watch_episodes"] == 1)
chk("iforest #2 lead 392.3", abs(inc_ifo["Air leak #2"]["lead_hours"] - 392.3) < 0.15)
chk("iforest #1/#3/#4 missed", all(inc_ifo[n]["detected"] is False
    for n in ("Air leak #1", "Air leak #3", "Air leak #4")))
chk("raw eps 0", raw["watch_episodes"] == 0)

for needle in ("Historical protocol v1", "protocol v2", "recalibrate"):
    chk("README distinguishes protocols: " + repr(needle), needle in md)
chk("README no longer claims 5.4 days lead", "5.4 days before" not in md
    and "130.6 h before" not in md)
chk("PITCH honest Metro section", "отрицательный" in pitch and "5.4 дня" not in pitch)
chk("EMAIL honest", "5.4 дня" not in email and "ни одной" in email)
chk("PITCH keeps Scania numbers", "993" in pitch and "22%" in pitch)

print()
print("=" * 96)
print("B. metro_cache spot check vs raw CSV")
print("=" * 96)
d = np.load(NPZ, allow_pickle=False)
minutes = d["minutes"].astype("datetime64[m]")
X = d["X"].astype(np.float64)
feats = [str(f) for f in d["features"]]
chk("22 features", len(feats) == 22)
chk("sorted", np.all(minutes[:-1] < minutes[1:]))
probe = np.datetime64("2020-05-15 14:00", "m")
k = int(np.searchsorted(minutes, probe))
chk("probe present", minutes[k] == probe)
lo_s, hi_s = (np.datetime64("2020-05-15 13:59:00", "s"),
              np.datetime64("2020-05-15 14:01:00", "s"))
rows, tss = [], []
with zipfile.ZipFile(ZIP) as zf:
    with zf.open("MetroPT3(AirCompressor).csv") as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        r = csv.reader(text)
        next(r)
        for row in r:
            t = np.datetime64(row[1].strip(), "s")
            if t < lo_s:
                continue
            if t > hi_s:
                break
            try:
                rows.append([float(v) for v in row[2:17]])
                tss.append(t)
            except ValueError:
                pass
rows = np.asarray(rows)
m = (np.asarray(tss) >= np.datetime64("2020-05-15 14:00:00", "s")) & \
     (np.asarray(tss) < np.datetime64("2020-05-15 14:01:00", "s"))
chk("raw rows in probe minute", m.sum() > 0, str(int(m.sum())))
if m.any():
    worst = max(abs(X[k, j] - rows[m, j].mean()) for j in range(15))
    chk("minute means match raw", worst < 1e-3, "worst {:.2e}".format(worst))
    std_idx = feats.index("TP2_std")
    chk("TP2_std matches raw", abs(X[k, std_idx] - rows[m, 0].std()) < 5e-3,
        "cache {:.4f} raw {:.4f}".format(X[k, std_idx], rows[m, 0].std()))

print()
print("=" * 96)
print("C. Protocol integrity")
print("=" * 96)
p = res["protocol"]
chk("slices disjoint", p["learn"][1] == p["calibrate"][0]
    and p["calibrate"][1] == p["watch"][0])
chk("incidents outside calibration",
    all(i["start"] >= p["calibrate"][1] for i in res["incidents_truth"]))
chk("budget 1.0 / halflife 7", p["fa_budget_per_month"] == 1.0
    and p["ewma_halflife_days"] == 7.0)

print()
print("=" * 96)
print("D. Current protocol-v2 replay (historical detection counts are not targets)")
print("=" * 96)
from shd import HologramCrystalLoader
from shd_watch import (MAX_ALERT_FRACTION, calibrate_tau, detrended_scores, episodes,
                       load_minute_table, slice_range, window_scores, window_bounds)

minutes, X, feats = load_minute_table(NPZ)
loader = HologramCrystalLoader(os.path.join(ROOT, "metro_demo_crystal.hologram.npz"))
c_lo, c_hi = slice_range(minutes, "2020-03-15", "2020-04-10")
w_lo, w_hi = slice_range(minutes, "2020-04-10", "2020-08-01")
mc, mw = minutes[c_lo:c_hi], minutes[w_lo:w_hi]
idx_c, sc_c = window_scores(loader, X[c_lo:c_hi], 60, 30, minutes=mc)
idx_w, sc_w = window_scores(loader, X[w_lo:w_hi], 60, 30, minutes=mw)
m_c, m_w = mc[idx_c], mw[idx_w]
chk("windows never cross missing minutes", all(
    ts[hi - 1] - ts[lo] == np.timedelta64(59, "m")
    for ts in (mc, mw) for lo, hi in window_bounds(ts, 60, 30)))
tau, count = calibrate_tau("2020-03-15", "2020-04-10", m_c, sc_c, 1.0, 6.0,
                           halflife_days=7.0, step_min=30)
dev_c, b_end = detrended_scores(m_c, sc_c, 7.0, 30, tau=tau, return_baseline=True)
chk("calibration count matches deployed replay", count == len(episodes(m_c, dev_c, tau, 6.0)))
chk("short calibration slice permits zero episodes", count == 0)
chk("alert fraction respects explicit cap", np.mean(dev_c >= tau) <= MAX_ALERT_FRACTION)
dev_warm = detrended_scores(m_w, sc_w, 7.0, 30, tau=tau, b_init=b_end,
                            baseline_time=m_c[-1])
chk("current replay finite", np.isfinite(tau) and np.isfinite(dev_warm).all())
print("  INFO current v2 replay on historical crystal: tau={}, watch episodes={}".format(
    tau, len(episodes(m_w, dev_warm, tau, 6.0))))

print()
print("RESULT: {} failures".format(len(FAIL)))
for f in FAIL:
    print("  " + f)
raise SystemExit(1 if FAIL else 0)

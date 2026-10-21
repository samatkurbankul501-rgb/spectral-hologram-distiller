"""Trust check for the Metro v2 negative result (2026-10-20).

The v2 run reports the SHD passport arm as silent on all four leaks. Before
treating that as a real negative result (and publishing it as such), this
script re-derives the SHD arm from the SAVED crystal file on disk:

  * reload metro_demo_v2_crystal.hologram.npz, rescore both slices,
  * re-run calibrate_tau and require the threshold to reproduce the value
    stored in bench_metro_watch_v2_results.json BIT-FOR-BIT,
  * replay the watch and require zero episodes,
  * then measure how far the radar actually got: max detrended deviation
    overall and inside each leak's 21-day pre-incident window, the margin to
    tau, and basic score-variance sanity so "silent" cannot mean "broken".

Exit code 0 = reproduction holds. ASCII output only.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_metro_watch import (NPZ, INCIDENTS_PATH, CALIB, WATCH, WIN, STEP,
                               COOLDOWN_H, BUDGET, HALFLIFE_D, MAX_LEAD_DAYS)
from shd import HologramCrystalLoader
from shd_watch import (MAX_ALERT_FRACTION, calibrate_tau, detrended_scores,
                       episodes, load_minute_table, slice_range, window_scores)

CRYSTAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "metro_demo_v2_crystal.hologram.npz")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bench_metro_watch_v2_results.json")

with open(RESULTS, encoding="utf-8") as f:
    published = {a["arm"]: a for a in json.load(f)["arms"]}
shd = published["shd_detrend"]
TAU_PUBLISHED = shd["tau"]

print("published SHD arm: tau={!r}  episodes={}  unmatched={}".format(
    TAU_PUBLISHED, shd["watch_episodes"], shd["unmatched_episodes"]))

minutes, X, feats = load_minute_table(NPZ)
c_lo, c_hi = slice_range(minutes, *CALIB)
w_lo, w_hi = slice_range(minutes, *WATCH)
loader = HologramCrystalLoader(CRYSTAL)
print("crystal loaded: {} medoids, {} features".format(
    len(loader.X), loader.X.shape[1]))

mc, mw = minutes[c_lo:c_hi], minutes[w_lo:w_hi]
idx_c, sc_c = window_scores(loader, X[c_lo:c_hi], WIN, STEP, minutes=mc)
idx_w, sc_w = window_scores(loader, X[w_lo:w_hi], WIN, STEP, minutes=mw)
m_c, m_w = mc[idx_c], mw[idx_w]
print("scored windows: calib={} watch={}".format(len(sc_c), len(sc_w)))

tau, n_cal = calibrate_tau(CALIB[0], CALIB[1], m_c, sc_c, BUDGET, COOLDOWN_H,
                           halflife_days=HALFLIFE_D, step_min=STEP,
                           max_alert_fraction=MAX_ALERT_FRACTION)
print("reproduced tau: {!r}".format(tau))
ok_tau = (float(tau) == TAU_PUBLISHED)
print("BIT-FOR-BIT tau match:", "PASS" if ok_tau else "FAIL")

dev_cal, b_end = detrended_scores(m_c, sc_c, HALFLIFE_D, STEP, tau=tau,
                                  return_baseline=True)
dev_w = detrended_scores(m_w, sc_w, HALFLIFE_D, STEP, tau=tau, b_init=b_end,
                         baseline_time=m_c[-1])
eps = episodes(m_w, dev_w, tau, COOLDOWN_H)
ok_eps = len(eps) == shd["watch_episodes"]
print("episode reproduction ({} vs published {}): {}".format(
    len(eps), shd["watch_episodes"], "PASS" if ok_eps else "FAIL"))

print("\n--- how close did the radar get? ---")
print("tau                          = {:.6f}".format(tau))
print("watch dev  q50/q95/q99/max   = {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
    np.percentile(dev_w, 50), np.percentile(dev_w, 95),
    np.percentile(dev_w, 99), dev_w.max()))
i_max = int(np.argmax(dev_w))
print("max dev at {} (dev={:.4f})".format(m_w[i_max], dev_w[i_max]))
print("raw score min/max/std        = {:.4f} / {:.4f} / {:.4f}".format(
    sc_w.min(), sc_w.max(), sc_w.std()))
near = int(np.sum(dev_w >= 0.9 * tau))
print("windows within 10% of tau    = {}".format(near))

incidents = json.load(open(INCIDENTS_PATH, encoding="utf-8"))
print("\nper-incident margins (matching span = start-21d .. end):")
for inc in incidents:
    s = np.datetime64(inc["start"], "m")
    e = np.datetime64(inc["end"], "m")
    lo = s - np.timedelta64(MAX_LEAD_DAYS, "D")
    mask = (m_w >= lo) & (m_w <= e)
    if not mask.any():
        print("  {:>12s}: no scored windows in span".format(inc["name"]))
        continue
    dmax = float(dev_w[mask].max())
    margin = float(tau) - dmax
    frac = dmax / float(tau)
    print("  {:>12s}: max dev {:.4f} = {:.1f}% of tau (margin {:+.4f})".format(
        inc["name"], dmax, 100.0 * frac, margin))

sys.exit(0 if (ok_tau and ok_eps) else 1)

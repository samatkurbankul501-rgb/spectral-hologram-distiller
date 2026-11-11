"""MetroPT-3 combined-radar prototype (clean rewrite).

diag_metro_bw.py proved the BW spectral-moment distance is nearly blind to
the mean shifts that leak signatures produce (incident BW medians 0.87-1.04
vs healthy 0.66, while z-score separates cleanly at 4.8-19.6). This
prototype tests a combined score on the SAME protocol as bench_metro_watch:

    score = bw_distance + ||V^T(mean U_win) - V^T(mean U_train)||

where V are the crystal's top-32 eigenvectors and U is the normalised RFF
sketch. The z-score arm is the reference. Same calibration, budget,
cooldown, EWMA-freeze replay for every arm.

Usage:
    python tools/proto_metro_bw.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shd import HologramCrystalLoader
from shd_watch import (calibrate_tau, detrended_scores, episodes,
                       load_minute_table, slice_range, window_bounds, window_scores)

_DATASETS = os.environ.get(
    "SHD_DATASETS", os.path.normpath(os.path.join(ROOT, "..", "..", "_datasets")))
NPZ = os.path.join(_DATASETS, "metro_minute_cache.npz")
INC = os.path.join(ROOT, "metro_incidents.json")
LEARN = ("2020-02-01", "2020-03-15")
CALIB = ("2020-03-15", "2020-04-10")
WATCH = ("2020-04-10", "2020-08-01")
WIN, STEP, COOLDOWN_H, BUDGET, HALFLIFE_D, MAX_LEAD_D = 60, 30, 6.0, 1.0, 7.0, 21


def exposure_months(score_minutes, step_min, t_from, t_to):
    gaps = np.diff(score_minutes) / np.timedelta64(1, "m")
    span = float((np.datetime64(t_to, "m") - np.datetime64(t_from, "m")) / np.timedelta64(1, "m"))
    observed = min(span, float(step_min + np.minimum(gaps, step_min).sum()))
    return observed / (30.4375 * 24 * 60)


def report_arm(name, incidents, minutes_c, sc_c, minutes_w, sc_w):
    tau, n_cal = calibrate_tau(CALIB[0], CALIB[1], minutes_c, sc_c, BUDGET, COOLDOWN_H,
                               halflife_days=HALFLIFE_D, step_min=STEP, max_alert_fraction=0.05)
    dev_cal, b_end = detrended_scores(minutes_c, sc_c, HALFLIFE_D, STEP, tau=tau,
                                      return_baseline=True)
    dev_w = detrended_scores(minutes_w, sc_w, HALFLIFE_D, STEP, tau=tau,
                             b_init=b_end, baseline_time=minutes_c[-1])
    eps = episodes(minutes_w, dev_w, tau, COOLDOWN_H)
    n_unmatched = 0
    print(f"\n--- {name} (tau={tau:.4f}, calib_eps={n_cal}, watch_eps={len(eps)}) ---")
    for inc in incidents:
        s = np.datetime64(inc["start"], "m") - np.timedelta64(MAX_LEAD_D, "D")
        e = np.datetime64(inc["end"], "m")
        first = next(((a, b, p) for a, b, p in eps if s <= a <= e), None)
        if first is not None:
            lead = max(float((np.datetime64(inc["start"], "m") - first[0]) / np.timedelta64(1, "h")), 0.0)
            print(f"  {inc['name']:<14s} first alert {str(first[0])[:16]} | lead {lead:.1f} h")
        else:
            n_unmatched += 1
            print(f"  {inc['name']:<14s} MISSED")
    months = exposure_months(minutes_w, STEP, WATCH[0], WATCH[1])
    print(f"  unmatched FA/month: {n_unmatched / max(months, 1e-9):.3f} "
          f"(budget {BUDGET:.1f}, exposure {months:.2f} mo)")


def run():
    minutes, X, feats = load_minute_table(NPZ)
    incidents = json.load(open(INC, encoding="utf-8"))
    l_lo, l_hi = slice_range(minutes, *LEARN)
    c_lo, c_hi = slice_range(minutes, *CALIB)
    w_lo, w_hi = slice_range(minutes, *WATCH)

    loader = HologramCrystalLoader(os.path.join(ROOT, "metro_demo_v2_crystal.hologram.npz"))

    # Normalised-feature statistics of the LEARN slice (the z-score reference
    # already uses these). The mean-shift term must live in feature space:
    # the RFF sketch normalises every row to unit norm, which averages out
    # mean information, so the RFF-space mean distance is ~0 and useless.
    mu, sd = X[l_lo:l_hi].mean(axis=0), X[l_lo:l_hi].std(axis=0)
    ok = sd > 1e-9

    def combined(X_slice, minutes_slice):
        idx, bw = window_scores(loader, X_slice, WIN, STEP, minutes=minutes_slice)
        ends = minutes_slice[idx]
        zs = np.zeros(len(idx))
        for j, end in enumerate(idx):
            lo, hi = int(end + 1 - WIN), int(end + 1)
            z = np.abs(X_slice[lo:hi].mean(axis=0) - mu) / np.where(ok, sd, 1.0)
            zs[j] = float(z[ok].max()) if ok.any() else 0.0
        return ends, bw + zs

    mc_c, sc_c = combined(X[c_lo:c_hi], minutes[c_lo:c_hi])
    mc_w, sc_w = combined(X[w_lo:w_hi], minutes[w_lo:w_hi])

    # z-score reference on identical windows (feature space)
    def zscores(X_slice, minutes_slice):
        z = []
        ends = []
        for lo, hi in window_bounds(minutes_slice, WIN, STEP):
            w = X_slice[lo:hi]
            zz = np.abs(w.mean(axis=0) - mu) / np.where(ok, sd, 1.0)
            ends.append(minutes_slice[hi - 1])
            z.append(float(zz[ok].max()) if ok.any() else 0.0)
        return np.asarray(ends, dtype="datetime64[m]"), np.asarray(z)

    zc_m, zc = zscores(X[c_lo:c_hi], minutes[c_lo:c_hi])
    zw_m, zw = zscores(X[w_lo:w_hi], minutes[w_lo:w_hi])

    report_arm("shd_bw+mean", incidents, mc_c, sc_c, mc_w, sc_w)
    report_arm("z_score_ref", incidents, zc_m, zc, zw_m, zw)


if __name__ == "__main__":
    run()

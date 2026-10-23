"""Early-warning benchmark on a REAL compressor: MetroPT-3 (metro train APU).

The commercial question, on real equipment with real labelled failures: at a
calibrated false-alarm budget, how much warning does each detector give before
each of the four documented air leaks -- and does the SHD passport beat the two
detectors an engineer would actually deploy first?

Arms (IDENTICAL protocol: same windows, same calibration slice, same budget,
same drift compensation -- the comparison is the score, not the plumbing):

  shd_detrend : Bures-Wasserstein distance to an SHD crystal, EWMA-detrended.
  z_detrend   : the engineer's baseline -- per-feature z-score of the window
                mean against the training minutes, max over features,
                EWMA-detrended.
  shd_radar   : the v3.13.0 combined radar -- the loader's
                check_domain_shift(mean_shift=True) scores every contiguous
                window (bw_distance + mean_shift_distance) and alarms under
                the same FA budget replay. This is the library path behind
                the 4/4 leak detection (FA 0.000/month vs budget 1.0).
                (tools/proto_metro_bw.py validated it first; v3.13.0 moved
                it into the loader).
  iforest_detrend : IsolationForest on healthy minutes, mean window anomaly
                score, EWMA-detrended. The standard "just buy a model" answer.
  shd_raw     : diagnostic context -- SHD WITHOUT drift compensation, to show
                why the trailing baseline is not optional on real telemetry.

Drift compensation (shd_watch.detrended_scores): deviations from a 7-day
EWMA of the detector's own score; the baseline freezes during active alarms
(see the docstring there). Calibration is a candidate search: tau is
chosen by replaying each candidate with freezing, just as watch does. Protocol
v2 excludes windows crossing missing minutes, enforces the empirical alarm-rate
budget without rounding up, and caps the fraction of alert windows at 5%.
Unmatched false-alarm rates use the same observed-exposure denominator as the
calibration budget; calendar-month rates are recorded alongside.

Protocol
--------
  learn       : 2020-02-01 .. 2020-03-15  (6 healthy weeks)
  calibrate   : 2020-03-15 .. 2020-04-10  (4 healthy weeks; NO failure info)
  watch       : 2020-04-10 .. 2020-08-01  (contains all four labelled leaks)
  windows     : 60 minutes, step 30, alert cooldown 6 h, budget 1 FA/month
  features    : 22 = 15 minute-means + 7 per-minute STDs of the analog sensors
                (oscillation depth is where a leak shows up first; minute means
                alone hide it -- measured, not assumed)

Ground truth: metro_incidents.json, transcribed from the dataset's own failure
reports (Data Description_Metro.pdf).

Usage:
    python bench_metro_watch.py
Writes bench_metro_watch_v2_results.json and metro_demo_v2_crystal.hologram.npz.
The historical v1 outputs are never overwritten by default.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.ensemble import IsolationForest

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd_watch import (DAY, PROTOCOL_VERSION, MAX_ALERT_FRACTION,
                       calibrate_tau, detrended_scores, episodes,
                       load_minute_table, slice_range, window_scores, window_bounds)

NPZ = r"C:\Users\User\Downloads\New folder\_datasets\metro_minute_cache.npz"
INCIDENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "metro_incidents.json")

LEARN = ("2020-02-01", "2020-03-15")
CALIB = ("2020-03-15", "2020-04-10")
WATCH = ("2020-04-10", "2020-08-01")

WIN, STEP, COOLDOWN_H, BUDGET = 60, 30, 6.0, 1.0
HALFLIFE_D = 7.0
MAX_LEAD_DAYS = 21


def per_incident(incidents, eps):
    rows = []
    for inc in incidents:
        s = np.datetime64(inc["start"], "m")
        e = np.datetime64(inc["end"], "m")
        first = None
        for a, b, p in eps:
            if (s - np.timedelta64(MAX_LEAD_DAYS, "D")) <= a <= e:
                first = (a, b, p)
                break
        lead = max(float((s - first[0]) / np.timedelta64(1, "h")), 0.0) if first else None
        rows.append({
            "name": inc["name"], "start": inc["start"],
            "first_alert": str(first[0]) if first else None,
            "lead_hours": lead,
            "detected": first is not None,
        })
    return rows


def exposure_months(score_minutes, step_min, t_from, t_to):
    """Observed-exposure months -- the same denominator calibrate_tau budgets by.

    Missing telemetry is not exposure: the gap between consecutive scored
    windows counts at the nominal cadence at most and the total is capped at
    the declared span. Reporting the FA rate against this denominator keeps
    it comparable with the budget the device itself enforces; the calendar
    rate is recorded alongside for transparency.
    """
    gaps = np.diff(score_minutes) / np.timedelta64(1, "m")
    span = float((np.datetime64(t_to, "m") - np.datetime64(t_from, "m"))
                 / np.timedelta64(1, "m"))
    observed = min(span, float(step_min + np.minimum(gaps, step_min).sum()))
    return observed / (30.4375 * 24 * 60)


def unmatched_fa(incidents, eps, t_from, t_to, score_minutes, step_min):
    spans = []
    for inc in incidents:
        lo = np.datetime64(inc["start"], "m") - np.timedelta64(MAX_LEAD_DAYS, "D")
        hi = np.datetime64(inc["end"], "m")
        spans.append((lo, hi))
    calendar = float((np.datetime64(t_to, "m") - np.datetime64(t_from, "m"))
                     / (30.4375 * DAY))
    observed = exposure_months(score_minutes, step_min, t_from, t_to)
    n_unmatched = sum(1 for a, b, p in eps
                      if not any(lo <= a <= hi for lo, hi in spans))
    return n_unmatched, {
        "unmatched": int(n_unmatched),
        "per_month_observed": n_unmatched / max(observed, 1e-9),
        "per_month_calendar": n_unmatched / max(calendar, 1e-9),
        "exposure_months": observed,
        "calendar_months": calendar,
    }


def score_feature_z(X_learn, X_slice, win, step, *, minutes):
    mu = X_learn.mean(axis=0)
    sd = X_learn.std(axis=0)
    ok = sd > 1e-9
    idx, scores = [], []
    for lo, hi in window_bounds(minutes, win, step):
        w = X_slice[lo:hi]
        z = np.abs(w.mean(axis=0) - mu) / np.where(ok, sd, 1.0)
        idx.append(hi - 1)
        scores.append(float(z[ok].max()) if ok.any() else 0.0)
    return np.asarray(idx, dtype=np.int64), np.asarray(scores)


def score_shd_radar(loader, X_slice, win, step, *, minutes):
    """Combined-radar arm (v3.13.0): loader.check_domain_shift(mean_shift=True)
    on every contiguous window; the combined score is what the radar alarms on.
    Returns (end-row indices, combined scores), matching window_scores' convention.
    """
    ends, scores = [], []
    for lo, hi in window_bounds(minutes, win, step):
        rep = loader.check_domain_shift(X_slice[lo:hi], mean_shift=True)
        ends.append(hi - 1)
        scores.append(rep["combined_distance"])
    return np.asarray(ends, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def score_iforest(forest, X_slice, win, step, *, minutes):
    idx, scores = [], []
    for lo, hi in window_bounds(minutes, win, step):
        idx.append(hi - 1)
        scores.append(float(-np.mean(forest.decision_function(X_slice[lo:hi]))))
    return np.asarray(idx, dtype=np.int64), np.asarray(scores)


def run_arm(name, incidents, raw_cal, raw_watch, minutes_cal, minutes_watch,
            detrend=True, extra=None):
    """Identical timestamp-aware calibration and deployed replay for every arm."""
    tau, n_cal_eps = calibrate_tau(
        CALIB[0], CALIB[1], minutes_cal, raw_cal, BUDGET, COOLDOWN_H,
        halflife_days=HALFLIFE_D if detrend else None, step_min=STEP,
        max_alert_fraction=MAX_ALERT_FRACTION)
    if detrend:
        dev_cal, b_end = detrended_scores(minutes_cal, raw_cal, HALFLIFE_D, STEP,
                                          tau=tau, return_baseline=True)
        dev_watch = detrended_scores(minutes_watch, raw_watch, HALFLIFE_D,
                                     STEP, tau=tau, b_init=b_end,
                                     baseline_time=minutes_cal[-1])
    else:
        dev_cal, dev_watch = raw_cal, raw_watch
    eps = episodes(minutes_watch, dev_watch, tau, COOLDOWN_H)
    un, rates = unmatched_fa(incidents, eps, WATCH[0], WATCH[1],
                             minutes_watch, STEP)
    rep = {
        "arm": name, "detrended": detrend, "tau": float(tau),
        "calibration_episodes": int(n_cal_eps),
        "calibration_alert_fraction": float(np.mean(dev_cal >= tau)),
        "calibration_windows": int(len(raw_cal)),
        "watch_windows": int(len(raw_watch)),
        "watch_episodes": len(eps),
        "unmatched_episodes": un,
        "unmatched_fa_per_month": round(rates["per_month_observed"], 3),
        "unmatched_fa_per_month_calendar": round(rates["per_month_calendar"], 3),
        "watch_exposure_months": round(rates["exposure_months"], 3),
        "watch_calendar_months": round(rates["calendar_months"], 3),
        "incidents": per_incident(incidents, eps),
    }
    if extra:
        rep.update(extra)
    return rep, eps


def run(output="bench_metro_watch_v2_results.json",
        crystal_path="metro_demo_v2_crystal.hologram.npz"):
    minutes, X, feats = load_minute_table(NPZ)
    incidents = json.load(open(INCIDENTS_PATH, encoding="utf-8"))
    print("MetroPT-3 minutes: {:,}  features: {}  span {} .. {}".format(
        len(minutes), len(feats), minutes[0], minutes[-1]), flush=True)

    l_lo, l_hi = slice_range(minutes, *LEARN)
    c_lo, c_hi = slice_range(minutes, *CALIB)
    w_lo, w_hi = slice_range(minutes, *WATCH)

    # SHD crystal on the learn slice
    t0 = time.perf_counter()
    d = SpectralHologramDistiller(
        crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=0)
    for i in range(l_lo, l_hi, 4000):
        d.partial_fit(X[i : min(i + 4000, l_hi)])
    d.finalize_crystal()
    crystal_kb = d.save_hologram(crystal_path)
    loader = HologramCrystalLoader(crystal_path)
    learn_s = time.perf_counter() - t0

    # raw scores on calib and watch for the SHD arm
    mc, mw = minutes[c_lo:c_hi], minutes[w_lo:w_hi]
    idx_c, sc_c = window_scores(loader, X[c_lo:c_hi], WIN, STEP, minutes=mc)
    t1 = time.perf_counter()
    idx_w, sc_w = window_scores(loader, X[w_lo:w_hi], WIN, STEP, minutes=mw)
    watch_scan_s = time.perf_counter() - t1
    m_c, m_w = mc[idx_c], mw[idx_w]
    rep_shd, eps_shd = run_arm("shd_detrend", incidents, sc_c, sc_w, m_c, m_w,
                               extra={"learn_sec": round(learn_s, 1),
                                      "crystal_kb": round(crystal_kb, 1),
                                      "watch_scan_sec": round(watch_scan_s, 1)})
    rep_raw, _ = run_arm("shd_raw", incidents, sc_c, sc_w, m_c, m_w, detrend=False)

    # feature-z arm
    z_idx_c, z_c = score_feature_z(X[l_lo:l_hi], X[c_lo:c_hi], WIN, STEP, minutes=mc)
    z_idx_w, z_w = score_feature_z(X[l_lo:l_hi], X[w_lo:w_hi], WIN, STEP, minutes=mw)
    assert np.array_equal(idx_c, z_idx_c) and np.array_equal(idx_w, z_idx_w)
    rep_z, _ = run_arm("z_detrend", incidents, z_c, z_w, m_c, m_w)

    # IsolationForest arm
    t2 = time.perf_counter()
    forest = IsolationForest(n_estimators=100, random_state=0, contamination="auto")
    forest.fit(X[l_lo:l_hi])
    if_fit_s = time.perf_counter() - t2
    i_idx_c, i_c = score_iforest(forest, X[c_lo:c_hi], WIN, STEP, minutes=mc)
    i_idx_w, i_w = score_iforest(forest, X[w_lo:w_hi], WIN, STEP, minutes=mw)
    assert np.array_equal(idx_c, i_idx_c) and np.array_equal(idx_w, i_idx_w)
    rep_i, _ = run_arm("iforest_detrend", incidents, i_c, i_w, m_c, m_w,
                        extra={"fit_sec": round(if_fit_s, 1)})

    # v3.13.0 combined radar (library path: check_domain_shift(mean_shift=True))
    r_idx_c, r_c = score_shd_radar(loader, X[c_lo:c_hi], WIN, STEP, minutes=mc)
    r_idx_w, r_w = score_shd_radar(loader, X[w_lo:w_hi], WIN, STEP, minutes=mw)
    assert np.array_equal(idx_c, r_idx_c) and np.array_equal(idx_w, r_idx_w)
    rep_radar, _ = run_arm("shd_radar", incidents, r_c, r_w, m_c, m_w,
                            extra={"radar": "bw_distance + mean_shift_distance (v3.13.0)"})

    results = {
        "dataset": "metropt3_watch", "title": "Early warning on a real metro APU",
        "protocol": {"version": PROTOCOL_VERSION,
                     "max_alert_fraction": MAX_ALERT_FRACTION,
                     "window_policy": "contiguous_minutes_only",
                     "learn": LEARN, "calibrate": CALIB, "watch": WATCH,
                     "window_min": WIN, "step_min": STEP,
                     "cooldown_h": COOLDOWN_H, "fa_budget_per_month": BUDGET,
                     "ewma_halflife_days": HALFLIFE_D,
                     "max_lead_days": MAX_LEAD_DAYS,
                     "features": list(feats)},
        "n_minutes": int(len(minutes)),
        "arms": [rep_shd, rep_z, rep_i, rep_raw, rep_radar],
        "incidents_truth": incidents,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 96)
    print("METRO APU EARLY WARNING: budget <= {:.0f} FA/month, window {}min, "
          "cooldown {}h, EWMA halflife {}d".format(BUDGET, WIN, COOLDOWN_H, HALFLIFE_D))
    print("=" * 96)
    for arm in results["arms"]:
        print("\n--- {} (tau={:.4f}) ---".format(arm["arm"], arm["tau"]))
        print("  calib eps: {} | watch eps: {} | unmatched FA/month: {:.2f}".format(
            arm["calibration_episodes"], arm["watch_episodes"],
            arm["unmatched_fa_per_month"]))
        for inc in arm["incidents"]:
            lead = ("{:.1f} h".format(inc["lead_hours"])
                    if inc["lead_hours"] is not None else "MISSED")
            print("  {:<14s} start {} | first alert {} | lead {}".format(
                inc["name"], inc["start"][:16],
                inc["first_alert"][:16] if inc["first_alert"] else "-", lead))
    print("\ncrystal: {:.1f} KB | learn {:.0f}s".format(crystal_kb, learn_s))
    print("Saved -> " + output)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="bench_metro_watch_v2_results.json")
    parser.add_argument("--crystal", default="metro_demo_v2_crystal.hologram.npz")
    args = parser.parse_args()
    run(args.output, args.crystal)

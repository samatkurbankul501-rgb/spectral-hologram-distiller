"""shd_watch: an early-warning device built on an SHD crystal.

This is the product surface, not a benchmark: point it at a CSV of sensor
readings and it (1) learns a tiny spectral passport of normal operation,
(2) calibrates its alert threshold against a FALSE-ALARM BUDGET on held-out
healthy data -- never against failures -- and (3) watches the stream and
raises alarms, reporting lead time to labelled incidents.

Why this exists: every benchmark in this repository scores regression error,
but a maintenance buyer pays for avoided failures and wasted inspections.
`shd_watch` reports exactly those: false alarms per month and hours of warning.

Design commitments
------------------
* No test-incident information ever touches threshold selection.
* The same calibration protocol is applied to every detector arm, so the
  comparison is about the score, not about who got a friendlier threshold.
* Pure numpy + the SHD crystal; runs on one core; the learned artifact is
  tens of kilobytes.
* Latency and memory are reported, not implied.

Example (MetroPT-3 compressor, see bench_metro_watch.py):
    python shd_watch.py learn --npz minutes.npz --from 2020-02-01 --to 2020-03-15 \
        --out crystal.hologram.npz
    python shd_watch.py calibrate --crystal crystal.hologram.npz --npz minutes.npz \
        --from 2020-03-15 --to 2020-04-10 --fa-per-month 1 --out threshold.json
    python shd_watch.py watch --crystal crystal.hologram.npz --threshold threshold.json \
        --npz minutes.npz --from 2020-04-10 --to 2020-08-01

Protocol v2 requires contiguous minute windows and replays each candidate
threshold with deployed EWMA-freeze semantics. The default 5% alert-window cap
is an explicit operational guard, not a learned or statistically guaranteed
false-positive bound. Legacy threshold files must be recalibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time

import numpy as np

from shd import HologramCrystalLoader, SpectralHologramDistiller

HOURS = np.timedelta64(1, "h")
MINUTE = np.timedelta64(1, "m")
DAY = np.timedelta64(1, "D")
HALFLIFE_D = 7.0  # EWMA baseline half-life (days) for drift compensation
PROTOCOL_VERSION = 2
MAX_ALERT_FRACTION = 0.05


# --------------------------------------------------------------------------- io

def load_minute_table(path):
    """Load the npz cache built by metro_cache.py (minutes as int64 epoch-min)."""
    with np.load(path, allow_pickle=False) as d:
        minutes = d["minutes"].astype("datetime64[m]")
        X = d["X"].astype(np.float64)
        features = [str(f) for f in d["features"]]
    _minute_axis(minutes, len(X))
    if X.ndim != 2 or X.shape[1] != len(features) or not np.isfinite(X).all():
        raise ValueError("minute cache must contain finite 2-D features with matching names")
    return minutes, X, features


def read_csv_minutes(path, tcol, columns):
    """Fallback reader for a plain CSV: returns a minute-mean table.

    Same input contract as load_minute_table: sensor values must be finite
    and every timestamp must parse. Rows whose sensor values do not parse are
    skipped (logged telemetry often carries textual status codes); a bad
    timestamp is a hard error, not something to guess around. Multiple
    samples inside one minute are averaged -- that is this reader's
    documented purpose, not silent deduplication.
    """
    tss, xs = [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = [header.index(c) for c in columns]
        ti = header.index(tcol)
        for row in r:
            if len(row) <= max(idx + [ti]):
                continue
            try:
                values = [float(row[i]) for i in idx]
            except ValueError:
                continue
            try:
                ts = np.datetime64(row[ti].replace("T", " ").strip(), "s")
            except ValueError:
                raise ValueError("csv timestamp {!r} in column {!r} does not "
                                 "parse; fix the row or choose another --tcol"
                                 .format(row[ti], tcol))
            if np.isnat(ts):
                raise ValueError("csv timestamp column {!r} contains NaT"
                                 .format(tcol))
            tss.append(ts)
            xs.append(values)
    if not tss:
        raise ValueError("csv contains no rows with parseable timestamps and features")
    X = np.asarray(xs, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(columns) or not np.isfinite(X).all():
        raise ValueError("csv feature columns must be finite numeric values; "
                          "nan/inf rows must be removed or imputed before "
                          "learn/calibrate/watch")
    t = np.asarray(tss)
    if (np.diff(t) < np.timedelta64(0, "s")).any():
        raise ValueError("csv timestamps must be non-decreasing in file order; "
                         "sort the CSV before learn/calibrate/watch")
    m = t.astype("datetime64[m]")
    uniq, inv = np.unique(m, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    Xm = np.column_stack([
        np.bincount(inv, weights=X[:, j]) / counts for j in range(X.shape[1])])
    _minute_axis(uniq.astype("datetime64[m]"), len(Xm))
    return uniq.astype("datetime64[m]"), Xm, list(columns)


# ------------------------------------------------------------------------ core

def slice_range(minutes, t_from, t_to):
    minutes = _minute_axis(minutes)
    start, end = np.datetime64(t_from, "m"), np.datetime64(t_to, "m")
    if np.isnat(start) or np.isnat(end) or end <= start:
        raise ValueError("--from must precede --to and both must be finite dates")
    return np.searchsorted(minutes, start), np.searchsorted(minutes, end)


def build_crystal(X, m=300, seed=0, batch=4000):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X), batch):
        d.partial_fit(X[i : i + batch])
    d.finalize_crystal()
    return d


def _minute_axis(minutes, n=None):
    minutes = np.asarray(minutes, dtype="datetime64[m]")
    if minutes.ndim != 1 or (n is not None and len(minutes) != n):
        raise ValueError("timestamps must be 1-D and aligned with observations")
    if np.isnat(minutes).any() or np.any(np.diff(minutes) <= MINUTE * 0):
        raise ValueError("timestamps must be finite, unique and strictly increasing")
    return minutes


def window_bounds(minutes, win, step):
    """Yield half-open row bounds for complete, contiguous minute windows.

    A missing minute breaks a segment. Each segment starts a new step grid;
    no window bridges a gap and absent samples are never interpolated.
    """
    minutes = _minute_axis(minutes)
    if int(win) != win or int(step) != step or win < 1 or step < 1:
        raise ValueError("window and step must be positive integer minutes")
    win, step = int(win), int(step)
    cuts = np.r_[0, np.flatnonzero(np.diff(minutes) != MINUTE) + 1, len(minutes)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        for start in range(int(lo), int(hi) - win + 1, step):
            yield start, start + win


def window_scores(loader, X, win, step, *, minutes):
    """Return (end-row indices, BW scores) for contiguous timestamped windows."""
    minutes = _minute_axis(minutes, len(X))
    ends, scores = [], []
    for lo, hi in window_bounds(minutes, win, step):
        rep = loader.check_domain_shift(X[lo:hi], adaptive_threshold=False)
        ends.append(hi - 1)
        scores.append(float(rep["bw_distance"]))
    return np.asarray(ends, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def episodes(score_minutes, score_vals, tau, cooldown_h=6.0):
    """Collapse per-window alerts into alarm episodes separated by `cooldown_h`.

    Two hits belong to the same episode when they are at most `cooldown_h` apart
    -- even if a few windows in between fell below tau (gaps inside an episode
    are normal: a drifting signal dips and recovers). Returns a list of
    (first_alert_minute, last_alert_minute, peak_score).
    """
    hits = np.flatnonzero(score_vals >= tau)
    eps = []
    gap = np.timedelta64(int(cooldown_h * 60), "m")
    for h in hits:
        t = score_minutes[h]
        if eps and (t - eps[-1][1]) <= gap:
            eps[-1][1] = t
            eps[-1][2] = max(eps[-1][2], float(score_vals[h]))
        else:
            eps.append([t, t, float(score_vals[h])])
    return [(a, b, p) for a, b, p in eps]


def fa_per_month(eps, t_from, t_to):
    days = float((np.datetime64(t_to) - np.datetime64(t_from)) / DAY)
    months = max(days / 30.4375, 1e-9)
    return len(eps) / months


def calibrate_tau(span_from, span_to, score_minutes, scores, fa_budget,
                  cooldown_h=6.0, *, max_alert_fraction=MAX_ALERT_FRACTION,
                  halflife_days=None, step_min=30, b_init=None):
    """First feasible candidate under the deployed detector's exact replay.

    ``scores`` are RAW scores when ``halflife_days`` is given; every candidate
    then replays the same freeze-on-alert EWMA used by watch. Candidates come
    from the always-updating deviations (and their next floats), but selection
    never assumes that episode counts are monotone. This is a candidate search,
    not a claim to find the globally smallest real-valued threshold.

    Enforce the literal observed episode rate (no rounding UP) and an explicit
    alert-window fraction cap. A finite nextafter(max_score) candidate permits
    zero alerts. Small calibration samples may therefore choose zero episodes;
    satisfying an empirical budget is not a statistical guarantee on new data.
    """
    score_minutes = _minute_axis(score_minutes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(score_minutes) or not len(scores):
        raise ValueError("calibration requires non-empty aligned 1-D scores")
    if not np.isfinite(scores).all():
        raise ValueError("calibration scores must be finite")
    start, end = np.datetime64(span_from, "m"), np.datetime64(span_to, "m")
    if np.isnat(start) or np.isnat(end) or end <= start:
        raise ValueError("calibration span must be finite and increasing")
    if score_minutes[0] < start or score_minutes[-1] >= end:
        raise ValueError("calibration scores must lie inside the declared span")
    if not np.isfinite(fa_budget) or fa_budget < 0:
        raise ValueError("false-alarm budget must be finite and non-negative")
    if not np.isfinite(cooldown_h) or cooldown_h < 0:
        raise ValueError("cooldown must be finite and non-negative")
    if not np.isfinite(max_alert_fraction) or not 0 <= max_alert_fraction < 1:
        raise ValueError("max_alert_fraction must be in [0, 1)")
    if not np.isfinite(step_min) or step_min <= 0:
        raise ValueError("step_min must be positive")
    # Missing telemetry is not healthy exposure. Count only observed score
    # intervals, capped at the nominal cadence and at the declared span.
    gaps = np.diff(score_minutes) / MINUTE
    observed_minutes = min(float((end - start) / MINUTE),
                           float(step_min + np.minimum(gaps, step_min).sum()))
    months = observed_minutes / (30.4375 * 24 * 60)
    allowed = int(np.floor(fa_budget * months + 1e-12))
    base = (detrended_scores(score_minutes, scores, halflife_days, step_min,
                             b_init=b_init)
            if halflife_days is not None else scores)
    candidates = np.unique(np.r_[base, np.nextafter(base, np.inf)])
    for tau in candidates:
        if not np.isfinite(tau):
            continue
        replay = (detrended_scores(score_minutes, scores, halflife_days, step_min,
                                   tau=float(tau), b_init=b_init)
                  if halflife_days is not None else scores)
        if np.mean(replay >= tau) > max_alert_fraction + 1e-12:
            continue
        eps = episodes(score_minutes, replay, float(tau), cooldown_h)
        if len(eps) <= allowed:
            return float(tau), len(eps)
    raise ValueError("no finite candidate satisfies the calibration budget")


def detrended_scores(score_minutes, scores, halflife_days, step_min, tau=None,
                     b_init=None, return_baseline=False, baseline_time=None):
    """EWMA drift compensation: deviation from a trailing baseline.

    Real telemetry drifts (season, wear, slow setpoint changes) enough that a
    fixed threshold is either deaf or constantly alarming -- on MetroPT the
    watch-period BW median sits 0.15 above the calibration median for exactly
    this reason. The detector therefore tracks a trailing EWMA baseline of its
    OWN score and alarms on the deviation.

    tau=None constructs calibration candidates with an always-updating baseline.
    Every candidate and the final watch replay use tau with freeze-on-alert:
    the baseline does not update while deviation >= tau. Threshold selection
    never uses incident labels. Calibration carries both baseline_end and its
    timestamp into watch; alpha uses actual elapsed time between observations.

    b_init: the EWMA value carried over from the calibration slice. A device
    MUST warm its baseline on the calibration data it already holds; starting
    from the first watch score (b_init=None) makes the first days of the watch
    a warm-up artefact. On MetroPT that artefact manufactured a phantom alert
    5.4 days before a real leak that vanished once the baseline was warmed --
    measured in bench_metro_watch, do not reintroduce it.
    """
    score_minutes = _minute_axis(score_minutes)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(score_minutes):
        raise ValueError("scores must be 1-D and aligned with timestamps")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(halflife_days) or halflife_days <= 0:
        raise ValueError("halflife_days must be positive")
    if not np.isfinite(step_min) or step_min <= 0:
        raise ValueError("step_min must be positive")
    if b_init is not None and not np.isfinite(b_init):
        raise ValueError("initial baseline must be finite")
    if tau is not None and np.isnan(tau):
        raise ValueError("threshold cannot be NaN")
    previous = (np.datetime64(baseline_time, "m")
                if baseline_time is not None else None)
    if previous is not None and (np.isnat(previous) or
                                (len(scores) and previous >= score_minutes[0])):
        raise ValueError("baseline timestamp must precede the scored window")
    out = np.empty_like(scores)
    b = float(b_init) if b_init is not None else (
        float(scores[0]) if len(scores) else 0.0)
    elapsed = np.empty(len(scores), dtype=np.float64)
    if len(scores):
        elapsed[0] = (float((score_minutes[0] - previous) / MINUTE)
                      if previous is not None else float(step_min))
        elapsed[1:] = np.diff(score_minutes) / MINUTE
    alphas = 1.0 - 0.5 ** (elapsed / (halflife_days * 24.0 * 60.0))
    for i, s in enumerate(scores):
        alpha = float(alphas[i])
        s = float(s)
        dev = s - b
        out[i] = dev
        if tau is None or dev < tau:
            b = (1.0 - alpha) * b + alpha * s
        previous = score_minutes[i]
    return (out, b) if return_baseline else out


def lead_time_report(incidents, eps):
    """For each labelled incident, the first episode that could be its warning.

    lead_hours is measured from the incident START: positive = advance warning
    before the acute window; 0 = first alert inside the window; None = missed.
    Also reports lead vs the maintenance action when one is recorded.
    """
    out = []
    for inc in incidents:
        s = np.datetime64(inc["start"], "m")
        e = np.datetime64(inc["end"], "m")
        maint = np.datetime64(inc["maintenance"], "m") if inc.get("maintenance") else None
        first = None
        for (a, b, peak) in eps:
            # episode that starts at most 21 days before the incident and
            # overlaps or precedes it
            if a <= e and b >= s - 21 * DAY:
                if a >= s - 21 * DAY and a <= e:
                    first = (a, b, peak)
                    break
        lead = None
        lead_m = None
        if first is not None:
            lead = max(float((s - first[0]) / HOURS), 0.0)
            if maint is not None:
                lead_m = max(float((maint - first[0]) / HOURS), 0.0)
        out.append({
            "name": inc.get("name", "incident"),
            "start": inc["start"], "end": inc["end"],
            "maintenance": inc.get("maintenance"),
            "first_alert": str(first[0]) if first else None,
            "lead_hours_vs_start": lead,
            "lead_hours_vs_maintenance": lead_m,
            "detected": first is not None,
        })
    return out


# ------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(prog="shd_watch",
                                 description=__doc__.split("Example")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_learn = sub.add_parser("learn", help="learn a crystal on healthy minutes")
    p_learn.add_argument("--csv", default=None)
    p_learn.add_argument("--tcol", default=None)
    p_learn.add_argument("--columns", default=None)
    p_learn.add_argument("--npz", default=None, help="minute cache (preferred)")
    p_learn.add_argument("--from", dest="t_from", required=True)
    p_learn.add_argument("--to", dest="t_to", required=True)
    p_learn.add_argument("--m", type=int, default=300)
    p_learn.add_argument("--out", required=True)

    p_cal = sub.add_parser("calibrate", help="pick tau from a FA budget")
    p_cal.add_argument("--crystal", required=True)
    p_cal.add_argument("--csv", default=None)
    p_cal.add_argument("--tcol", default=None)
    p_cal.add_argument("--columns", default=None)
    p_cal.add_argument("--npz", default=None)
    p_cal.add_argument("--from", dest="t_from", required=True)
    p_cal.add_argument("--to", dest="t_to", required=True)
    p_cal.add_argument("--fa-per-month", type=float, default=1.0)
    p_cal.add_argument("--max-alert-fraction", type=float, default=MAX_ALERT_FRACTION,
                       help="maximum fraction of calibration windows in alert (default: 0.05)")
    p_cal.add_argument("--window-min", type=int, default=60)
    p_cal.add_argument("--step-min", type=int, default=30)
    p_cal.add_argument("--cooldown-h", type=float, default=6.0)
    p_cal.add_argument("--out", required=True)

    p_watch = sub.add_parser("watch", help="replay and alarm")
    p_watch.add_argument("--crystal", required=True)
    p_watch.add_argument("--threshold", required=True)
    p_watch.add_argument("--csv", default=None)
    p_watch.add_argument("--tcol", default=None)
    p_watch.add_argument("--columns", default=None)
    p_watch.add_argument("--npz", default=None)
    p_watch.add_argument("--from", dest="t_from", required=True)
    p_watch.add_argument("--to", dest="t_to", required=True)
    p_watch.add_argument("--window-min", type=int, default=None,
                         help="override the calibrated window (default: threshold JSON)")
    p_watch.add_argument("--step-min", type=int, default=None,
                         help="override the calibrated step (default: threshold JSON)")
    p_watch.add_argument("--incidents", default=None,
                         help="JSON file with labelled incidents")

    args = ap.parse_args(argv)
    if bool(args.npz) == bool(args.csv):
        ap.error("provide exactly one input: --npz or --csv")
    if args.csv and (not args.tcol or not args.columns):
        ap.error("--csv requires --tcol and --columns")
    for key in ("window_min", "step_min"):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            ap.error(key.replace("_", "-") + " must be positive")

    if args.cmd == "learn":
        if args.npz:
            minutes, X, feats = load_minute_table(args.npz)
        else:
            minutes, X, feats = read_csv_minutes(
                args.csv, args.tcol, args.columns.split(","))
        lo, hi = slice_range(minutes, args.t_from, args.t_to)
        t0 = time.perf_counter()
        d = build_crystal(X[lo:hi], m=args.m)
        kb = d.save_hologram(args.out)
        print(json.dumps({
            "cmd": "learn", "rows": int(hi - lo), "minutes": int(hi - lo),
            "crystal_kb": round(kb, 1), "sec": round(time.perf_counter() - t0, 1),
            "out": args.out,
        }, indent=1))
        return

    if args.cmd == "calibrate":
        loader = HologramCrystalLoader(args.crystal)
        if args.npz:
            minutes, X, feats = load_minute_table(args.npz)
        elif args.csv and args.tcol and args.columns:
            minutes, X, feats = read_csv_minutes(args.csv, args.tcol,
                                                 args.columns.split(","))
        else:
            sys.exit("input required: --npz minute cache (preferred; build with "
                     "metro_cache.py) or --csv together with --tcol and --columns")
        lo, hi = slice_range(minutes, args.t_from, args.t_to)
        idx, scores = window_scores(loader, X[lo:hi], args.window_min, args.step_min,
                                     minutes=minutes[lo:hi])
        sm = minutes[lo:hi][idx]
        if len(scores) == 0:
            sys.exit("no complete windows in the calibration slice; "
                     "widen the range or lower --window-min")
        if not np.isfinite(scores).all():
            sys.exit("calibration scores contain NaN/Inf; refusing to pick tau")
        # Replay every candidate under the same state machine used by watch.
        tau, n_eps = calibrate_tau(args.t_from, args.t_to, sm, scores,
                                   args.fa_per_month, args.cooldown_h,
                                   max_alert_fraction=args.max_alert_fraction,
                                   halflife_days=HALFLIFE_D, step_min=args.step_min)
        dev_cal, b_end = detrended_scores(sm, scores, HALFLIFE_D, args.step_min,
                                          tau=tau, return_baseline=True)
        payload = {"protocol_version": PROTOCOL_VERSION,
                   "tau": tau, "fa_per_month_budget": args.fa_per_month,
                   "max_alert_fraction": args.max_alert_fraction,
                   "calibration_alert_fraction": float(np.mean(dev_cal >= tau)),
                   "baseline_time": str(sm[-1]),
                   "features": feats,
                   "window_min": args.window_min, "step_min": args.step_min,
                   "cooldown_h": args.cooldown_h,
                   "halflife_days": HALFLIFE_D,
                   "detrended": True,
                   "baseline_end": float(b_end),
                   "calibration_episodes": int(n_eps),
                   "calibration_from": args.t_from, "calibration_to": args.t_to}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(json.dumps(payload, indent=1))
        return

    if args.cmd == "watch":
        loader = HologramCrystalLoader(args.crystal)
        with open(args.threshold, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("protocol_version") != PROTOCOL_VERSION:
            sys.exit("legacy threshold protocol: re-run calibrate before watch")
        missing = [k for k in ("tau", "window_min", "step_min", "cooldown_h",
                               "features", "calibration_to", "baseline_end",
                               "baseline_time", "halflife_days") if k not in cfg]
        if missing:
            sys.exit("threshold file is missing keys ({}); re-run calibrate"
                     .format(", ".join(missing)))
        tau = float(cfg["tau"])
        if not np.isfinite(tau):
            sys.exit("threshold must be finite; re-run calibrate")
        for key in ("window_min", "step_min"):
            override = getattr(args, key)
            if override is not None and override != int(cfg[key]):
                sys.exit("changing calibrated window/step requires re-running calibrate")
        # The calibrated window and cadence are part of the threshold contract.
        win = (int(args.window_min) if args.window_min is not None
               else int(cfg.get("window_min", 60)))
        step = (int(args.step_min) if args.step_min is not None
                else int(cfg.get("step_min", 30)))
        cooldown_h = float(cfg.get("cooldown_h", 6.0))
        if args.npz:
            minutes, X, feats = load_minute_table(args.npz)
        elif args.csv and args.tcol and args.columns:
            minutes, X, feats = read_csv_minutes(args.csv, args.tcol,
                                                 args.columns.split(","))
        else:
            sys.exit("input required: --npz minute cache (preferred; build with "
                     "metro_cache.py) or --csv together with --tcol and --columns")
        lo, hi = slice_range(minutes, args.t_from, args.t_to)
        Xw, mw = X[lo:hi], minutes[lo:hi]
        t0 = time.perf_counter()
        if feats != cfg.get("features"):
            sys.exit("feature order differs from calibration; re-run calibrate")
        if np.datetime64(args.t_from, "m") < np.datetime64(cfg["calibration_to"], "m"):
            sys.exit("watch must start after the calibration slice")
        idx, scores = window_scores(loader, Xw, win, step, minutes=mw)
        sm = mw[idx]
        if not len(scores):
            sys.exit("no complete contiguous windows in the watch slice")
        b_init = cfg["baseline_end"]
        halflife = float(cfg["halflife_days"])
        detrended = True
        dev = detrended_scores(sm, scores, halflife, step, tau=tau,
                               b_init=b_init, baseline_time=cfg["baseline_time"])
        eps = episodes(sm, dev, tau, cooldown_h)
        report = {
            "watch_from": args.t_from, "watch_to": args.t_to,
            "windows": int(len(scores)), "episodes": len(eps),
            "tau": tau,
            "detrended": detrended,
            "window_min": win, "step_min": step,
            "scan_sec": round(time.perf_counter() - t0, 1),
            "alerts": [{"start": str(a), "end": str(b), "peak_bw": round(p, 4)}
                       for a, b, p in eps],
        }
        if args.incidents:
            with open(args.incidents, encoding="utf-8") as fh:
                incidents = json.load(fh)
            report["incidents"] = lead_time_report(incidents, eps)
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

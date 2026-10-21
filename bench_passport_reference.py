"""Does a background-only spectral passport fix the reference contamination?

The background-only passport (``passport_reference="background"``) was added
because a whole-stream passport absorbs a shifted regime that occupies part
of the TRAINING data: the reference then contains the shift, and healthy
windows look anomalous. The claim was measured on one synthetic 20k stream
(100% of normal windows flagged). This benchmark asks the question the claim
leaves open: does the background reference fix the false alarms WITHOUT
losing the detection of the shift?

Two confounds are separated explicitly, because conflating them is how a
radar "fix" gets believed without evidence:

* **contamination** -- the training stream contains a fraction f of the
  shifted (critical) regime. Measured on windows held out from the SAME time
  span as the training data, so the temporal drift cannot explain it.
* **temporal drift** -- the evaluation windows come from a LATER period.
  Measured separately, because a reference that does not survive seasonal
  drift is a different failure than a contaminated reference.

Arms (identical protocol, seeds vary):

* ``synthetic`` -- 8-dim stationary base stream plus a shifted regime that is
  the rare high-target phase (the case the passport exists for).
* ``pm25`` -- the real Beijing PM2.5 telemetry cached in ``data/pm25.csv``
  with the repository's own 14-feature construction, time-ordered.

For every contamination level both references are distilled from the SAME
data with the SAME seed, then the radar is scored on held-out normal windows
(false alarms) and held-out shifted windows (detection).

Writes ``bench_passport_reference_results.json`` next to this file.

    python bench_passport_reference.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
sys.path.insert(0, REPO)

from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402

SEEDS_SYNTH = 10
SEEDS_PM25 = 5
CONTAM = [0.0, 0.10, 0.20, 0.40]
REFERENCES = ["stream", "background"]
CRYSTAL = 200
WINDOW = 300
N_WINDOWS = 20
BATCH = 2500
TAIL_Q = 84.0
BAND = 0.26
CRIT_Q = 97.0


# ------------------------------------------------------------------ datasets
def synth_arm(seed: int):
    """Stationary base + a shifted regime that is the rare high-target phase.

    Returns (base_train, base_y, base_eval, shift_train, shift_y, shift_eval).
    """
    rng = np.random.default_rng(seed)
    n, d = 20000, 8
    X = rng.normal(size=(n, d))
    y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * rng.normal(size=n)
    regime = rng.normal(loc=1.8, scale=1.7, size=(n, d))
    y_regime = y + 6.0
    cut = int(0.8 * n)
    cut2 = int(0.9 * n)
    # base_train: first 80%. base_eval: the next 10% -- held out, same period
    # (so temporal drift cannot explain a false alarm). The regime rows are a
    # different draw: shift_train is the contamination source, shift_eval is
    # never trained on.
    return {
        "base_train": X[:cut], "y_base_train": y[:cut],
        "base_eval": X[cut:cut2],
        "shift_src": regime[:cut2], "y_shift_src": y_regime[:cut2],
        "eval_shift": regime[cut2:], "eval_drift": None,
    }


def pm25_arm(seed: int, feature_set: str = "full"):
    """Real PM2.5 telemetry with the repository's own 14-feature construction.

    The file in ``data/pm25.csv`` has the UCI PRSA column layout:
    No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir.
    The cache on disk is a truncated slice of the 2010-2014 series, which is
    stated in the results so the numbers are not read as paper-grade.

    ``feature_set="full"`` keeps the repository's 14 features, which include
    the RAW calendar columns (year, month, day, hour) -- designed for
    regression, not for a covariance radar. ``feature_set="sensors"`` drops
    them and keeps the six physical sensors. Comparing the two isolates
    whether the radar reacts to the operating regime or to the calendar.
    """
    path = os.path.join(REPO, "data", "pm25.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    cbwd_values = ["NW", "NE", "SE", "cv"]
    data = []
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        try:
            y = float(r[5])
            nums = [float(r[i]) for i in (1, 2, 3, 4, 6, 7, 8, 10, 11, 12)]
        except (ValueError, IndexError):
            continue
        cb = r[9]
        if cb not in cbwd_values:
            continue
        if feature_set == "sensors":
            # DEWP, TEMP, PRES, Iws, Is, Ir -- no calendar, no one-hot
            feats = nums[4:10]
        else:
            onehot = [1.0 if cb == v else 0.0 for v in cbwd_values]
            feats = nums + onehot
        data.append(feats + [y])
    if not data:
        return None
    arr = np.asarray(data, dtype=np.float64)
    X, y = arr[:, :-1], arr[:, -1]
    if feature_set == "full":
        # the file is already time-ordered (UCI export); keep it, but sort
        # defensively with a lexicographic key on the calendar columns
        key = np.lexsort((X[:, 3], X[:, 2], X[:, 1], X[:, 0]))
        X, y = X[key], y[key]

    n = len(X)
    n_future = int(0.20 * n)
    train_pool, future = X[:-n_future], X[-n_future:]
    y_train_pool, y_future = y[:-n_future], y[-n_future:]
    cut = int(0.80 * len(train_pool))
    base_train, base_eval = train_pool[:cut], train_pool[cut:]
    y_base_train = y_train_pool[:cut]

    # The later segment is a different operating regime (other season, other
    # sensor drift). Half of it is the contamination source, half is the
    # detection population -- they must be disjoint, otherwise the detector
    # would only be re-seeing its own training rows.
    half = len(future) // 2
    shift_src, eval_shift = future[:half], future[half:]
    y_shift_src = y_future[:half]
    thr = float(np.percentile(y_train_pool, CRIT_Q))
    eval_drift = eval_shift[y_future[half:] < thr]
    if len(eval_drift) < WINDOW:
        eval_drift = None
    if len(eval_shift) < WINDOW or len(shift_src) < 10:
        return None
    return {
        "base_train": base_train, "y_base_train": y_base_train,
        "base_eval": base_eval,
        "shift_src": shift_src, "y_shift_src": y_shift_src,
        "eval_shift": eval_shift, "eval_drift": eval_drift,
    }


# ------------------------------------------------------------------ protocol
def build_crystal(X, y, reference, seed, tmp):
    d = SpectralHologramDistiller(
        crystal_size=CRYSTAL, high_band_ratio=BAND, sketch_dim=256,
        tail_quantile=TAIL_Q, phase_kappa=3.2, random_state=seed,
        passport_reference=reference, max_init_buffer=8192,
    )
    for i in range(0, len(X), BATCH):
        d.partial_fit(X[i:i + BATCH], y[i:i + BATCH])
    d.finalize_crystal()
    path = os.path.join(tmp, f"c_{reference}_{seed}.hologram.npz")
    d.save_hologram(path, include_projector=True)
    return HologramCrystalLoader(path)


def score_windows(loader, X, n_windows, rng):
    """Score held-out windows: (median BW, alert rate, threshold)."""
    if len(X) < WINDOW:
        return None
    bw, thr, flags = [], [], []
    for _ in range(n_windows):
        start = int(rng.integers(0, len(X) - WINDOW + 1))
        rep = loader.check_domain_shift(X[start:start + WINDOW])
        bw.append(float(rep["bw_distance"]))
        thr.append(float(rep["threshold"]))
        flags.append(bool(rep["is_shifted"]))
    return {
        "median_bw": float(np.median(bw)),
        "alert_rate": float(np.mean(flags)),
        "median_threshold": float(np.median(thr)),
    }


def run_arm(arm_name, builder, n_seeds):
    out = {}
    tmp = os.path.join(REPO, ".bench_tmp")
    os.makedirs(tmp, exist_ok=True)
    for f in CONTAM:
        for reference in REFERENCES:
            key = f"{reference}_f{f:.2f}"
            rows = []
            for seed in range(n_seeds):
                built = builder(seed)
                if built is None:
                    continue
                base_tr = built["base_train"]
                y_base_tr = built["y_base_train"]
                base_eval = built["base_eval"]
                shift_tr = built["shift_src"]
                y_shift_tr = built["y_shift_src"]
                shift_eval = built["eval_shift"]
                drift_X = built["eval_drift"]
                rng = np.random.default_rng(1000 + seed)
                n_shift = int(round(f * len(base_tr)))
                if n_shift > 0 and len(shift_tr) > 0:
                    take = min(n_shift, len(shift_tr))
                    idx = rng.choice(len(shift_tr), size=take, replace=False)
                    train_X = np.vstack([base_tr, shift_tr[idx]])
                    train_y = np.concatenate([y_base_tr, y_shift_tr[idx]])
                else:
                    train_X, train_y = base_tr, y_base_tr
                loader = build_crystal(train_X, train_y, reference, seed, tmp)
                normal = score_windows(loader, base_eval, N_WINDOWS, rng)
                shifted = score_windows(loader, shift_eval, N_WINDOWS, rng)
                drift = (score_windows(loader, drift_X, N_WINDOWS, rng)
                         if drift_X is not None else None)
                # control: windows that ARE in the training stream. If the
                # radar flags these too, it is broken by construction; if it
                # does not, the out-of-sample alarms are reference drift.
                insample = score_windows(loader, base_tr, N_WINDOWS, rng)
                if normal is None or shifted is None:
                    continue
                row = {
                    "seed": seed,
                    "n_train": int(len(train_X)),
                    "normal": normal,
                    "shifted": shifted,
                }
                if insample is not None:
                    row["insample"] = insample
                if drift is not None:
                    row["drift"] = drift
                rows.append(row)
            if not rows:
                print(f"  {arm_name:9s} {key:22s} (no usable windows)")
                continue
            entry = {
                "n_seeds": len(rows),
                "fa_rate_median": float(np.median([r["normal"]["alert_rate"] for r in rows])),
                "fa_rate_max": float(np.max([r["normal"]["alert_rate"] for r in rows])),
                "detect_median": float(np.median([r["shifted"]["alert_rate"] for r in rows])),
                "detect_min": float(np.min([r["shifted"]["alert_rate"] for r in rows])),
                "normal_bw_median": float(np.median([r["normal"]["median_bw"] for r in rows])),
                "shifted_bw_median": float(np.median([r["shifted"]["median_bw"] for r in rows])),
                "threshold_median": float(np.median([r["normal"]["median_threshold"] for r in rows])),
            }
            if "drift" in rows[0]:
                entry["drift_fa_median"] = float(np.median([r["drift"]["alert_rate"] for r in rows]))
                entry["drift_bw_median"] = float(np.median([r["drift"]["median_bw"] for r in rows]))
            if "insample" in rows[0]:
                entry["insample_fa_median"] = float(np.median([r["insample"]["alert_rate"] for r in rows]))
                entry["insample_bw_median"] = float(np.median([r["insample"]["median_bw"] for r in rows]))
            out[key] = entry
            extra = f" DRIFT_FA={entry['drift_fa_median']:.2f}" if "drift_fa_median" in entry else ""
            extra += (f" IN_FA={entry['insample_fa_median']:.2f}"
                      if "insample_fa_median" in entry else "")
            print(f"  {arm_name:9s} {key:22s} "
                  f"FA={entry['fa_rate_median']:.2f} "
                  f"DET={entry['detect_median']:.2f} "
                  f"BW_n={entry['normal_bw_median']:.3f} "
                  f"BW_s={entry['shifted_bw_median']:.3f} "
                  f"thr={entry['threshold_median']:.3f}{extra}")
    return out


def main():
    t0 = time.perf_counter()
    res = {
        "protocol": {
            "crystal_size": CRYSTAL, "window": WINDOW, "n_windows": N_WINDOWS,
            "tail_quantile": TAIL_Q, "high_band_ratio": BAND, "crit_quantile": CRIT_Q,
            "seeds_synthetic": SEEDS_SYNTH, "seeds_pm25": SEEDS_PM25,
            "contamination": CONTAM,
            "note": "normal windows are held out from the training stream; the "
                    "pm25 arm additionally reports a drifted (later-period) "
                    "normal population; the pm25 cache in data/pm25.csv is a "
                    "TRUNCATED slice of the 2010-2014 series, so its absolute "
                    "numbers are not paper-grade",
        },
        "arms": {},
    }
    print("== synthetic ==")
    res["arms"]["synthetic"] = run_arm("synthetic", synth_arm, SEEDS_SYNTH)
    print("== pm25 (real telemetry, 14 features incl. raw calendar) ==")
    res["arms"]["pm25_full"] = run_arm(
        "pm25_full", lambda s: pm25_arm(s, "full"), SEEDS_PM25)
    print("== pm25 (real telemetry, sensors only) ==")
    res["arms"]["pm25_sensors"] = run_arm(
        "pm25_sens", lambda s: pm25_arm(s, "sensors"), SEEDS_PM25)

    verdict = {}
    for arm, data in res["arms"].items():
        for f in CONTAM:
            s = data.get(f"stream_f{f:.2f}")
            b = data.get(f"background_f{f:.2f}")
            if not s or not b:
                continue
            fa_s, fa_b = s["fa_rate_median"], b["fa_rate_median"]
            det_s, det_b = s["detect_median"], b["detect_median"]
            # A tie at FA=1.00 is NOT a win: both references flag everything,
            # so the comparison carries no information. Report that plainly.
            if fa_s >= 0.99 and fa_b >= 0.99:
                label = "both_broken_fa_1.00"
            elif fa_b < fa_s and det_b >= det_s:
                label = "background_better"
            elif fa_b > fa_s and det_b <= det_s:
                label = "background_worse"
            else:
                label = "no_difference"
            verdict[f"{arm}_f{f:.2f}"] = {
                "fa_stream": fa_s, "fa_background": fa_b,
                "det_stream": det_s, "det_background": det_b,
                "insample_fa_stream": s.get("insample_fa_median"),
                "insample_fa_background": b.get("insample_fa_median"),
                "label": label,
                "background_wins": label == "background_better",
            }
    res["verdict"] = verdict
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)

    out = os.path.join(HERE, "bench_passport_reference_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("\n== verdict ==")
    for k, v in verdict.items():
        print(f"  {k:22s} FA {v['fa_stream']:.2f} -> {v['fa_background']:.2f} | "
              f"DET {v['det_stream']:.2f} -> {v['det_background']:.2f} | "
              f"in-sample FA {v['insample_fa_stream']} | {v['label']}")
    print(f"\nwrote {out} in {res['elapsed_s']}s")


if __name__ == "__main__":
    main()

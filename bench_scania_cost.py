"""Operational benchmark: failure-detection COST, not RMSE (Scania APS).

The audit's third hole: every existing benchmark scores regression error, while a
maintenance buyer pays for avoided failures and wasted inspections. The UCI
"APS Failure at Scania Trucks" experiment here uses a CUSTOM cost scenario,
not the official UCI challenge metric (500 * FN + 10 * FP):

    cost = 10 * false_negatives + 1 * false_positive

i.e. a missed failure costs ten times a false alarm in THIS scenario. UCI uses
a fifty-to-one ratio: https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks
Official-metric results require retuning thresholds and rerunning; changing the
label on the old JSON is not valid. Thresholds are retuned on the validation
slice for whatever cost model is passed, so the official metric is a
first-class rerun via --fn-cost/--fp-cost, never a relabel. This benchmark asks:

    at an equal 300-row budget, does a distilled crystal buy a cheaper
    maintenance decision than a random subsample or KMeans?

Protocol
--------
* Official train/test files (60,000 / 16,000 rows, 170 features plus label).
* Training is split into 48,000 fit rows and 12,000 threshold-validation rows.
* The decision threshold is chosen on a held-out slice of TRAIN, never on test.
* Same classifier (HistGradientBoostingClassifier) for every coreset.
* Arms: random 300, KMeans 300, SHD 300, plus full-train (oracle) and the
  trivial "never alert" policy for reference.

Usage:
    python bench_scania_cost.py --seeds 10                # custom 10:1 scenario (historical default)
    python bench_scania_cost.py --seeds 10 --fn-cost 500 --fp-cost 10 \
        --output bench_scania_cost_official_results.json  # official UCI challenge metric
Writes the JSON named by --output (default: bench_scania_cost_results.json,
the preserved historical custom-scenario artifact).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
import zipfile

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import build_kmeans, build_random
from shd import SpectralHologramDistiller

URL = ("https://archive.ics.uci.edu/static/public/421/"
       "aps+failure+at+scania+trucks.zip")
FN_COST, FP_COST = 10, 1


def _read_member(zf, suffix):
    name = [n for n in zf.namelist() if n.endswith(suffix)][0]
    raw = zf.read(name).decode("utf-8", "replace")
    lines = raw.splitlines()[20:]            # 20 metadata lines before the header
    return "\n".join(lines)


def load_scania(cache=r"C:\Users\User\Downloads\New folder\_datasets\scania_aps.zip"):
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with urllib.request.urlopen(URL) as r, open(cache, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(cache) as zf:
        tr = _read_member(zf, "training_set.csv")
        te = _read_member(zf, "test_set.csv")

    def parse(text):
        head, *rows = text.splitlines()
        cols = [c.strip() for c in head.split(",")]
        body = [r for r in rows if r.strip()]
        arr = np.empty((len(body), len(cols) - 1), dtype=np.float64)
        cls = np.zeros(len(body), dtype=np.int8)
        for i, row in enumerate(body):
            parts = row.split(",")
            cls[i] = 1 if parts[0].strip() == "pos" else 0
            for j, v in enumerate(parts[1:]):
                v = v.strip()
                arr[i, j] = np.nan if v == "na" else float(v)
        return arr, cls, cols[1:]

    X_tr, y_tr, names = parse(tr)
    X_te, y_te, _ = parse(te)
    return X_tr, y_tr, X_te, y_te, names


def pick_threshold(scores, y, fn_cost=FN_COST, fp_cost=FP_COST):
    """Exact cost-minimising threshold, tie-safe.

    Sweeps the *actual* achievable operating points (one per distinct score) plus
    the "alert never" policy, and evaluates each with the real ``>=`` rule the
    scorer will apply. A naive positional sweep (cut index k) is wrong here: with
    tied scores ``scores >= thr`` selects the whole tie group, so a degenerate
    constant-score model would be scored as "alerts on everything" instead of
    "alerts on nothing". Returns ``+inf`` when never alerting is optimal.
    """
    n = len(scores)
    total_pos = int(y.sum())
    order = np.argsort(scores)[::-1]              # descending scores
    s_desc = scores[order]
    y_desc = y[order].astype(np.int64)
    cum_pos = np.cumsum(y_desc)

    # End index of each distinct-score group: predicting every point with
    # score >= that group's value.
    change = np.r_[True, s_desc[1:] != s_desc[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:] - 1, n - 1]

    n_pred = (ends + 1).astype(np.float64)
    pos_pred = cum_pos[ends].astype(np.float64)
    fn = total_pos - pos_pred
    fp = n_pred - pos_pred
    cost = fn_cost * fn + fp_cost * fp

    # Candidate: alert on nothing (cheapest when the model is uninformative).
    cost_none = float(fn_cost * total_pos)
    best_i = int(np.argmin(cost))
    if cost_none <= cost[best_i]:
        return float("inf"), int(cost_none)
    return float(s_desc[ends[best_i]]), int(cost[best_i])


def evaluate(scores, y, thr, fn_cost=FN_COST, fp_cost=FP_COST):
    pred = (scores >= thr).astype(np.int8)
    fn = int(np.sum((pred == 0) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tp = int(np.sum((pred == 1) & (y == 1)))
    return {
        "cost": int(fn_cost * fn + fp_cost * fp),
        "fn": fn, "fp": fp, "tp": tp,
        "recall": float(tp / max(1, tp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "alarms": int(tp + fp),
    }


def build_shd(X_tr, y_tr, seed, m, batch=2500):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed, tail="high",
    )
    yf = y_tr.astype(np.float64)
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i : i + batch], yf[i : i + batch])
    d.finalize_crystal()
    return (d.crystal_X, np.rint(np.asarray(d.crystal_Y, dtype=np.float64)).astype(np.int8),
            d.gamma_weights)


def fit_score(Xc, yc, wc, X_fit, X_eval, model="hgb", scaler=None):
    """Train on the coreset, return scores on X_fit and X_eval.

    Two very different learners are reported so the conclusion does not hinge on
    one model family: HGB (axis-aligned trees, handles the 170 raw-scale features
    directly) and a balanced linear model on standardized features. Both use the
    same coreset and the same threshold-selection protocol.
    """
    if len(np.unique(yc)) < 2:                    # degenerate coreset: all one class
        const = float(yc.mean()) if len(yc) else 0.0
        return np.full(len(X_fit), const), np.full(len(X_eval), const)

    if model == "logit":
        from sklearn.linear_model import LogisticRegression
        Zc = scaler.transform(Xc)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Zc, yc, sample_weight=wc)
        return (clf.predict_proba(scaler.transform(X_fit))[:, 1],
                clf.predict_proba(scaler.transform(X_eval))[:, 1])

    clf = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=42)
    clf.fit(Xc, yc, sample_weight=wc)
    return clf.predict_proba(X_fit)[:, 1], clf.predict_proba(X_eval)[:, 1]


def run(seeds, m=300, fn_cost=FN_COST, fp_cost=FP_COST):
    X_tr, y_tr, X_te, y_te, _ = load_scania()
    print("Scania APS: train={:,} (pos {:.2%})  test={:,} (pos {:.2%})  d={}".format(
        len(X_tr), y_tr.mean(), len(X_te), y_te.mean(), X_tr.shape[1]), flush=True)

    # Threshold selection slice, carved from TRAIN only.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.20, random_state=42, stratify=y_tr)

    # Median imputation, fitted on the fit slice only and applied to every arm.
    # KMeans and SHD cannot consume NaN while HGB could, so imputing keeps the
    # comparison identical across coresets instead of favouring one of them.
    imputer = SimpleImputer(strategy="median").fit(X_fit)
    X_fit = imputer.transform(X_fit)
    X_val = imputer.transform(X_val)
    X_te = imputer.transform(X_te)
    Z_fit = StandardScaler().fit(X_fit).transform(X_fit)
    scaler_all = StandardScaler().fit(X_fit)

    arms = ["random", "kmeans", "shd"]
    models = ["hgb", "logit"]
    records = []
    t0 = time.perf_counter()

    # Oracle + trivial reference, seed-independent.
    oracle = {}
    never = evaluate(np.zeros(len(y_te)), y_te, 1.0, fn_cost, fp_cost)
    for model in models:
        sc_val, sc_te = fit_score(X_fit, y_fit, None, X_val, X_te,
                                  model=model, scaler=scaler_all)
        thr, _ = pick_threshold(sc_val, y_val, fn_cost, fp_cost)
        oracle[model] = evaluate(sc_te, y_te, thr, fn_cost, fp_cost)

    for seed in range(seeds):
        rec = {"seed": seed}
        for arm in arms:
            if arm == "random":
                Xc, yc, wc = build_random(X_fit, y_fit, seed, m)
                wc = None
            elif arm == "kmeans":
                Xc, yc, wc = build_kmeans(X_fit, y_fit, Z_fit, seed, m)
                yc = np.rint(yc).astype(np.int8)
            else:
                Xc, yc, wc = build_shd(X_fit, y_fit, seed, m)
            for model in models:
                sc_val, sc_te = fit_score(Xc, yc, wc, X_val, X_te,
                                          model=model, scaler=scaler_all)
                thr, _ = pick_threshold(sc_val, y_val, fn_cost, fp_cost)
                rec["{}_{}".format(arm, model)] = evaluate(sc_te, y_te, thr,
                                                          fn_cost, fp_cost)
            rec["pos_" + arm] = int(yc.sum())
        records.append(rec)
        print("  seed {:2d}: cost(hgb) random={:6d} kmeans={:6d} shd={:6d} | "
              "cost(logit) random={:6d} kmeans={:6d} shd={:6d} | pos r/k/s={}/{}/{}".format(
                  seed,
                  rec["random_hgb"]["cost"], rec["kmeans_hgb"]["cost"], rec["shd_hgb"]["cost"],
                  rec["random_logit"]["cost"], rec["kmeans_logit"]["cost"], rec["shd_logit"]["cost"],
                  rec["pos_random"], rec["pos_kmeans"],
                  rec["pos_shd"]), flush=True)

    med = lambda key, k: float(np.median([r[key][k] for r in records]))
    med_scalar = lambda key: float(np.median([r[key] for r in records]))
    tests = {}
    for model in models:
        for arm in ("random", "kmeans"):
            a = np.array([r["shd_" + model]["cost"] for r in records])
            b = np.array([r[arm + "_" + model]["cost"] for r in records])
            diff = a - b
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(a, b).pvalue)
            tests["shd_vs_{}_{}_cost".format(arm, model)] = {
                "median_shd": float(np.median(a)),
                "median_" + arm: float(np.median(b)),
                "wins_shd": int(np.sum(diff < 0)), "p_value": p,
            }

    return {
        "dataset": "scania_aps",
        "title": "APS failure cost (FN={}, FP={})".format(fn_cost, fp_cost),
        "m": int(m), "n_seeds": len(records), "seeds": list(range(len(records))),
        "n_train": int(len(X_fit)), "n_test": int(len(X_te)),
        "pos_rate_train": float(y_fit.mean()), "pos_rate_test": float(y_te.mean()),
        "cost_model": {"fn": int(fn_cost), "fp": int(fp_cost)},
        "models": models,
        "oracle_fulltrain": oracle, "never_alert": never,
        "summary": {
            "{}_{}".format(arm, model): {
                k: med("{}_{}".format(arm, model), k)
                for k in ("cost", "fn", "fp", "recall", "precision", "alarms")
            }
            for arm in arms for model in models
        },
        "crystal_positives": {arm: med_scalar("pos_" + arm) for arm in arms},
        "wilcoxon": tests, "records": records,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--fn-cost", type=int, default=FN_COST,
                    help="cost of a missed failure (default %(default)s, the "
                         "historical custom scenario; official UCI metric uses 500)")
    ap.add_argument("--fp-cost", type=int, default=FP_COST,
                    help="cost of a false alarm (default %(default)s; official UCI metric uses 10)")
    ap.add_argument("--output", default="bench_scania_cost_results.json")
    args = ap.parse_args()
    res = run(args.seeds, args.m, fn_cost=args.fn_cost, fp_cost=args.fp_cost)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 88)
    print("SCANIA APS: maintenance cost at an equal {}-row budget ({} seeds)".format(
        res["m"], res["n_seeds"]))
    print("cost = {}*FN + {}*FP;  never-alert = {:d}".format(
        args.fn_cost, args.fp_cost, res["never_alert"]["cost"]))
    print("=" * 88)
    print("{:<20s} {:>9s} {:>6s} {:>6s} {:>8s} {:>8s}".format(
        "coreset / model", "cost", "FN", "FP", "recall", "prec"))
    for model in res["models"]:
        o = res["oracle_fulltrain"][model]
        print("{:<20s} {:>9d} {:>6d} {:>6d} {:>8.3f} {:>8.3f}".format(
            "full train / " + model, o["cost"], o["fn"], o["fp"], o["recall"], o["precision"]))
    for model in res["models"]:
        for arm in ("random", "kmeans", "shd"):
            s = res["summary"]["{}_{}".format(arm, model)]
            # One decimal: 10-seed medians of integer counts are fractional
            # (e.g. cost 1271.5, fn 77.5); '{:.0f}' would banker-round them.
            print("{:<20s} {:>9.1f} {:>6.1f} {:>6.1f} {:>8.3f} {:>8.3f}".format(
                arm + " / " + model, s["cost"], s["fn"], s["fp"], s["recall"], s["precision"]))
    print("crystal positives (median of 300 rows): " + ", ".join(
        "{}={:.1f}".format(a, v) for a, v in res["crystal_positives"].items()))
    for key, v in res["wilcoxon"].items():
        print("  {}: wins={}/{}  p={:.3g}".format(
            key, v["wins_shd"], res["n_seeds"], v["p_value"]))
    print("Saved -> {} ({:.0f}s)".format(args.output, res["elapsed_sec"]))


if __name__ == "__main__":
    main()

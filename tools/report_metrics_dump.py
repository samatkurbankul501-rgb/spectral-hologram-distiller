"""Dump key metrics from the four bench JSON artifacts (v3.13.0 report inputs)."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def med_iqr(vals):
    v = np.array(vals, dtype=float)
    m = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    return m, iqr


def jsonload(path):
    return json.load(open(path, encoding="utf-8"))


def dump_seeds():
    d = jsonload(ROOT + "/bench_seeds_results.json")
    for k, v in d["results"].items():
        print("=== " + k + " (" + v["title"] + ")")
        print("n_seeds:", v["n_seeds"], "| oracle:", v["oracle"])
        recs = v["records"]
        for side in ("random", "kmeans", "kt", "skt"):
            for f in ("krr_rmse_crit", "gbm_rmse_crit", "krr_rmse_all", "gbm_rmse_all", "krr_r2", "gbm_r2"):
                for model_side in ("shd", side):
                    vals = [r[model_side][f] for r in recs if f in r[model_side]]
                    if vals:
                        m, iqr = med_iqr(vals)
                        print(f"{model_side:6s} {f:14s} median={m:.3f} iqr={iqr:.3f} (n={len(vals)})")
        w = v["wilcoxon"]
        for name, e in w.items():
            print(name, "-> wins %d/%d p=%.3g" % (e["wins_shd"], len(recs), e["p_value"]))


def dump_metro():
    d = jsonload(ROOT + "/bench_metro_watch_v2_results.json")
    arms = d["results"]["metropt3"]["arms"] if "results" in d else d.get("arms")
    print("=== metro v2 arms:")
    for a in arms:
        print(a["arm"], "| tau=%.3f | calib_episodes=%s | FA/month=%.3f | calib_windows=%s | watch_windows=%s" % (
            a["tau"], a.get("calibration_episodes"), a.get("unmatched_fa_per_month", 0.0),
            a.get("calibration_windows"), a.get("watch_windows")))
        print("   other keys:", {k: v for k, v in a.items() if k not in (
            "arm", "tau", "calibration_episodes", "unmatched_fa_per_month",
            "calibration_windows", "watch_windows", "detrended")})


def dump_pm25():
    d = jsonload(ROOT + "/bench_pm25_results.json")
    print("pm25 top keys:", list(d.keys()))
    res = d.get("results", d)
    if "pm25" in res:
        res = res["pm25"]
    print("=== pm25")
    print("n_seeds:", res.get("n_seeds"), "oracle:", res.get("oracle"))
    recs = res["records"]
    for side in ("random", "kmeans", "kt", "skt"):
        for f in ("krr_rmse_crit", "gbm_rmse_crit", "krr_rmse_all", "gbm_rmse_all", "krr_r2", "gbm_r2"):
            for model_side in ("shd", side):
                vals = [r[model_side][f] for r in recs if f in r[model_side]]
                if vals:
                    m, iqr = med_iqr(vals)
                    print(f"{model_side:6s} {f:14s} median={m:.3f} iqr={iqr:.3f}")
    for name, e in res["wilcoxon"].items():
        print(name, "-> wins %d/%d p=%.3g" % (e["wins_shd"], len(recs), e["p_value"]))


def dump_cmapss():
    d = jsonload(ROOT + "/bench_cmapss_results.json")
    res = d["results"]["fd001"]
    print("=== cmapss fd001")
    print("n_seeds:", res["n_seeds"], "oracle:", res.get("oracle"), "m:", res.get("m"))
    recs = res["records"]
    print("record sides:", list(recs[0].keys()))
    for side in ("random", "kmeans", "kt", "skt"):
        for f in ("krr_rmse_crit", "gbm_rmse_crit", "krr_rmse_all", "gbm_rmse_all"):
            if side in recs[0]:
                vals = [r[side][f] for r in recs if f in r[side]]
                if vals:
                    m, iqr = med_iqr(vals)
                    print(f"{side:8s} {f:14s} median={m:.3f} iqr={iqr:.3f}")
    for br in ("0.26", "0.5", "0.65"):
        side_key = "shd_r" + br
        if side_key in recs[0]:
            for f in ("krr_rmse_crit", "gbm_rmse_crit", "krr_rmse_all", "gbm_rmse_all", "krr_r2", "gbm_r2"):
                key = side_key + "_" + f
                if key in recs[0]:
                    vals = [r[side_key][f] for r in recs if f in r[side_key]]
                    if vals:
                        m, iqr = med_iqr(vals)
                        print(f"{side_key:10s} {f:14s} median={m:.3f} iqr={iqr:.3f}")
    vo = res.get("vs_oracle", {})
    for br in ("0.26", "0.5", "0.65"):
        blk = vo.get("shd_r" + br, {})
        for metric in ("krr_rmse_crit", "krr_rmse_all", "gbm_rmse_crit", "gbm_rmse_all"):
            if metric in blk:
                e = blk[metric]
                print("vs_oracle shd_r%s %s: shd=%.3f oracle=%.3f wins=%d/%d" % (
                    br, metric, e["median_shd"], e["oracle"], e["wins_vs_oracle"], res["n_seeds"]))


if __name__ == "__main__":
    dump_seeds()
    dump_metro()
    dump_pm25()
    dump_cmapss()

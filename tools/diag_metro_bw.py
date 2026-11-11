"""MetroPT-3 radar diagnostic: is BW distance insensitive to mean shifts?

Loads the minute cache + incidents, scores contiguous 60-min windows the same
way shd_watch does (60-min windows, 30-min step), and reports median BW
distance for healthy vs each incident span. If incident medians are not
clearly above healthy medians, the SHD radar is structurally blind to
mean-level leak signatures (covariance spectrum alone cannot see a mean
shift), which explains the 0/4 negative result.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shd import HologramCrystalLoader
from shd_watch import (DAY, calibrate_tau, detrended_scores, episodes,
                       load_minute_table, slice_range, window_scores, window_bounds)

_DATASETS = os.environ.get(
    "SHD_DATASETS", os.path.normpath(os.path.join(ROOT, "..", "..", "_datasets")))
NPZ = os.path.join(_DATASETS, "metro_minute_cache.npz")
INC = os.path.join(ROOT, "metro_incidents.json")
LEARN = ("2020-02-01", "2020-03-15")
WATCH = ("2020-04-10", "2020-08-01")
WIN, STEP = 60, 30

def run():
    minutes, X, feats = load_minute_table(NPZ)
    incidents = json.load(open(INC, encoding="utf-8"))
    l_lo, l_hi = slice_range(minutes, *LEARN)
    w_lo, w_hi = slice_range(minutes, *WATCH)
    loader = HologramCrystalLoader(os.path.join(ROOT, "metro_demo_v2_crystal.hologram.npz"))
    mw = minutes[w_lo:w_hi]
    idx_w, sc_w = window_scores(loader, X[w_lo:w_hi], WIN, STEP, minutes=mw)

    def median_between(a, b):
        m = np.asarray(a, dtype="datetime64[m]")
        n = np.asarray(b, dtype="datetime64[m]")
        mask = (wends >= m) & (wends <= n)
        v = sc_w[mask]
        return float(np.median(v)) if len(v) else float("nan"), int(mask.sum())

    # healthy baseline: the whole watch span minus incident spans (per window end) (per window end)
    wends = mw[idx_w]
    spans = [(np.datetime64(i["start"], "m"), np.datetime64(i["end"], "m")) for i in incidents]
    healthy_mask = np.ones(len(sc_w), dtype=bool)
    for s, e in spans:
        healthy_mask &= ~((wends >= s) & (wends <= e))
    hv = sc_w[healthy_mask]
    print(f"healthy windows: {int(healthy_mask.sum())}  BW median={float(np.median(hv)):.4f}  p95={float(np.quantile(hv, 0.95)):.4f}  max={float(np.max(hv)):.4f}")
    for i in incidents:
        med, n = median_between(i["start"], i["end"])
        print(f"incident {i['name']}: {n} windows, BW median={med:.4f}")
    # z-score reference on the same windows
    learn = X[l_lo:l_hi]
    mu, sd = learn.mean(axis=0), learn.std(axis=0)
    ok = sd > 1e-9
    zvals = []
    for lo, hi in window_bounds(mw, WIN, STEP):
        w = X[w_lo + lo:w_lo + hi]
        z = np.abs(w.mean(axis=0) - mu) / np.where(ok, sd, 1.0)
        zvals.append(float(z[ok].max()) if ok.any() else 0.0)
    zvals = np.asarray(zvals)
    print(f"z-score healthy median={float(np.median(zvals[healthy_mask])):.4f} vs incident medians below")
    for i in incidents:
        m = np.datetime64(i["start"], "m"); n = np.datetime64(i["end"], "m")
        mask = (wends >= m) & (wends <= n)
        print(f"incident {i['name']}: z median={float(np.median(zvals[mask])):.4f} ({int(mask.sum())} windows)")

if __name__ == "__main__":
    run()

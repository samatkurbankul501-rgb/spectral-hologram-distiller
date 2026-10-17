"""Third real dataset: UCI Appliances Energy Prediction.

19,735 rows of real building telemetry (10-min sampling). Time-ordered 80/20
split; the critical regime is the top 3% of training appliance energy.
Compared: Random / K-Means / KT / SKT / SHD-C (critical-aware auto band).

Usage:
    python bench_appliances.py --seeds 5
Writes bench_appliances_results.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.request
import zipfile

import numpy as np
from sklearn.preprocessing import StandardScaler

from bench_seeds import (
    build_kt,
    build_kmeans,
    build_random,
    build_skt,
    eval_models,
)
from shd import SpectralHologramDistiller

URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"


def load_appliances():
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "appliances.zip")
    if os.path.exists(local):
        with open(local, "rb") as f:
            b = f.read()
    else:
        with urllib.request.urlopen(URL, timeout=180) as r:
            b = r.read()
    z = zipfile.ZipFile(io.BytesIO(b))
    text = z.read(z.namelist()[0]).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    # date, Appliances, lights, T1..T9, RH_1..9, T_out, Press, RH_out, Windspeed,
    # Visibility, Tdewpoint, rv1, rv2
    data = []
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        try:
            y = float(r[1])
            feats = [float(x) for x in r[2:]]
        except ValueError:
            continue
        data.append(feats + [y])
    arr = np.asarray(data, dtype=np.float64)
    return arr[:, :-1], arr[:, -1], "Appliances Energy (real building telemetry)"


def run(seeds, m=300, batch=2500, model_seed=42):
    X, y, title = load_appliances()
    n_tr = int(round(0.80 * len(X)))
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]
    crit = float(np.percentile(y_tr, 97.0))
    crit_mask = y_te >= crit
    X_te_crit, y_te_crit = X_te[crit_mask], y_te[crit_mask]
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te, Z_te_crit = scaler.transform(X_tr), scaler.transform(X_te), scaler.transform(X_te_crit)
    gamma = 1.0 / (X.shape[1] * 1.1)
    print(f"{title}: n={len(X):,} train={len(X_tr):,} test={len(X_te):,} "
          f"crit_thr={crit:.1f} test_crit={len(X_te_crit):,} d={X.shape[1]}", flush=True)

    out = {"dataset": "appliances", "title": title, "n_total": int(len(X)),
           "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
           "n_test_crit": int(len(X_te_crit)), "crit_threshold": crit,
           "n_features": int(X.shape[1]), "n_seeds": len(seeds), "records": [],
           "summary": {}, "elapsed_sec": 0.0}
    t0 = time.perf_counter()
    for seed in seeds:
        rec = {"seed": int(seed)}
        for arm, builder in (
            ("random", lambda: build_random(X_tr, y_tr, seed, m)),
            ("kmeans", lambda: build_kmeans(X_tr, y_tr, Z_tr, seed, m)),
            ("kt", lambda: build_kt(X_tr, y_tr, Z_tr, seed, m)),
            ("skt", lambda: build_skt(X_tr, y_tr, Z_tr, seed, m)),
        ):
            Xc, yc, wc = builder()
            rec[arm] = eval_models(Xc, yc, wc, scaler, model_seed, X_te, Z_te, y_te,
                                   X_te_crit, Z_te_crit, y_te_crit, gamma)
            rec[arm]["rare_captured"] = int(np.sum(yc >= crit))

        d = SpectralHologramDistiller(
            crystal_size=m, sketch_dim=256, random_state=seed,
            critical_threshold=crit, auto_band_ratio=True,
        )
        for i in range(0, len(X_tr), batch):
            d.partial_fit(X_tr[i : i + batch], y_tr[i : i + batch])
        d.finalize_crystal()
        rec["shd_c"] = eval_models(d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64),
                                   d.gamma_weights, scaler, model_seed, X_te, Z_te, y_te,
                                   X_te_crit, Z_te_crit, y_te_crit, gamma)
        rec["shd_c"]["rare_captured"] = int(np.sum(d.crystal_Y >= crit))
        out["records"].append(rec)
        print(f"seed {seed}: shd_c={rec['shd_c']['krr_rmse_crit']:.2f} "
              f"kt={rec['kt']['krr_rmse_crit']:.2f} "
              f"skt={rec['skt']['krr_rmse_crit']:.2f} "
              f"shd_all={rec['shd_c']['krr_rmse_all']:.2f} "
              f"rare={rec['shd_c']['rare_captured']}", flush=True)

    arms = [k for k in out["records"][0] if k != "seed"]
    for arm in arms:
        out["summary"][arm] = {
            "krr_crit_median": float(np.median([r[arm]["krr_rmse_crit"] for r in out["records"]])),
            "krr_all_median": float(np.median([r[arm]["krr_rmse_all"] for r in out["records"]])),
            "gbm_crit_median": float(np.median([r[arm]["gbm_rmse_crit"] for r in out["records"]])),
            "gbm_all_median": float(np.median([r[arm]["gbm_rmse_all"] for r in out["records"]])),
            "rare_median": float(np.median([r[arm]["rare_captured"] for r in out["records"]])),
        }
    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="bench_appliances_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for arm, v in res["summary"].items():
        print(f"  {arm:8s} crit={v['krr_crit_median']:.2f} all={v['krr_all_median']:.2f} "
              f"gbm_all={v['gbm_all_median']:.2f} rare={v['rare_median']:.0f}")


if __name__ == "__main__":
    main()

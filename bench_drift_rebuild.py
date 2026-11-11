"""Late-regime capture on a drifting stream: the finalize-time reservoir rebuild.

With ``lattice_rebuild_interval=0`` (the reproducible default) the lattice is
built once from the warm-up prefix and never rebuilt, so a regime that arrives
LAST gets no dedicated medoids. Enabling ``interval>0`` rebuilds during
streaming, and (since the 2026-11-11 metric-growth fix) ``finalize_crystal``
rebuilds once more from the reservoir so a stream ending mid-interval still
captures its final regime.

Protocol: the synthetic reactor stream with the train split sorted by ascending
``y`` so the hot regime arrives last; compare ``interval=0`` (blind) against
``interval=1500`` (rebuild + finalize fix). Same split, scaler, KRR/GBM head and
critical mask as ``bench_seeds``. Writes ``bench_drift_rebuild_results.json``.

Usage:
    python bench_drift_rebuild.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bench_seeds import eval_models, load_reactor
from shd import SpectralHologramDistiller


def run(seeds):
    X_raw, y_raw, crit, title = load_reactor()
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    order = np.argsort(y_tr)            # drift: hot regime arrives LAST
    X_tr, y_tr = X_tr[order], y_tr[order]
    scaler = StandardScaler().fit(X_tr)
    Z_te = scaler.transform(X_te)
    cm = y_te >= crit
    X_tc, y_tc = X_te[cm], y_te[cm]
    Z_tc = scaler.transform(X_tc)
    gamma = 1.0 / (X_raw.shape[1] * 1.1)

    out = {"dataset": "reactor-drift (train sorted by y, hot regime last)",
           "crit_threshold": float(crit), "seeds": list(seeds), "arms": {}}
    for interval in (0, 1500):
        recs = []
        for seed in seeds:
            d = SpectralHologramDistiller(
                crystal_size=300, high_band_ratio=0.26, sketch_dim=256,
                tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
                lattice_rebuild_interval=interval)
            for i in range(0, len(X_tr), 2500):
                d.partial_fit(X_tr[i:i + 2500], y_tr[i:i + 2500])
            d.finalize_crystal()
            yc = np.asarray(d.crystal_Y, dtype=np.float64)
            m = eval_models(d.crystal_X, yc, d.gamma_weights, scaler, 42,
                            X_te, Z_te, y_te, X_tc, Z_tc, y_tc, gamma)
            m["rare_captured"] = int(np.sum(yc >= crit))
            m["seed"] = int(seed)
            recs.append(m)
        out["arms"][str(interval)] = {
            "records": recs,
            "krr_rmse_crit_median": float(np.median([r["krr_rmse_crit"] for r in recs])),
            "krr_rmse_all_median": float(np.median([r["krr_rmse_all"] for r in recs])),
            "rare_captured_median": float(np.median([r["rare_captured"] for r in recs])),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", default="bench_drift_rebuild_results.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = run(list(range(args.seeds)))
    out["elapsed_sec"] = time.perf_counter() - t0
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Saved -> {args.output} ({out['elapsed_sec']:.0f}s)")
    for k, v in out["arms"].items():
        print(f"  interval={k}: crit={v['krr_rmse_crit_median']:.3f} "
              f"all={v['krr_rmse_all_median']:.3f} rare={v['rare_captured_median']:.1f}")


if __name__ == "__main__":
    main()
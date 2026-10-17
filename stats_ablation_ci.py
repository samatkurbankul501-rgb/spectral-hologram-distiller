"""Bootstrap 95% CIs for the one-at-a-time ablation.

Uses the per-seed records saved by ``bench_ablation.py`` (10 seeds) and reports
the paired difference vs the published default for the critical-regime KRR
RMSE, with a percentile bootstrap over seeds.

Usage:
    python stats_ablation_ci.py
Writes bench_ablation_ci_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
N_BOOT = 10_000


def boot_ci(vals, n_boot=N_BOOT):
    vals = np.asarray(vals, dtype=np.float64)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(RNG.choice(vals, size=len(vals), replace=True))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="bench_ablation_ci.json")
    ap.add_argument("--output", default="bench_ablation_ci_summary.json")
    args = ap.parse_args()
    src = os.path.join(ROOT, args.input) if not os.path.isabs(args.input) else args.input
    d = json.load(open(src, encoding="utf-8"))
    out = {}
    for dataset, block in d.items():
        defaults = block["defaults"]
        out[dataset] = {}
        for const, rows in block["sweeps"].items():
            default_val = defaults[const]
            default_row = next(r for r in rows if abs(r["value"] - default_val) < 1e-12)
            base = np.array([r["krr_rmse_crit"] for r in default_row["records"]], dtype=np.float64)
            out[dataset][const] = {}
            for row in rows:
                vals = np.array([r["krr_rmse_crit"] for r in row["records"]], dtype=np.float64)
                lo, hi = boot_ci(vals)
                diff = vals - base
                dlo, dhi = boot_ci(diff)
                out[dataset][const][str(row["value"])] = {
                    "median": float(np.median(vals)),
                    "ci95": [lo, hi],
                    "median_diff_vs_default": float(np.median(diff)),
                    "diff_ci95": [dlo, dhi],
                    "n_seeds": int(len(vals)),
                }
    path = os.path.join(ROOT, args.output) if not os.path.isabs(args.output) else args.output
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Saved -> {path}\n")
    for dataset, block in out.items():
        print(f"[{dataset}]")
        for const, vals in block.items():
            print(f"  {const}")
            for val, r in vals.items():
                lo, hi = r["ci95"]
                dlo, dhi = r["diff_ci95"]
                print(f"    {val:>5s}: {r['median']:.3f} [{lo:.3f}, {hi:.3f}] "
                      f"diff={r['median_diff_vs_default']:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")


if __name__ == "__main__":
    main()

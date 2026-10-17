"""Holm-Bonferroni adjustment for the existing SHD benchmark JSONs.

The paper reports 40+ paired Wilcoxon tests without a multiple-comparison
correction. This script re-reads the published artifacts and writes a compact
table of raw and Holm-adjusted p-values, per dataset/family.

Usage:
    python stats_holm.py
Writes stats_holm_results.json and prints a compact summary.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def holm(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def collect_seeds(path):
    out = {}
    if not os.path.exists(path):
        return out
    d = json.load(open(path, encoding="utf-8"))
    for dataset, r in d.get("results", {}).items():
        keys = list(r.get("wilcoxon", {}))
        if not keys:
            continue
        raw = [r["wilcoxon"][k]["p_value"] for k in keys]
        adj = holm(raw)
        out[dataset] = {
            k: {
                "p_raw": float(r["wilcoxon"][k]["p_value"]),
                "p_holm": float(a),
                "wins_shd": int(r["wilcoxon"][k].get("wins_shd", -1)),
            }
            for k, a in zip(keys, adj)
        }
    return out


def collect_flat(path, key="wilcoxon"):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8"))
    w = d.get(key, {})
    if not w:
        return {}
    keys = [k for k in w if "p_value" in w[k]]
    if not keys:
        return {}
    raw = [w[k]["p_value"] for k in keys]
    adj = holm(raw)
    return {k: {"p_raw": float(w[k]["p_value"]), "p_holm": float(a),
                "wins_shd": int(w[k].get("wins_shd", -1))} for k, a in zip(keys, adj)}


def main():
    out = {"bench_seeds": collect_seeds(os.path.join(ROOT, "bench_seeds_results.json")),
           "bridge": collect_flat(os.path.join(ROOT, "bridge_neon_bo_results.json")),
           "heal": collect_flat(os.path.join(ROOT, "bench_heal_superconduct_results.json")),
           "pm25": collect_flat(os.path.join(ROOT, "bench_pm25_results.json"))}
    path = os.path.join(ROOT, "stats_holm_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Saved -> {path}\n")
    for family, block in out.items():
        if family == "bench_seeds":
            for dataset, tests in block.items():
                print(f"[{family}/{dataset}] {len(tests)} tests")
                for k, v in tests.items():
                    if v["p_raw"] < 0.05 or v["p_holm"] < 0.05:
                        print(f"  {k:52s} raw={v['p_raw']:.2e} holm={v['p_holm']:.2e}")
        elif block:
            print(f"[{family}] {len(block)} tests")
            for k, v in block.items():
                if v["p_raw"] < 0.05 or v["p_holm"] < 0.05:
                    print(f"  {k:52s} raw={v['p_raw']:.2e} holm={v['p_holm']:.2e}")


if __name__ == "__main__":
    main()

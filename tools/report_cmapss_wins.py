"""Compact FD001 summary: SHD arms vs baselines per metric (medians + wins)."""
from __future__ import annotations

import json

import numpy as np

def _r(x):
    return {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in x.items()}


with open("bench_cmapss_results.json", encoding="utf-8") as fh:
    doc = json.load(fh)

res = doc["results"]["fd001"]
recs = res["records"]
arms = ["random", "kmeans", "kt", "skt"]
shd_arms = [k for k in recs[0] if k.startswith("shd_r")]
metrics = ["krr_rmse_all", "krr_rmse_crit", "krr_phm08",
           "gbm_rmse_all", "gbm_rmse_crit", "gbm_phm08"]

print(f"n_seeds={len(recs)}  partial={res.get('partial')}")
print(f"oracle: " + json.dumps(_r(res['oracle'])))
hdr = "metric".ljust(14) + "".join(a.ljust(10) for a in shd_arms + arms)
print(hdr)
for met in metrics:
    row = met.ljust(14)
    for arm in shd_arms + arms:
        try:
            v = float(np.median([r[arm][met] for r in recs]))
        except KeyError:
            v = float("nan")
        row += f"{v:<10.2f}"
    print(row)

print("\npaired wins (shd arm better) per metric:")
for sa in shd_arms:
    for arm in arms:
        wins = {}
        for met in metrics:
            a = np.array([r[sa][met] for r in recs])
            b = np.array([r[arm][met] for r in recs])
            wins[met] = int(np.sum(a < b))
        print(f"  {sa} vs {arm}: " + " ".join(f"{m}={w}/{len(recs)}" for m, w in wins.items()))

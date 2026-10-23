"""Dump selected bench JSON artifacts compactly (medians + config, no records).

Usage: python tools/report_jsons.py name1 name2 ...
"""
from __future__ import annotations

import json
import sys

MED_KEYS = ("krr_crit_median", "krr_all_median", "gbm_crit_median", "gbm_all_median",
            "rare_median", "shd_median", "random_median", "kmeans_median",
            "kt_median", "skt_median", "gbm_full_median", "krr_full_median")


def _compact(obj, depth=0):
    if isinstance(obj, dict):
        keep = {}
        for k, v in obj.items():
            if k == "records" and isinstance(v, list):
                keep[k] = f"<{len(v)} records>"
            elif isinstance(v, (dict, list)):
                keep[k] = _compact(v, depth + 1)
            elif isinstance(v, float):
                keep[k] = round(v, 3)
            else:
                keep[k] = v
        return keep
    if isinstance(obj, list):
        return obj if len(obj) <= 6 else f"<{len(obj)} items>"
    return obj


def main(argv):
    for name in argv:
        print("===", name)
        try:
            with open(name, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:  # noqa: BLE001
            print("  ERR", e)
            continue
        print(json.dumps(_compact(d), ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main(sys.argv[1:])

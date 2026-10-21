"""Render the Scania operational-cost artifacts as the markdown used by README §8.

Numbers are read from the JSON produced by ``bench_scania_cost.py`` so the README
table cannot drift from the artifact through manual transcription. Run after the
benchmark and paste the output verbatim.

Usage:
    python make_scania_table.py [bench_scania_cost_results.json]
"""

from __future__ import annotations

import json
import sys


def fmt(x, nd=3):
    return "n/a" if x is None else ("{:.{}f}".format(x, nd))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_scania_cost_results.json"
    d = json.load(open(path, encoding="utf-8"))
    models = d["models"]
    s = d["summary"]
    cp = d["crystal_positives"]
    n_test = d["n_test"]
    fn_c = d["cost_model"]["fn"]
    fp_c = d["cost_model"]["fp"]

    print("<!-- generated from {} by make_scania_table.py -->".format(path))
    print("| Coreset ({} rows) | Model | Failures missed | False alarms | "
          "Recall | Precision | Cost = {}·FN + {}·FP |".format(
              d["m"], fn_c, fp_c))
    print("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for model in models:
        o = d["oracle_fulltrain"][model]
        print("| **Full train (oracle)** | {} | {:.0f} | {:.0f} | {} | {} | **{:.0f}** |".format(
            model, o["fn"], o["fp"], fmt(o["recall"]), fmt(o["precision"]), o["cost"]))
    for model in models:
        for arm in ("random", "kmeans", "shd"):
            v = s["{}_{}".format(arm, model)]
            # One decimal: medians over an even number of seeds are fractional
            # (10-seed FN 77.5, FP 555.5, cost 964.5); '{:.0f}' would
            # banker-round them to "78"/"556"/"964" -- silently wrong in both
            # directions. Recall/precision keep 3 decimals via fmt().
            print("| {} | {} | {:.1f} | {:.1f} | {} | {} | **{:.1f}** |".format(
                arm, model, v["fn"], v["fp"], fmt(v["recall"]), fmt(v["precision"]), v["cost"]))
    print()
    # One decimal: a median of an even number of integer counts is fractional,
    # and a 0-decimal format would silently round 7.5 to "8" (banker's rounding).
    print("Positive rows inside the 300-row coreset (median over {} seeds): ".format(
        d["n_seeds"]) + ", ".join(
        "{} {:.1f}".format(a, cp[a]) for a in ("random", "kmeans", "shd")))
    print("Never-alert policy cost: {:.0f}; test rows: {:,}.".format(
        d["never_alert"]["cost"], n_test))
    print()
    print("Paired Wilcoxon (SHD cheaper = win), {} seeds:".format(d["n_seeds"]))
    for key, v in d["wilcoxon"].items():
        print("- `{}`: wins {}/{}, p = {:.3g}".format(
            key, v["wins_shd"], d["n_seeds"], v["p_value"]))


if __name__ == "__main__":
    main()

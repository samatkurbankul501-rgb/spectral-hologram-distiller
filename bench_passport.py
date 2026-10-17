"""Passport recalibration and domain-bank benchmark.

Single-crystal heal has three passport modes on a strong A -> B shift:

* ``blend`` (default): the passport is a mixture; on strong drift both domains
  can remain above the 0.085 alert threshold.
* ``recalibrate``: the new window's covariance replaces the passport; B is
  accepted, A is flagged.
* ``bank``: one crystal/passport per domain; both are accepted and a window is
  routed to the nearest passport.

Run 5 seeds on UCI Superconductivity (A: Tc < 40 K, B: Tc >= 40 K), M=200.

Usage: python bench_passport.py --seeds 5
Writes bench_passport_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from sklearn.model_selection import train_test_split

from bench_seeds import load_superconduct
from shd import DomainBank, HologramCrystalLoader, SpectralHologramDistiller


def build_crystal(X_tr, y_tr, seed, path, m=200):
    d = SpectralHologramDistiller(
        crystal_size=m, sketch_dim=256, tail_quantile=84.0, phase_kappa=3.2,
        random_state=seed,
    )
    for i in range(0, len(X_tr), 2500):
        d.partial_fit(X_tr[i : i + 2500], y_tr[i : i + 2500])
    d.finalize_crystal()
    d.save_hologram(path, include_projector=True)
    return HologramCrystalLoader(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--crystal-size", type=int, default=200)
    args = ap.parse_args()

    X, y, _, title = load_superconduct()
    A, B = y < 40.0, y >= 40.0
    XA, yA, XB, yB = X[A], y[A], X[B], y[B]
    XA_tr, XA_te, yA_tr, yA_te = train_test_split(XA, yA, test_size=0.20, random_state=42)
    XB_tr, XB_te, yB_tr, yB_te = train_test_split(XB, yB, test_size=0.20, random_state=42)

    tmp = tempfile.mkdtemp(prefix="shd_passport_")
    records = []
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        pa = os.path.join(tmp, f"A_{seed}.hologram.npz")
        pb = os.path.join(tmp, f"B_{seed}.hologram.npz")
        lad = build_crystal(XA_tr, yA_tr, seed, pa, args.crystal_size)
        pre = {
            "A": float(lad.check_domain_shift(XA_te)["bw_distance"]),
            "B": float(lad.check_domain_shift(XB_te)["bw_distance"]),
            "threshold_A": float(lad.check_domain_shift(XA_te)["threshold"]),
            "threshold_B": float(lad.check_domain_shift(XB_te)["threshold"]),
        }

        blend = HologramCrystalLoader(pa)
        rb = blend.heal_domain_shift(
            XB_tr, yB_tr, n_inject=45, max_crystal_size=220, random_state=seed
        )
        blend_bw = {
            "A": float(blend.check_domain_shift(XA_te)["bw_distance"]),
            "B": float(blend.check_domain_shift(XB_te)["bw_distance"]),
            "threshold_A": float(blend.check_domain_shift(XA_te)["threshold"]),
            "threshold_B": float(blend.check_domain_shift(XB_te)["threshold"]),
        }

        recal = HologramCrystalLoader(pa)
        rr = recal.heal_domain_shift(
            XB_tr, yB_tr, n_inject=45, max_crystal_size=220, random_state=seed,
            passport_recalibrate=True,
        )
        recal_bw = {
            "A": float(recal.check_domain_shift(XA_te)["bw_distance"]),
            "B": float(recal.check_domain_shift(XB_te)["bw_distance"]),
            "threshold_A": float(recal.check_domain_shift(XA_te)["threshold"]),
            "threshold_B": float(recal.check_domain_shift(XB_te)["threshold"]),
        }

        ldb = build_crystal(XB_tr, yB_tr, seed, pb, args.crystal_size)
        bank = DomainBank([lad, ldb])
        route_A = bank.check_domain_shift(XA_te)
        route_B = bank.check_domain_shift(XB_te)
        bank_res = {
            "domain_A": int(route_A["domain_index"]),
            "domain_B": int(route_B["domain_index"]),
            "shifted_A": bool(route_A["is_shifted"]),
            "shifted_B": bool(route_B["is_shifted"]),
        }

        rec = {
            "seed": seed,
            "pre": pre,
            "blend": blend_bw,
            "recalibrate": recal_bw,
            "bank": bank_res,
            "heal": {k: rb[k] for k in ("healed", "n_injected", "n_pruned", "bw_after")},
            "recal_heal": {k: rr[k] for k in ("healed", "n_injected", "n_pruned", "bw_after")},
        }
        records.append(rec)
        print(
            f"seed {seed}: blend B={blend_bw['B']:.3f} (thr {blend_bw['threshold_B']:.3f}) "
            f"recal B={recal_bw['B']:.3f} A={recal_bw['A']:.3f} | "
            f"bank route A/B={bank_res['domain_A']}/{bank_res['domain_B']} "
            f"shifted={bank_res['shifted_A']}/{bank_res['shifted_B']}",
            flush=True,
        )

    def med(values):
        return float(np.median(values))

    summary = {
        "pre_B": med([r["pre"]["B"] for r in records]),
        "pre_A": med([r["pre"]["A"] for r in records]),
        "blend_B": med([r["blend"]["B"] for r in records]),
        "blend_A": med([r["blend"]["A"] for r in records]),
        "recal_B": med([r["recalibrate"]["B"] for r in records]),
        "recal_A": med([r["recalibrate"]["A"] for r in records]),
        "blend_B_above": int(sum(r["blend"]["B"] > r["blend"]["threshold_B"] for r in records)),
        "blend_A_above": int(sum(r["blend"]["A"] > r["blend"]["threshold_A"] for r in records)),
        "recal_B_below": int(sum(r["recalibrate"]["B"] < r["recalibrate"]["threshold_B"] for r in records)),
        "recal_A_above": int(sum(r["recalibrate"]["A"] > r["recalibrate"]["threshold_A"] for r in records)),
        "bank_route_correct": int(sum(
            r["bank"]["domain_A"] == 0 and r["bank"]["domain_B"] == 1 for r in records
        )),
        "bank_accepted_both": int(sum(
            not r["bank"]["shifted_A"] and not r["bank"]["shifted_B"] for r in records
        )),
    }
    out = {
        "dataset": title,
        "n_seeds": len(records),
        "crystal_size": args.crystal_size,
        "records": records,
        "summary": summary,
        "elapsed_sec": time.perf_counter() - t0,
    }
    with open("bench_passport_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("summary:", json.dumps(summary, indent=1))
    print("saved -> bench_passport_results.json")


if __name__ == "__main__":
    main()

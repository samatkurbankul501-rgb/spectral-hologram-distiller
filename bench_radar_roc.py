"""Graded ROC/AUC characterization of the Bures-Wasserstein radar.

The paper reports one easy operating point (Domain A <40 K vs Domain B >=40 K,
20/20 detections, 0/20 false alarms). This script trains the crystal on a
narrow low-Tc regime and sweeps progressively harder OOD bands, so the radar is
characterized by a real ROC curve instead of a single perfect split.

Usage:
    python bench_radar_roc.py --seeds 10 --train-cut 20
    python bench_radar_roc.py --seeds 10 --train-cut 40
Writes bench_radar_roc_results.json (one entry per train_cut).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller


def windows(X, B):
    n = (len(X) // B) * B
    if n == 0:
        return []
    return [X[i : i + B] for i in range(0, n, B)]


def score_windows(loader, X, B):
    return np.array([loader.check_domain_shift(w)["bw_distance"] for w in windows(X, B)])


def band_defs(y, train_cut):
    if train_cut == 40:
        return {"ood_ge40": y >= 40}
    if train_cut == 20:
        return {
            "ood_20_30": (y >= 20) & (y < 30),
            "ood_30_40": (y >= 30) & (y < 40),
            "ood_40_77": (y >= 40) & (y < 77),
            "ood_77_plus": y >= 77,
        }
    raise ValueError("train_cut must be 20 or 40")


def run(seeds, crystal_size=200, train_cut=20):
    X, y, _, title = load_superconduct()
    if X.shape[1] != 81:
        raise RuntimeError("UCI fetch failed (fallback dataset); aborting")
    in_mask = y < train_cut
    X_in, y_in = X[in_mask], y[in_mask]
    X_tr_in, X_te_in, _, _ = train_test_split(X_in, y_in, test_size=0.20, random_state=42)
    bands = {name: X[mask] for name, mask in band_defs(y, train_cut).items()}
    bands = {name: Xb for name, Xb in bands.items() if len(Xb) > 0}

    out = {
        "dataset": title, "n_seeds": len(seeds), "crystal_size": crystal_size,
        "train_cut": train_cut, "n_in_test": int(len(X_te_in)),
        "band_sizes": {k: int(len(v)) for k, v in bands.items()},
        "by_batch": {},
    }
    tmp = tempfile.mkdtemp(prefix="shd_roc_")
    t0 = time.perf_counter()

    for B in (64, 256, 1024):
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            Xs = X_tr_in[rng.permutation(len(X_tr_in))]
            d = SpectralHologramDistiller(
                crystal_size=crystal_size, high_band_ratio=0.26, sketch_dim=256,
                tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
            )
            for i in range(0, len(Xs), 2500):
                d.partial_fit(Xs[i : i + 2500], None)  # radar passport is label-free
            d.finalize_crystal()
            path = os.path.join(tmp, f"roc_{seed}.hologram.npz")
            d.save_hologram(path)
            loader = HologramCrystalLoader(path)

            s_in = score_windows(loader, X_te_in, B)
            tau = 0.085 * max(1.0, np.sqrt(1024.0 / max(B, 16)))
            row = {"seed": int(seed), "bands": {}}
            all_out = []
            for name, Xb in bands.items():
                s_out = score_windows(loader, Xb, B)
                if len(s_out) == 0:
                    continue
                all_out.append(s_out)
                labels = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
                scores = np.concatenate([s_in, s_out])
                auc = float(roc_auc_score(labels, scores))
                fpr, tpr, _ = roc_curve(labels, scores)
                idx05 = max(0, min(int(np.searchsorted(fpr, 0.05, side="right") - 1),
                                   len(tpr) - 1))
                row["bands"][name] = {
                    "auc": auc,
                    "tpr_at_fpr05": float(tpr[idx05]),
                    "fpr_at_fpr05": float(fpr[idx05]),
                    "fpr_published": float(np.mean(s_in > tau)),
                    "tpr_published": float(np.mean(s_out > tau)),
                    "bw_in_median": float(np.median(s_in)),
                    "bw_out_median": float(np.median(s_out)),
                    "n_windows_out": int(len(s_out)),
                }
            if all_out:
                s_all = np.concatenate(all_out)
                labels = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_all))])
                scores = np.concatenate([s_in, s_all])
                auc = float(roc_auc_score(labels, scores))
                fpr, tpr, _ = roc_curve(labels, scores)
                idx05 = max(0, min(int(np.searchsorted(fpr, 0.05, side="right") - 1),
                                   len(tpr) - 1))
                row["all_ood"] = {
                    "auc": auc, "tpr_at_fpr05": float(tpr[idx05]),
                    "fpr_at_fpr05": float(fpr[idx05]),
                    "fpr_published": float(np.mean(s_in > tau)),
                    "tpr_published": float(np.mean(s_all > tau)),
                    "n_windows_out": int(len(s_all)),
                }
            per_seed.append(row)
            print(f"[cut={train_cut}, B={B:4d}] seed {seed:2d}: "
                  f"all_ood_auc={row.get('all_ood', {}).get('auc', float('nan')):.4f}", flush=True)

        bands_out = {}
        for name in bands:
            vals = [r["bands"][name] for r in per_seed if name in r["bands"]]
            if not vals:
                continue
            bands_out[name] = {
                "auc_median": float(np.median([v["auc"] for v in vals])),
                "auc_iqr": float(np.percentile([v["auc"] for v in vals], 75)
                                 - np.percentile([v["auc"] for v in vals], 25)),
                "tpr_at_fpr05_median": float(np.median([v["tpr_at_fpr05"] for v in vals])),
                "tpr_published_median": float(np.median([v["tpr_published"] for v in vals])),
                "fpr_published_median": float(np.median([v["fpr_published"] for v in vals])),
            }
        all_vals = [r["all_ood"] for r in per_seed if "all_ood" in r]
        out["by_batch"][str(B)] = {
            "per_seed": per_seed,
            "bands": bands_out,
            "all_ood": {
                "auc_median": float(np.median([v["auc"] for v in all_vals])),
                "auc_iqr": float(np.percentile([v["auc"] for v in all_vals], 75)
                                 - np.percentile([v["auc"] for v in all_vals], 25)),
                "tpr_at_fpr05_median": float(np.median([v["tpr_at_fpr05"] for v in all_vals])),
                "tpr_published_median": float(np.median([v["tpr_published"] for v in all_vals])),
                "fpr_published_median": float(np.median([v["fpr_published"] for v in all_vals])),
            },
        }
        print(f"[cut={train_cut}, B={B:4d}] all-OOD median AUC="
              f"{out['by_batch'][str(B)]['all_ood']['auc_median']:.4f}", flush=True)

    out["elapsed_sec"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-cut", type=int, choices=(20, 40), default=20)
    ap.add_argument("--output", default="bench_radar_roc_results.json")
    args = ap.parse_args()
    res = run(list(range(args.seeds)), train_cut=args.train_cut)
    merged = {}
    try:
        with open(args.output, encoding="utf-8") as f:
            merged = json.load(f)
    except FileNotFoundError:
        pass
    if isinstance(merged, dict) and "by_cut" in merged:
        merged["by_cut"][str(args.train_cut)] = res
    else:
        merged = {"by_cut": {str(args.train_cut): res}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {args.output} ({res['elapsed_sec']:.0f}s)")
    for B, v in res["by_batch"].items():
        print(f"  B={B}: all-OOD AUC={v['all_ood']['auc_median']:.4f}, "
              f"TPR@5%={v['all_ood']['tpr_at_fpr05_median']:.3f}, "
              f"TPR@tau={v['all_ood']['tpr_published_median']:.3f}, "
              f"FPR@tau={v['all_ood']['fpr_published_median']:.3f}")
        for name, bv in v["bands"].items():
            print(f"    {name:12s} AUC={bv['auc_median']:.4f} "
                  f"TPR@5%={bv['tpr_at_fpr05_median']:.3f}")


if __name__ == "__main__":
    main()

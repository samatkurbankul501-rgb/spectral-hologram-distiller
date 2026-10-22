"""§6 frontier benchmark: high_band_ratio sweep at M=600 vs equal-budget KMeans
and the §1 oracle (2026-10-22, v3.12.0).

README §6 ("Choosing the operating point") was the only section backed by
one-off working scripts instead of a committed per-seed artifact: the
M=600 `high_band_ratio` tables (SHD vs KMeans, reactor + superconductivity)
and the oracle-beating row (br=0.65 crystal vs the full-data reference).
This benchmark regenerates all of §6 under the canonical `bench_seeds.py`
protocol and saves every per-seed record to `bench_frontier_results.json`,
so the section can finally cite an artifact like every other one.

Design (mirrors bench_seeds.py exactly):
- Train/test split FIXED (random_state=42, shuffled); the same
  `eval_models` (KRR alpha=0.08, gamma=1/(d*1.1); GBM 120/20/5, seed 42),
  the same `batch=2500` streaming and the same seeds 0..n-1.
- The oracle is NOT recomputed here: §1's oracle denominators are the
  committed `bench_seeds_results.json` values (KRR on the first 9,000
  shuffled train rows, GBM on the full train split). This benchmark joins
  them by dataset key and stores them with an explicit `oracle_source`, so
  the §6 comparison cites the same reference the README §1 table cites.
- Arms per seed: SHD at high_band_ratio in {0.26, 0.50, 0.65} (M=600) and
  MiniBatch K-Means at M=600 (the equal-budget baseline of §6). Paired
  Wilcoxon SHD-vs-KMeans per metric; per-seed wins vs the oracle are
  recorded for the oracle-beating row.
- Progress is checkpointed after every (arm, seed); an interrupted run
  resumes and `partial: true` marks incomplete datasets.

Usage:
    python bench_frontier.py [--seeds 20] [--dataset superconduct|reactor|both]
                             [--ratios 0.26,0.50,0.65] [--m 600]
                             [--output bench_frontier_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import wilcoxon

from bench_seeds import build_kmeans, eval_models, load_reactor, load_superconduct
from shd import SpectralHologramDistiller

ORACLE_SOURCE = "bench_seeds_results.json"
METRICS = ["krr_rmse_all", "krr_rmse_crit", "gbm_rmse_all", "gbm_rmse_crit"]


def load_dataset(key):
    if key == "superconduct":
        X_raw, y_raw, crit, title = load_superconduct()
        if X_raw.shape[1] != 81 or len(X_raw) <= 20000:
            raise RuntimeError(
                f"superconductivity fetch returned {X_raw.shape}; refusing to "
                "run the frontier benchmark on the California-housing fallback"
            )
        return X_raw, y_raw, crit, title
    return load_reactor()


def build_shd(X_tr, y_tr, seed, m, ratio, batch, crit_threshold):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=ratio, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i:i + batch], y_tr[i:i + batch])
    d.finalize_crystal()
    n_rare = int(np.sum(np.asarray(d.crystal_Y, dtype=np.float64) >= crit_threshold))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare


def paired_tests(records_a, records_b, metrics=METRICS):
    tests = {}
    for metric in metrics:
        a = np.array([r[metric] for r in records_a], dtype=np.float64)
        b = np.array([r[metric] for r in records_b], dtype=np.float64)
        diff = a - b
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b)
        tests[metric] = {
            "median_shd": float(np.median(a)),
            "iqr_shd": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "median_kmeans": float(np.median(b)),
            "iqr_kmeans": float(np.percentile(b, 75) - np.percentile(b, 25)),
            "wins_shd": int(np.sum(diff < 0)),
            "delta_median_pct": float(100.0 * (np.median(a) / np.median(b) - 1.0)),
            "statistic": float(stat),
            "p_value": float(p),
        }
    return tests


def vs_oracle(records, oracle, metrics=("krr_rmse_crit", "gbm_rmse_crit")):
    out = {}
    for metric in metrics:
        a = np.array([r[metric] for r in records], dtype=np.float64)
        out[metric] = {
            "median_shd": float(np.median(a)),
            "oracle": float(oracle[metric]),
            "wins_vs_oracle": int(np.sum(a < oracle[metric])),
            "delta_median_pct": float(100.0 * (np.median(a) / oracle[metric] - 1.0)),
        }
    return out


def save(path, out):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run_dataset(key, args, out):
    t0 = time.perf_counter()
    with open(ORACLE_SOURCE, encoding="utf-8") as f:
        oracle_ref = json.load(f)["results"][key]["oracle"]
    oracle = {
        "krr_rmse_all": float(oracle_ref["krr_rmse_all"]),
        "krr_rmse_crit": float(oracle_ref["krr_rmse_crit"]),
        "gbm_rmse_all": float(oracle_ref["gbm_rmse_all"]),
        "gbm_rmse_crit": float(oracle_ref["gbm_rmse_crit"]),
    }

    X_raw, y_raw, crit, title = load_dataset(key)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=42)
    is_crit = y_te >= crit
    X_te_crit, y_te_crit = X_te[is_crit], y_te[is_crit]
    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te = scaler.transform(X_te)
    Z_te_crit = scaler.transform(X_te_crit)
    rbf_gamma = 1.0 / (X_raw.shape[1] * 1.1)
    print(f"[{key}] {title}: n_train={len(X_tr):,}, test_crit={len(X_te_crit):,}, "
          f"oracle joined from {ORACLE_SOURCE} (krr_crit={oracle['krr_rmse_crit']:.2f})", flush=True)

    res = out["results"].setdefault(key, {
        "title": title, "m": args.m, "ratios": args.ratios,
        "n_seeds": args.seeds, "seeds": list(range(args.seeds)),
        "split_random_state": 42, "batch": args.batch, "model_seed": args.model_seed,
        "oracle": oracle, "oracle_source": ORACLE_SOURCE,
        "arms": {}, "done": [],
    })
    done = set(res.get("done", []))
    res.setdefault("arms", {})

    def evaluate(Xc, yc, wc):
        return eval_models(Xc, yc, wc, scaler, args.model_seed,
                           X_te, Z_te, y_te, X_te_crit, Z_te_crit, y_te_crit, rbf_gamma)

    arm_keys = [("shd_r{}".format(r), ("shd", r)) for r in args.ratios] + [("kmeans", ("kmeans", None))]
    for arm_key, (kind, ratio) in arm_keys:
        records = res["arms"].setdefault(arm_key, [])
        for seed in range(args.seeds):
            k = f"{key}_{arm_key}_s{seed}"
            if k in done:
                continue
            t_s = time.perf_counter()
            if kind == "shd":
                Xc, yc, wc, n_rare = build_shd(X_tr, y_tr, seed, args.m, ratio,
                                               args.batch, crit)
            else:
                Xc, yc, wc = build_kmeans(X_tr, y_tr, Z_tr, seed, args.m)
                n_rare = int(np.sum(yc >= crit))
            rec = {"seed": seed, "rare_captured": n_rare}
            rec.update(evaluate(Xc, yc, wc))
            records.append(rec)
            records.sort(key=lambda r: r["seed"])
            done.add(k)
            res["done"] = sorted(done)
            res["partial"] = True
            res["elapsed_sec"] = time.perf_counter() - t0
            save(args.output, out)
            print(f"[{key}] {arm_key} seed={seed:2d}: krr_crit={rec['krr_rmse_crit']:.2f} "
                  f"gbm_all={rec['gbm_rmse_all']:.2f} gbm_crit={rec['gbm_rmse_crit']:.2f} "
                  f"rare={n_rare} ({time.perf_counter() - t_s:.1f}s)", flush=True)

    # Finalize: stats + partial flag.
    km = res["arms"]["kmeans"]
    res["wilcoxon"] = {f"shd_r{r}": paired_tests(res["arms"][f"shd_r{r}"], km)
                      for r in args.ratios}
    res["vs_oracle"] = {f"shd_r{r}": vs_oracle(res["arms"][f"shd_r{r}"], oracle)
                        for r in args.ratios}
    res["partial"] = any(len(res["arms"][a]) != args.seeds for a, _ in arm_keys)
    res["elapsed_sec"] = time.perf_counter() - t0
    res.pop("done", None)
    save(args.output, out)
    print(f"[{key}] done in {res['elapsed_sec']:.1f}s (partial={res['partial']})", flush=True)

    for r in args.ratios:
        t = res["wilcoxon"][f"shd_r{r}"]
        print(f"  br={r}: krr_crit {t['krr_rmse_crit']['median_shd']:.2f} vs km "
              f"{t['krr_rmse_crit']['median_kmeans']:.2f} ({t['krr_rmse_crit']['delta_median_pct']:+.1f}%, "
              f"wins {t['krr_rmse_crit']['wins_shd']}/{args.seeds}, "
              f"p={t['krr_rmse_crit']['p_value']:.1e}) | gbm_all "
              f"{t['gbm_rmse_all']['delta_median_pct']:+.1f}% (p={t['gbm_rmse_all']['p_value']:.2f})", flush=True)
    vo_key = f"shd_r{args.ratios[-1]}"
    vo = res["vs_oracle"][vo_key]
    print(f"  {vo_key} vs oracle: krr_crit {vo['krr_rmse_crit']['median_shd']:.2f} vs "
          f"{vo['krr_rmse_crit']['oracle']:.2f} (wins {vo['krr_rmse_crit']['wins_vs_oracle']}/{args.seeds}, "
          f"{vo['krr_rmse_crit']['delta_median_pct']:+.1f}%) | gbm_crit "
          f"{vo['gbm_rmse_crit']['median_shd']:.2f} vs {vo['gbm_rmse_crit']['oracle']:.2f} "
          f"(wins {vo['gbm_rmse_crit']['wins_vs_oracle']}/{args.seeds}, "
          f"{vo['gbm_rmse_crit']['delta_median_pct']:+.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--dataset", choices=["superconduct", "reactor", "both"], default="both")
    ap.add_argument("--ratios", default="0.26,0.50,0.65")
    ap.add_argument("--m", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2500)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--output", default="bench_frontier_results.json")
    args = ap.parse_args()
    args.ratios = [float(r) for r in args.ratios.split(",")]

    out = {"config": {"m": args.m, "ratios": args.ratios, "n_seeds": args.seeds,
                      "batch": args.batch, "model_seed": args.model_seed},
           "results": {}}
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev.get("results"), dict):
                out["results"] = prev["results"]
        except (json.JSONDecodeError, OSError):
            pass

    keys = ["superconduct", "reactor"] if args.dataset == "both" else [args.dataset]
    for key in keys:
        run_dataset(key, args, out)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()

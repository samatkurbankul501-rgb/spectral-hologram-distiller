"""NASA C-MAPSS FD001: remaining-useful-life benchmark on the LOW target tail
(2026-10-22, v3.12.0).

The paper's future-work list named the NASA CMAPSS turbofan degradation
benchmark; this benchmark adds it as the third real dataset, under the
canonical `bench_seeds.py` protocol (20 seeds, paired Wilcoxon, same KRR /
GBM models, M=300) but with the official FD001 test evaluation and a
critical regime on the OPPOSITE side of the target from SC / reactor /
Scania: near-failure cycles (low RUL), so SHD runs with `tail="low"` and
the §6 operating-point sweep `high_band_ratio` in {0.26, 0.50, 0.65}.

Protocol:
- Data: `data/cmapss/{train,test,RUL}_FD001.txt` -- the pristine NASA C-MAPSS
  FD001 split (20,631 train cycles / 13,096 test cycles / 100 engines each;
  test RUL range 7-145), mirrored from public GitHub LFS copies of the
  official `CMAPSSData.zip` (row/column counts verified against the
  published dataset before use).
- Features: the standard 14-sensor FD001 recipe (s2, s3, s4, s7, s8, s9,
  s11, s12, s13, s14, s15, s17, s20, s21). Sensors s1, s5, s10, s16, s18,
  s19 are exactly constant (std = 0) in the single-operating-condition
  FD001 split and are dropped, as is the de-facto convention in the RUL
  literature. No other feature engineering: this benchmark compares CORESET
  methods under identical raw features and models, not RUL pipelines.
- Target: piecewise-linear RUL capped at 125 cycles (the standard
  transform). The critical regime is the low tail `y <= 25` -- the last 25
  cycles before failure (12.1% of train rows) -- so `tail="low"`.
- Test protocol: the official FD001 evaluation -- predict RUL at each test
  engine's last cycle (100 predictions). Metrics: RMSE over all 100
  engines, RMSE over the 19 engines whose true RUL <= 25, and the
  asymmetric PHM08 score (sum of exp(-d/13)-1 for late, exp(d/10)-1 for
  early predictions; lower is better).
- Arms per seed (0..19): SHD (M=300, tail="low", high_band_ratio in
  {0.26, 0.50, 0.65}), Random, K-Means, KT, SKT at the same M=300 over an
  iid shuffled pool (fixed permutation, rng seed 42) -- as in bench_seeds,
  only the coreset construction randomness varies per seed.
- Oracle (seed-independent): KRR on a fixed 9,000-row uniform subsample
  (rng seed 42) + GBM on the full 20,631-row pool, the bench_seeds
  convention.
- Paired Wilcoxon of each SHD arm vs each baseline per metric; per-seed
  wins vs the oracle; progress is checkpointed after every seed.

Usage:
    python bench_cmapss.py [--seeds 20] [--ratios 0.26,0.50,0.65]
                           [--output bench_cmapss_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from bench_seeds import build_kmeans, build_kt, build_random, build_skt
from shd import SpectralHologramDistiller

# 1-based FD001 sensor ids kept by the standard recipe (s1, s5, s10, s16,
# s18, s19 are exactly constant in FD001 and dropped).
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
RUL_CAP = 125.0
CRIT_LE = 25.0
METRICS = ["krr_rmse_all", "krr_rmse_crit", "krr_phm08",
           "gbm_rmse_all", "gbm_rmse_crit", "gbm_phm08"]


def load_fd001():
    root = Path(__file__).resolve().parent / "data" / "cmapss"
    tr = np.loadtxt(root / "train_FD001.txt")
    te = np.loadtxt(root / "test_FD001.txt")
    rul = np.loadtxt(root / "RUL_FD001.txt")
    if tr.shape != (20631, 26) or te.shape != (13096, 26) or rul.shape != (100,):
        raise RuntimeError(f"unexpected FD001 shapes {tr.shape}/{te.shape}/{rul.shape}")

    cols = [4 + s for s in SENSORS]  # col 0 = unit, 1 = cycle, 2..4 = op settings
    unit, cyc = tr[:, 0], tr[:, 1]
    life = np.array([np.max(cyc[unit == u]) for u in range(1, 101)])
    y_tr = np.minimum(life[unit.astype(int) - 1] - cyc + 1.0, RUL_CAP)
    X_tr = tr[:, cols]

    last_idx = np.array([np.max(np.nonzero(te[:, 0] == u)[0]) for u in range(1, 101)])
    X_te = te[last_idx][:, cols]
    y_te = rul
    return X_tr, y_tr, X_te, y_te


def phm08(pred, true):
    """Asymmetric PHM08 score: late (d<0) exp(-d/13)-1, early (d>=0) exp(d/10)-1."""
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def build_shd_low(X_tr, y_tr, seed, m, ratio, batch):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=ratio, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, tail="low", random_state=seed,
    )
    for i in range(0, len(X_tr), batch):
        d.partial_fit(X_tr[i:i + batch], y_tr[i:i + batch])
    d.finalize_crystal()
    n_rare = int(np.sum(np.asarray(d.crystal_Y, dtype=np.float64) <= CRIT_LE))
    return d.crystal_X, np.asarray(d.crystal_Y, dtype=np.float64), d.gamma_weights, n_rare


def evaluate(X_c, y_c, w_c, scaler, model_seed, X_te, Z_te, y_te, crit, rbf_gamma):
    Z_c = scaler.transform(X_c)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=rbf_gamma)
    krr.fit(Z_c, y_c, sample_weight=w_c)
    kp = krr.predict(Z_te)
    gbm = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=20, min_samples_leaf=5, random_state=model_seed
    )
    gbm.fit(X_c, y_c, sample_weight=w_c)
    gp = gbm.predict(X_te)
    return {
        "krr_rmse_all": float(np.sqrt(mean_squared_error(y_te, kp))),
        "krr_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit], kp[crit]))),
        "krr_phm08": phm08(kp, y_te),
        "gbm_rmse_all": float(np.sqrt(mean_squared_error(y_te, gp))),
        "gbm_rmse_crit": float(np.sqrt(mean_squared_error(y_te[crit], gp[crit]))),
        "gbm_phm08": phm08(gp, y_te),
    }


def save(path, out):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--ratios", default="0.26,0.50,0.65")
    ap.add_argument("--batch", type=int, default=2500)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--output", default="bench_cmapss_results.json")
    args = ap.parse_args()
    ratios = [float(r) for r in args.ratios.split(",")]
    seeds = list(range(args.seeds))
    shd_arms = [f"shd_r{r}" for r in ratios]

    t0 = time.perf_counter()
    X_pool, y_pool, X_te, y_te = load_fd001()
    # Fixed iid permutation (rng 42): everyone sees the same shuffled pool,
    # mirroring the shuffled train_test_split of bench_seeds.
    perm = np.random.default_rng(42).permutation(len(X_pool))
    X_tr, y_tr = X_pool[perm], y_pool[perm]
    scaler = StandardScaler().fit(X_tr)
    Z_tr = scaler.transform(X_tr)
    Z_te = scaler.transform(X_te)
    rbf_gamma = 1.0 / (X_tr.shape[1] * 1.1)
    crit = y_te <= CRIT_LE
    print(f"[fd001] C-MAPSS FD001: n_train={len(X_tr):,} cycles, M={args.m}, "
          f"crit<= {CRIT_LE:.0f} cycles: train {100.0*np.mean(y_tr <= CRIT_LE):.1f}%, "
          f"test engines {int(np.sum(crit))}/100", flush=True)

    # Oracle (seed-independent): KRR on a fixed 9,000-row uniform subsample +
    # GBM on the full pool, the bench_seeds convention.
    rng_o = np.random.default_rng(42)
    sub = rng_o.choice(len(X_tr), size=min(9000, len(X_tr)), replace=False)
    oracle = evaluate(X_tr[sub], y_tr[sub], None, scaler, args.model_seed,
                      X_te, Z_te, y_te, crit, rbf_gamma)
    oracle_full_gbm = evaluate(X_tr, y_tr, None, scaler, args.model_seed,
                               X_te, Z_te, y_te, crit, rbf_gamma)
    oracle["gbm_full_pool"] = {k: oracle_full_gbm[k] for k in METRICS}
    print(f"[fd001] oracle: krr(9000) rmse={oracle['krr_rmse_all']:.2f} "
          f"crit={oracle['krr_rmse_crit']:.2f} | gbm(full) "
          f"rmse={oracle_full_gbm['gbm_rmse_all']:.2f} crit={oracle_full_gbm['gbm_rmse_crit']:.2f} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    out = {"config": {"m": args.m, "ratios": ratios, "n_seeds": args.seeds,
                      "seeds": seeds, "batch": args.batch,
                      "model_seed": args.model_seed, "rul_cap": RUL_CAP,
                      "crit_le": CRIT_LE, "sensors": SENSORS,
                      "pool_perm_seed": 42},
           "results": {"fd001": {}}}
    if os.path.exists(args.output):
        try:
            with open(args.output, encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev.get("results"), dict) and "fd001" in prev["results"]:
                out["results"]["fd001"] = prev["results"]["fd001"]
        except (json.JSONDecodeError, OSError):
            pass

    res = out["results"]["fd001"]
    res.setdefault("records", [])
    done = {r["seed"] for r in res["records"]}
    res.update({"oracle": oracle, "n_test_engines": int(len(y_te)),
                "n_test_crit": int(np.sum(crit)), "partial": True})

    for seed in seeds:
        if seed in done:
            continue
        t_s = time.perf_counter()
        rec = {"seed": seed}
        for ratio in ratios:
            Xc, yc, wc, n_rare = build_shd_low(X_tr, y_tr, seed, args.m, ratio,
                                               args.batch)
            rec[f"shd_r{ratio}"] = evaluate(Xc, yc, wc, scaler, args.model_seed,
                                            X_te, Z_te, y_te, crit, rbf_gamma)
            rec[f"shd_r{ratio}"]["rare_captured"] = n_rare

        Xc, yc, wc = build_random(X_tr, y_tr, seed, args.m)
        rec["random"] = evaluate(Xc, yc, wc, scaler, args.model_seed,
                                 X_te, Z_te, y_te, crit, rbf_gamma)
        rec["random"]["rare_captured"] = int(np.sum(yc <= CRIT_LE))

        Xc, yc, wc = build_kmeans(X_tr, y_tr, Z_tr, seed, args.m)
        rec["kmeans"] = evaluate(Xc, yc, wc, scaler, args.model_seed,
                                 X_te, Z_te, y_te, crit, rbf_gamma)
        rec["kmeans"]["rare_captured"] = int(np.sum(yc <= CRIT_LE))

        for name, builder in (("kt", build_kt), ("skt", build_skt)):
            Xc, yc, wc = builder(X_tr, y_tr, Z_tr, seed, args.m)
            rec[name] = evaluate(Xc, yc, wc, scaler, args.model_seed,
                                 X_te, Z_te, y_te, crit, rbf_gamma)
            rec[name]["rare_captured"] = int(np.sum(yc <= CRIT_LE))

        res["records"].append(rec)
        res["records"].sort(key=lambda r: r["seed"])
        res["elapsed_sec"] = time.perf_counter() - t0
        save(args.output, out)
        print(f"[fd001] seed {seed:2d}: r26 krr_crit={rec['shd_r0.26']['krr_rmse_crit']:.2f} "
              f"r50={rec['shd_r0.5']['krr_rmse_crit']:.2f} r65={rec['shd_r0.65']['krr_rmse_crit']:.2f} "
              f"| kt={rec['kt']['krr_rmse_crit']:.2f} skt={rec['skt']['krr_rmse_crit']:.2f} "
              f"km={rec['kmeans']['krr_rmse_crit']:.2f} | rare 26/50/65="
              f"{rec['shd_r0.26']['rare_captured']}/{rec['shd_r0.5']['rare_captured']}/"
              f"{rec['shd_r0.65']['rare_captured']} ({time.perf_counter() - t_s:.1f}s)", flush=True)

    # Paired Wilcoxon of each SHD arm vs each baseline, canonical structure,
    # plus per-arm wins vs the oracle.
    records = res["records"]
    tests = {}
    for arm in shd_arms:
        for metric in METRICS:
            for pair in ["random", "kmeans", "kt", "skt"]:
                a = np.array([r[arm][metric] for r in records], dtype=np.float64)
                b = np.array([r[pair][metric] for r in records], dtype=np.float64)
                diff = a - b
                stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
                tests[f"{arm}_vs_{pair}_{metric}"] = {
                    "median_shd": float(np.median(a)),
                    "iqr_shd": float(np.percentile(a, 75) - np.percentile(a, 25)),
                    "median_" + pair: float(np.median(b)),
                    "iqr_" + pair: float(np.percentile(b, 75) - np.percentile(b, 25)),
                    "wins_shd": int(np.sum(diff < 0)),
                    "statistic": float(stat),
                    "p_value": float(p),
                }
    res["wilcoxon"] = tests
    res["vs_oracle"] = {}
    for arm in shd_arms:
        vo = {}
        for metric in ["krr_rmse_crit", "krr_rmse_all", "gbm_rmse_crit", "gbm_rmse_all"]:
            a = np.array([r[arm][metric] for r in records], dtype=np.float64)
            ref = oracle["gbm_full_pool"][metric] if metric.startswith("gbm") else oracle[metric]
            vo[metric] = {"median_shd": float(np.median(a)), "oracle": float(ref),
                          "wins_vs_oracle": int(np.sum(a < ref))}
        res["vs_oracle"][arm] = vo
    res["partial"] = len(records) != args.seeds
    res["elapsed_sec"] = time.perf_counter() - t0
    save(args.output, out)
    print(f"[fd001] done in {res['elapsed_sec']:.1f}s (partial={res['partial']})", flush=True)

    for arm in shd_arms:
        t = tests[f"{arm}_vs_kt_krr_rmse_crit"]
        tk = tests[f"{arm}_vs_kmeans_krr_rmse_crit"]
        ts = tests[f"{arm}_vs_skt_krr_rmse_crit"]
        vo = res["vs_oracle"][arm]["krr_rmse_crit"]
        print(f"  {arm}: krr_crit={t['median_shd']:.2f} | vs kt {t['wins_shd']:2d}/{len(records)} "
              f"p={t['p_value']:.1e} | vs kmeans {tk['wins_shd']:2d}/{len(records)} "
              f"p={tk['p_value']:.1e} | vs skt {ts['wins_shd']:2d}/{len(records)} "
              f"| vs oracle {vo['wins_vs_oracle']:2d}/{len(records)}", flush=True)
    print(f"  oracle: krr(9000) rmse={oracle['krr_rmse_all']:.2f} crit={oracle['krr_rmse_crit']:.2f} | "
          f"gbm(full) rmse={oracle_full_gbm['gbm_rmse_all']:.2f} "
          f"crit={oracle_full_gbm['gbm_rmse_crit']:.2f}", flush=True)
    print(f"\nSaved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()

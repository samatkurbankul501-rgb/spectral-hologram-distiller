"""Bridge benchmark: SHD v3.7 as a streaming coreset for JARVIS Lab BO surrogates.

Why this bridge matters:
- Gaussian Process (GP) surrogates in Bayesian Optimization scale as O(N^3) and
  cannot digest raw streaming telemetry / high-throughput screening logs
  (N ~ 10^4 - 10^5).
- Standard subsampling (Random or K-Means coresets of size M << N) starves the
  GP of points inside narrow phase-transition / discontinuity zones, so Expected
  Improvement (EI) misses the peak regime.
- SHD v3.7 compresses the stream into M medoids on the phase manifold
  [Z || kappa * y], preserving the discontinuity boundary so a standard Matérn GP
  fitted on the M-point crystal sees the phase transition immediately.

Experiment (20 seeds, paired Wilcoxon):
1. Stream N = 10,000 noisy observations from a 4D materials/reactor landscape
   with a sharp phase transition (smooth background + step-jump peak zone,
   mirroring JARVIS Lab's HiddenWorldPhase).
2. Compress the stream into M = 120 points via Random, K-Means, and SHD v3.7.
3. Fit an identical Matérn-5/2 GaussianProcessRegressor on each 120-point coreset
   (with alpha = sigma_noise^2 / gamma_weights for weighted coresets).
4. Evaluate on an unseen candidate pool (N_pool = 3,000):
   - gp_rmse_peak: GP prediction RMSE inside the rare phase-transition zone;
   - ei_recall_at_20: fraction of top-20 Expected Improvement (EI) acquisitions
     that land inside the true phase-transition peak;
   - bo_regret_10: simple regret (y_opt - max_{t<=10} y_t) after 10 sequential
     GP+EI acquisitions from the candidate pool.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from shd import SpectralHologramDistiller


def sample_discontinuous_world(n: int, seed: int, noise_std: float = 0.08):
    """4D landscape with a narrow phase-transition pocket ( ~2.5% volume )."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 4))
    # Smooth background activity
    y_clean = (
        0.6 * np.cos(1.3 * X[:, 0])
        + 0.4 * np.sin(1.1 * X[:, 1] * X[:, 2])
        - 0.25 * (X[:, 3] ** 2)
    )
    # Discontinuous phase pocket around (0.85, -0.75, 0.60)
    center = np.array([0.85, -0.75, 0.60])
    r2 = np.sum((X[:, :3] - center) ** 2, axis=1)
    in_pocket = r2 < 0.42**2
    # Sharp discontinuity step + local dome inside the pocket
    y_clean[in_pocket] += 4.5 + 2.2 * np.exp(-3.0 * r2[in_pocket]) * (1.0 + 0.3 * X[in_pocket, 3])
    y_obs = y_clean + noise_std * rng.normal(size=n)
    return X, y_obs, y_clean, in_pocket


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    Z = imp / sigma
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * Z * Z)
    return imp * ndtr(Z) + sigma * pdf


def fit_gp_surrogate(Z_c: np.ndarray, y_c: np.ndarray, w_c: np.ndarray | None, base_noise: float = 0.05):
    if w_c is None:
        alpha = np.full(len(y_c), base_noise, dtype=np.float64)
    else:
        alpha = base_noise / np.clip(w_c, 0.2, 5.0)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=42,
    )
    gp.fit(Z_c, y_c)
    return gp


def run_bo_loop(
    Z_init: np.ndarray,
    y_init: np.ndarray,
    w_init: np.ndarray | None,
    Z_pool: np.ndarray,
    y_pool_obs: np.ndarray,
    y_pool_clean: np.ndarray,
    in_pocket_pool: np.ndarray,
    n_steps: int = 10,
):
    gp = fit_gp_surrogate(Z_init, y_init, w_init)
    mu0, std0 = gp.predict(Z_pool, return_std=True)

    # 1. Surrogate RMSE on the unseen phase-transition pocket before BO steps
    gp_rmse_peak = float(np.sqrt(mean_squared_error(y_pool_clean[in_pocket_pool], mu0[in_pocket_pool])))
    gp_rmse_all = float(np.sqrt(mean_squared_error(y_pool_clean, mu0)))

    # 2. One-shot top-20 EI precision/hit-rate on the phase pocket
    ei0 = expected_improvement(mu0, std0, float(np.max(y_init)))
    top20 = np.argsort(ei0)[-20:]
    ei_hits_20 = int(np.sum(in_pocket_pool[top20]))

    # 3. Sequential 10-step GP+EI active learning loop on the pool
    Z_cur = Z_init.copy()
    y_cur = y_init.copy()
    w_cur = np.ones(len(y_init), dtype=np.float64) if w_init is None else w_init.copy()
    chosen = set()
    best_clean_so_far = float(np.max(y_pool_clean[np.argmin(cdist(Z_init, Z_pool), axis=1)]))
    y_star = float(np.max(y_pool_clean))

    for _ in range(n_steps):
        gp_step = fit_gp_surrogate(Z_cur, y_cur, w_cur)
        mu, std = gp_step.predict(Z_pool, return_std=True)
        ei = expected_improvement(mu, std, float(np.max(y_cur)))
        for idx in chosen:
            ei[idx] = -1.0
        pick = int(np.argmax(ei))
        chosen.add(pick)
        best_clean_so_far = max(best_clean_so_far, float(y_pool_clean[pick]))
        Z_cur = np.vstack([Z_cur, Z_pool[pick : pick + 1]])
        y_cur = np.append(y_cur, y_pool_obs[pick])
        w_cur = np.append(w_cur, 1.0)

    regret_10 = float(max(0.0, y_star - best_clean_so_far))
    return {
        "gp_rmse_all": gp_rmse_all,
        "gp_rmse_peak": gp_rmse_peak,
        "ei_hits_at_20": ei_hits_20,
        "bo_regret_10": regret_10,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--crystal-size", type=int, default=120)
    args = ap.parse_args()

    t0 = time.perf_counter()
    X_pool, y_pool_obs, y_pool_clean, in_pocket_pool = sample_discontinuous_world(3000, seed=999)
    print(
        f"[bridge] Candidate pool: {len(X_pool)} pts, pocket={int(np.sum(in_pocket_pool))} pts "
        f"({100*np.mean(in_pocket_pool):.2f}%), y_star={np.max(y_pool_clean):.3f}"
    )

    records = []
    for seed in range(args.seeds):
        ts = time.perf_counter()
        X_stream, y_stream, _, in_pocket_stream = sample_discontinuous_world(10000, seed=100 + seed)
        scaler = StandardScaler().fit(X_stream)
        Z_stream = scaler.transform(X_stream)
        Z_pool = scaler.transform(X_pool)

        m = args.crystal_size

        # 1. SHD v3.7 crystal
        distiller = SpectralHologramDistiller(
            crystal_size=m,
            high_band_ratio=0.26,
            sketch_dim=256,
            tail_quantile=84.0,
            phase_kappa=3.2,
            random_state=seed,
        )
        for i in range(0, len(X_stream), 2500):
            distiller.partial_fit(X_stream[i : i + 2500], y_stream[i : i + 2500])
        distiller.finalize_crystal()
        Z_shd = scaler.transform(distiller.crystal_X)
        y_shd = np.asarray(distiller.crystal_Y, dtype=np.float64)
        w_shd = distiller.gamma_weights

        # 2. Random coreset
        rng = np.random.default_rng(seed)
        r_idx = rng.choice(len(X_stream), size=m, replace=False)
        Z_rand, y_rand = Z_stream[r_idx], y_stream[r_idx]

        # 3. K-Means coreset
        km = MiniBatchKMeans(n_clusters=m, batch_size=2048, random_state=seed, n_init=3).fit(Z_stream)
        counts = np.bincount(km.labels_, minlength=m).astype(np.float64) + 1.0
        w_km = counts / np.mean(counts)
        km_idx = np.argmin(cdist(km.cluster_centers_, Z_stream, metric="sqeuclidean"), axis=1)
        Z_km, y_km = Z_stream[km_idx], y_stream[km_idx]

        rec = {
            "seed": seed,
            "stream_pocket_pts": int(np.sum(in_pocket_stream)),
            "shd": run_bo_loop(Z_shd, y_shd, w_shd, Z_pool, y_pool_obs, y_pool_clean, in_pocket_pool),
            "random": run_bo_loop(Z_rand, y_rand, None, Z_pool, y_pool_obs, y_pool_clean, in_pocket_pool),
            "kmeans": run_bo_loop(Z_km, y_km, w_km, Z_pool, y_pool_obs, y_pool_clean, in_pocket_pool),
        }
        records.append(rec)
        print(
            f"[bridge] seed {seed:2d}: gp_rmse_peak shd={rec['shd']['gp_rmse_peak']:.2f} "
            f"rand={rec['random']['gp_rmse_peak']:.2f} km={rec['kmeans']['gp_rmse_peak']:.2f} | "
            f"EI@20 hits shd={rec['shd']['ei_hits_at_20']:2d} rand={rec['random']['ei_hits_at_20']:2d} "
            f"km={rec['kmeans']['ei_hits_at_20']:2d} | "
            f"regret10 shd={rec['shd']['bo_regret_10']:.3f} rand={rec['random']['bo_regret_10']:.3f} "
            f"km={rec['kmeans']['bo_regret_10']:.3f} ({time.perf_counter()-ts:.1f}s)"
        )

    tests = {}
    for metric in ["gp_rmse_all", "gp_rmse_peak", "ei_hits_at_20", "bo_regret_10"]:
        higher_is_better = metric == "ei_hits_at_20"
        for pair in ["random", "kmeans"]:
            a = np.array([r["shd"][metric] for r in records], dtype=np.float64)
            b = np.array([r[pair][metric] for r in records], dtype=np.float64)
            diff = a - b
            stat, p = (0.0, 1.0) if np.allclose(diff, 0) else wilcoxon(a, b)
            wins = int(np.sum(diff > 0)) if higher_is_better else int(np.sum(diff < 0))
            tests[f"shd_vs_{pair}_{metric}"] = {
                "median_shd": float(np.median(a)),
                "iqr_shd": float(np.percentile(a, 75) - np.percentile(a, 25)),
                "median_" + pair: float(np.median(b)),
                "iqr_" + pair: float(np.percentile(b, 75) - np.percentile(b, 25)),
                "wins_shd": wins,
                "statistic": float(stat),
                "p_value": float(p),
            }

    out = {
        "n_seeds": args.seeds,
        "crystal_size": args.crystal_size,
        "stream_size": 10000,
        "pool_size": 3000,
        "elapsed_sec": time.perf_counter() - t0,
        "records": records,
        "wilcoxon": tests,
    }
    path = "bridge_jarvis_bo_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved -> {path} ({out['elapsed_sec']:.1f}s)")
    for k, v in tests.items():
        base_key = [x for x in v if x.startswith("median_") and x != "median_shd"][0]
        print(
            f"  {k:34s}: shd={v['median_shd']:.3f} (IQR {v['iqr_shd']:.3f}) "
            f"vs {v[base_key]:.3f} | wins={v['wins_shd']}/{args.seeds} | p={v['p_value']:.2e}"
        )


if __name__ == "__main__":
    main()

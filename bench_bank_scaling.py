"""Does `DomainBank` actually solve retention under repeated drift?

The single-crystal `heal_domain_shift` path accumulates forgetting (see
``bench_lifelong_results.json``). The bank stores one crystal per regime and
routes by spectral passport, which should be architecturally immune to
forgetting -- but only if routing is accurate.

This benchmark simulates K=10 successive operating regimes on the same 4D sensor
(each shifts X and changes the y law, like seasonal or equipment-state drift) at
three drift strengths, and compares three strategies:

  bank    : one crystal per regime, routed by BW distance
  heal    : ONE crystal, sequentially healed with every new regime
  frozen  : ONE crystal built on regime 0, never updated

Drift strength matters: strongly separated regimes are easy to route but punish
a single crystal; near-identical regimes are easy for one crystal but make
routing ambiguous. The sweep exposes that band.

Usage:
    python bench_bank_scaling.py --domains 10 --seeds 3
Writes bench_bank_scaling_v2_results.json (default; all drift conditions in
one file). The historical bench_bank_scaling_results.json is never overwritten
by default.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from shd import DomainBank, HologramCrystalLoader, SpectralHologramDistiller


def make_regime(k, n, seed, shift_scale=1.0, novel=False):
    """Regime k: shifted inputs AND a different input->output law."""
    rng = np.random.default_rng(seed + 1000 * k)
    mu = np.zeros(4)
    mu[:2] = (0.55 * shift_scale) * k          # progressive shift in x0, x1
    scale = 1.0 + 0.06 * shift_scale * k       # slow variance drift
    X = rng.normal(mu, scale, size=(n, 4))
    freq = 2.0 + 0.35 * shift_scale * k        # the law itself changes
    y = (np.sin(freq * X[:, 0])
         + (0.5 + 0.08 * shift_scale * k) * X[:, 1] * X[:, 2]
         + 0.25 * np.cos(1.3 * X[:, 3])
         + 0.10 * rng.normal(size=n))
    if novel:                                  # unseen regime placed off-manifold
        X = X + 4.0
        y = y + 5.0
    return X, y


def build_crystal(X, y, m, seed, batch=1000):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=64,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    for i in range(0, len(X), batch):
        d.partial_fit(X[i : i + batch], y[i : i + batch])
    d.finalize_crystal()
    return d


def rmse_on(loader, X_te, y_te):
    Xc, yc, wc = loader.get_training_data()
    scaler = StandardScaler().fit(Xc)
    krr = KernelRidge(kernel="rbf", alpha=0.08, gamma=1.0 / (Xc.shape[1] * 1.1))
    krr.fit(scaler.transform(Xc), yc, sample_weight=wc)
    return float(np.sqrt(mean_squared_error(y_te, krr.predict(scaler.transform(X_te)))))


def run_condition(k_domains, m, n_train, n_test, seeds, shift_scale, tmpdir):
    per_seed = []
    for seed in seeds:
        regimes = [make_regime(k, n_train + n_test, seed, shift_scale=shift_scale)
                   for k in range(k_domains)]
        Xnov, _ = make_regime(0, n_test, seed, shift_scale=shift_scale, novel=True)

        Xtr = [r[0][:n_train] for r in regimes]
        ytr = [r[1][:n_train] for r in regimes]
        Xte = [r[0][n_train:] for r in regimes]
        yte = [r[1][n_train:] for r in regimes]

        # --- ARM 1: bank, one crystal per regime ---
        bank = DomainBank()
        paths = []
        for k in range(k_domains):
            d = build_crystal(Xtr[k], ytr[k], m, seed + k)
            p = os.path.join(tmpdir, "s{}_sh{}_{}.hologram.npz".format(seed, shift_scale, k))
            d.save_hologram(p)
            paths.append(p)
            bank.add(p)

        bank_quality = [rmse_on(bank[i], Xte[i], yte[i]) for i in range(k_domains)]

        routing_hits = 0
        conf = np.zeros((k_domains, k_domains), dtype=int)
        for k in range(k_domains):
            rep = bank.check_domain_shift(Xte[k][:256])
            conf[k, rep["domain_index"]] += 1
            routing_hits += int(rep["domain_index"] == k)

        false_shift = sum(int(bank.check_domain_shift(Xte[k][:256])["is_shifted"])
                          for k in range(k_domains))
        novel_flag = int(bank.check_domain_shift(Xnov[:256])["is_shifted"])
        total_bytes = sum(os.path.getsize(p) for p in paths)

        # --- ARM 2: one crystal, sequentially healed ---
        d_heal = build_crystal(Xtr[0], ytr[0], m, seed)
        p_heal = os.path.join(tmpdir, "s{}_sh{}_heal.hologram.npz".format(seed, shift_scale))
        d_heal.save_hologram(p_heal)
        loader_heal = HologramCrystalLoader(p_heal)
        heal_curve = [rmse_on(loader_heal, Xte[0], yte[0])]
        for k in range(1, k_domains):
            loader_heal.heal_domain_shift(Xtr[k], ytr[k], n_inject=30, random_state=seed + k)
            heal_curve.append(rmse_on(loader_heal, Xte[0], yte[0]))
        heal_adapt = rmse_on(loader_heal, Xte[-1], yte[-1])

        # --- ARM 3: one crystal, frozen on regime 0 ---
        loader_frozen = HologramCrystalLoader(p_heal)
        frozen_r0 = rmse_on(loader_frozen, Xte[0], yte[0])
        frozen_last = rmse_on(loader_frozen, Xte[-1], yte[-1])

        per_seed.append({
            "seed": int(seed),
            "bank_quality": [round(v, 4) for v in bank_quality],
            "bank_retention_r0": round(bank_quality[0], 4),
            "bank_last": round(bank_quality[-1], 4),
            "routing_accuracy": round(routing_hits / k_domains, 4),
            "routing_confusion": conf.tolist(),
            "in_domain_false_shift_rate": round(false_shift / k_domains, 4),
            "novel_domain_detected": novel_flag,
            "bank_total_bytes": int(total_bytes),
            "heal_retention_r0_curve": [round(v, 4) for v in heal_curve],
            "heal_adapt_last": round(heal_adapt, 4),
            "frozen_retention_r0": round(frozen_r0, 4),
            "frozen_adapt_last": round(frozen_last, 4),
        })
        print("  shift={:.2f} seed={} routing={:.2f} bank_r0={:.3f} bank_last={:.3f} "
              "heal_r0={:.3f} heal_last={:.3f} frozen_last={:.3f}".format(
                  shift_scale, seed, per_seed[-1]["routing_accuracy"],
                  per_seed[-1]["bank_retention_r0"], per_seed[-1]["bank_last"],
                  heal_curve[-1], heal_adapt, frozen_last), flush=True)

    med = lambda key: float(np.median([r[key] for r in per_seed]))
    return {
        "shift_scale": float(shift_scale),
        "summary": {
            "bank_retention_r0": med("bank_retention_r0"),
            "bank_last": med("bank_last"),
            "routing_accuracy": med("routing_accuracy"),
            "in_domain_false_shift_rate": med("in_domain_false_shift_rate"),
            "novel_domain_detected_rate": float(np.mean([r["novel_domain_detected"] for r in per_seed])),
            "bank_total_bytes": med("bank_total_bytes"),
            "heal_retention_r0_end": float(np.median([r["heal_retention_r0_curve"][-1] for r in per_seed])),
            "heal_adapt_last": med("heal_adapt_last"),
            "frozen_retention_r0": med("frozen_retention_r0"),
            "frozen_adapt_last": med("frozen_adapt_last"),
        },
        "records": per_seed,
    }


def run(k_domains=10, m=300, n_train=3000, n_test=800, seeds=(0, 1, 2),
        shift_scales=(1.0, 0.5, 0.25)):
    tmpdir = tempfile.mkdtemp(prefix="shd_bank_")
    t0 = time.perf_counter()
    conditions = {}
    for sh in shift_scales:
        print("=== shift_scale={:.2f} ===".format(sh), flush=True)
        conditions["{:.2f}".format(sh)] = run_condition(
            k_domains, m, n_train, n_test, seeds, sh, tmpdir)
    return {
        "dataset": "bank_scaling",
        "title": "Sequential drift: DomainBank vs sequential heal vs frozen",
        "n_domains": int(k_domains),
        "m": int(m),
        "seeds": [int(s) for s in seeds],
        "n_seeds": len(seeds),
        "shift_scales": [float(s) for s in shift_scales],
        "conditions": conditions,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shift-scales", default="1.0,0.5,0.25",
                    help="per-regime drift strengths; 1.0 = well separated, "
                         "0.25 = near-identical regimes where routing is at risk")
    ap.add_argument("--output", default="bench_bank_scaling_v2_results.json")
    args = ap.parse_args()
    scales = tuple(float(s) for s in args.shift_scales.split(","))
    res = run(k_domains=args.domains, seeds=tuple(range(args.seeds)), shift_scales=scales)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 90)
    print("BANK SCALING: {} regimes, M={}, {} seeds".format(
        res["n_domains"], res["m"], res["n_seeds"]))
    print("=" * 90)
    print("{:<8s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "shift", "routing", "bank_R0", "bank_last", "heal_R0", "heal_last", "frozen_last"))
    for key, cond in res["conditions"].items():
        s = cond["summary"]
        print("{:<8s} {:>8.2f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            key, s["routing_accuracy"], s["bank_retention_r0"], s["bank_last"],
            s["heal_retention_r0_end"], s["heal_adapt_last"], s["frozen_adapt_last"]))
    first = res["conditions"][list(res["conditions"])[0]]["summary"]
    print("\nper-domain footprint: {:.1f} KB".format(
        first["bank_total_bytes"] / 1024 / res["n_domains"]))
    print("Saved -> {} ({:.0f}s)".format(args.output, res["elapsed_sec"]))


if __name__ == "__main__":
    main()

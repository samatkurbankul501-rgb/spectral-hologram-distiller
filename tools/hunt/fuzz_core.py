"""Randomised contract fuzzer for the SHD core.

Blasts the public API with random but *valid* inputs and checks the
invariants the project itself claims:

  C1  crystal medoids are verbatim rows of the input stream
  C2  crystal size == requested crystal_size (or the documented exception)
  C3  gamma weights are finite and mean-normalised to 1
  C4  the artifact reloads and the radar still works
  C5  heal never exceeds max_crystal_size
  C6  heal never mutates the loader when it rejects the request
  C7  no NaN/Inf anywhere in the crystal
  C8  supervised and label-free modes never mix in one stream
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

REPO = "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller"
sys.path.insert(0, REPO)
from shd import DomainBank, HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402

VIOLATIONS = []
tmp = tempfile.mkdtemp(prefix="fuzz_")


def violate(code, msg, dump=None):
    VIOLATIONS.append((code, msg))
    print(f"  !! VIOLATION {code}: {msg}")
    if dump is not None:
        np.savez(os.path.join(tmp, f"violation_{len(VIOLATIONS):02d}.npz"), **dump)


def check_stream(rng):
    n = int(rng.integers(200, 2500))
    d = int(rng.integers(2, 40))
    M = int(rng.integers(5, 400))
    br = float(rng.choice([0.0, 0.1, 0.26, 0.5, 0.75, 1.0]))
    tq = float(rng.choice([50.0, 84.0, 97.0, 99.5]))
    tail = str(rng.choice(["high", "low", "both"]))
    k = int(rng.choice([8, 16, 64, 256]))
    label_free = bool(rng.random() < 0.25)
    deltas = bool(rng.random() < 0.2)

    scale = float(rng.choice([1e-3, 1.0, 1e3]))
    X = rng.normal(size=(n, d)) * scale
    y = (2.0 * X[:, 0] + 0.3 * rng.normal(size=n)) if d > 0 else rng.normal(size=n)
    if rng.random() < 0.5:
        hot = rng.random(n) < 0.05
        y[hot] += 10.0
    if rng.random() < 0.2:
        # constant column
        X[:, 0] = 3.0
    if rng.random() < 0.15:
        # duplicated rows
        X[n // 2:] = X[:n - n // 2]

    kw = dict(crystal_size=M, high_band_ratio=br, sketch_dim=k, tail_quantile=tq,
              tail=tail, random_state=int(rng.integers(0, 1000)),
              include_deltas=deltas, max_init_buffer=int(rng.integers(64, 9000)))
    if rng.random() < 0.3:
        kw["use_residual"] = False
    if rng.random() < 0.2:
        kw["herding_mix"] = float(rng.choice([0.0, 1.0]))
    if rng.random() < 0.2:
        kw["voronoi_temper"] = float(rng.choice([0.0, 1.0]))
    if rng.random() < 0.15:
        kw["adaptive_y_stats"] = True

    cfg = f"n={n} d={d} M={M} br={br} tq={tq} tail={tail} k={k} lf={label_free} deltas={deltas}"

    try:
        dist = SpectralHologramDistiller(**kw)
    except ValueError as e:
        return  # rejected at construction: fine
    except Exception as e:
        violate("C0-ctor", f"{cfg} -> {type(e).__name__}: {e}")
        return

    try:
        bs = int(rng.integers(1, 2000))
        for i in range(0, n, bs):
            xb = X[i:i + bs]
            yb = None if label_free else y[i:i + bs]
            dist.partial_fit(xb, yb)
        rel = dist.finalize_crystal()
    except Exception as e:
        violate("C0-fit", f"{cfg} -> {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return

    # C2 crystal size
    if len(dist.crystal_X) > M:
        violate("C2", f"{cfg} -> crystal has {len(dist.crystal_X)} medoids > M={M}")
    if len(dist.crystal_X) == 0:
        violate("C2", f"{cfg} -> empty crystal")
        return

    # C7 finiteness
    if not np.isfinite(dist.crystal_X).all():
        violate("C7", f"{cfg} -> non-finite crystal_X")
    if not np.isfinite(dist.gamma_weights).all():
        violate("C7", f"{cfg} -> non-finite gamma_weights")
    if not np.isfinite(rel):
        violate("C7", f"{cfg} -> non-finite rel spectral error {rel}")

    # C3 weights
    if abs(float(np.mean(dist.gamma_weights)) - 1.0) > 1e-4:
        violate("C3", f"{cfg} -> gamma weights mean {np.mean(dist.gamma_weights):.6f}")

    # C1 real medoids
    if not deltas:
        raw = X
        hits = 0
        for row in dist.crystal_X[:12]:
            dmin = np.linalg.norm(raw - row, axis=1).min()
            rel_d = dmin / max(np.abs(row).max(), 1e-9)
            hits += 1 if rel_d < 1e-4 else 0
        if hits < 12:
            violate("C1", f"{cfg} -> only {hits}/12 medoids are verbatim stream rows",
                    dump={"X": X, "y": y, "crystal_X": dist.crystal_X,
                          "crystal_Y": dist.crystal_Y, "cfg": np.array(cfg)})

    # C4 artifact round-trip
    p = os.path.join(tmp, "f.npz")
    try:
        dist.save_hologram(p)
        ld = HologramCrystalLoader(p)
        rep = ld.check_domain_shift(X[: min(64, n)])
        if not np.isfinite(rep["bw_distance"]):
            violate("C4", f"{cfg} -> non-finite radar distance")
    except Exception as e:
        violate("C4", f"{cfg} -> save/load/radar {type(e).__name__}: {e}")
        return

    # C5/C6 heal
    cap = int(rng.integers(1, max(2, len(ld.X) * 2)))
    nin = int(rng.integers(1, 80))
    before = len(ld.X)
    before_x = ld.X.copy()
    try:
        ld.heal_domain_shift(X[: min(200, n)] * 1.7 + 1.0, None if label_free else y[: min(200, n)],
                             n_inject=nin, max_crystal_size=cap, random_state=1)
        if len(ld.X) > cap:
            violate("C5", f"{cfg} cap={cap} nin={nin} -> healed crystal has {len(ld.X)}")
    except ValueError:
        # rejected: the loader must be untouched
        if len(ld.X) != before or not np.array_equal(ld.X, before_x):
            violate("C6", f"{cfg} cap={cap} nin={nin} -> loader MUTATED while rejecting the heal")
    except Exception as e:
        violate("C5", f"{cfg} cap={cap} nin={nin} -> {type(e).__name__}: {e}")

    # C8 mode mixing
    if rng.random() < 0.3:
        try:
            dist.partial_fit(X[:5], None if not label_free else y[:5])
            violate("C8", f"{cfg} -> switching supervised/label-free was accepted")
        except ValueError:
            pass
        except Exception as e:
            violate("C8", f"{cfg} -> wrong error type for mode switch: {type(e).__name__}")


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    for i in range(rounds):
        print(f"--- fuzz round {i + 1}/{rounds}")
        check_stream(rng)
    print("\n" + "=" * 72)
    print(f"VIOLATIONS: {len(VIOLATIONS)}")
    for code, msg in VIOLATIONS:
        print(f"  {code}: {msg}")
    print("=" * 72)


if __name__ == "__main__":
    main()

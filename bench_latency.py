"""Radar latency benchmark on UCI Superconductivity.

Measures ``HologramCrystalLoader.check_domain_shift`` for a bounded set of
window sizes on a fixed crystal built on the Domain A training split
(``y < 40``, ``test_size=0.20``, ``random_state=42`` -- the same split the
published numbers used, so timings stay comparable).

Truthfulness contract (regression fix for the legacy ``5780`` label):
- every measured record reports ``requested`` and ``actual_rows``, and
  ``measured_windows`` lists exactly the row counts that were really timed;
- a requested window larger than the available held-out pool is skipped with
  an explicit reason in ``skipped_windows`` -- never silently timed on a
  truncated slice. The legacy ``XA_te[:B]`` slice was that bug: with only
  ~1.3k held-out rows available it timed the whole pool while labelling the
  result ``5780``;
- ``--full-heldout`` optionally times the entire available pool as its own
  ``full`` entry, labelled with the true row count (never padded or wrapped).

Provenance of the legacy artifact: ``bench_latency_results.json`` holds
measured medians spanning 4.78--24.93 ms over the labels it recorded, but its
``5780`` entry did not measure 5780 rows; those numbers are historical and are
not re-derived here. This script defaults to ``bench_latency_v2_results.json``
and never rewrites the legacy artifact unless ``--output`` points at it with
``--force``.

Usage:
    python bench_latency.py [--windows 64,256,1024] [--repeats 5]
                            [--full-heldout] [--output PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from sklearn.model_selection import train_test_split

from bench_seeds import load_superconduct
from shd import HologramCrystalLoader, SpectralHologramDistiller

DEFAULT_WINDOWS = (64, 256, 1024)
REPEATS = 5
DEFAULT_OUTPUT = "bench_latency_v2_results.json"
CRYSTAL_SIZE = 200
SKETCH_DIM = 256
FIT_BATCH = 2500
SPLIT_TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

EXIT_OK = 0
EXIT_NOTHING_MEASURED = 1
EXIT_FAILED = 3


def _fail(message: str) -> int:
    print(f"[error] {message}", file=sys.stderr)
    return EXIT_FAILED


# --- window planning ----------------------------------------------------------


def plan_windows(requested, available, include_full=False):
    """Resolve requested window sizes against the real held-out pool size.

    Returns ``(measured, skipped)``. Every ``measured`` entry has
    ``requested == actual_rows`` -- numpy slicing would truncate an oversize
    request silently, which is what produced the bogus legacy ``5780`` record,
    so such requests go to ``skipped`` with an explicit reason instead.
    Duplicates collapse onto the first occurrence, and ``include_full`` adds
    the entire pool as its own ``"full"`` entry. Non-positive requests are
    skipped rather than guessed at.
    """
    available = int(available)
    measured, skipped = [], []
    seen = set()
    for b in requested:
        b = int(b)
        label = str(b)
        if b in seen:
            skipped.append({"label": label, "requested": b, "reason": "duplicate request"})
            continue
        seen.add(b)
        if b <= 0:
            skipped.append({"label": label, "requested": b, "reason": "window size must be positive"})
            continue
        if b > available:
            skipped.append({
                "label": label, "requested": b,
                "reason": f"requested {b} rows exceeds the {available} held-out rows available",
            })
            continue
        measured.append({"label": label, "requested": b, "actual_rows": b})
    if include_full:
        if available > 0:
            measured.append({"label": "full", "requested": None, "actual_rows": available})
        else:
            skipped.append({"label": "full", "requested": None, "reason": "held-out pool is empty"})
    return measured, skipped


# --- measurement ---------------------------------------------------------------


def measure_latency(loader, rows, repeats=REPEATS, timer=time.perf_counter, warmup=True):
    """Time ``loader.check_domain_shift(rows)``: one warm-up call, then repeats.

    The external timer is a parameter so tests can inject a deterministic one;
    the default matches the legacy artifact's timing method.
    """
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup:
        loader.check_domain_shift(rows)
    times = []
    for _ in range(repeats):
        t0 = timer()
        loader.check_domain_shift(rows)
        times.append((timer() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n": repeats,
    }


def default_distiller_factory(crystal_size=CRYSTAL_SIZE, sketch_dim=SKETCH_DIM):
    """Same hyper-parameters as the legacy artifact (crystal 200, sketch 256)."""

    def factory():
        return SpectralHologramDistiller(
            crystal_size=crystal_size, sketch_dim=sketch_dim, tail_quantile=84.0,
            phase_kappa=3.2, random_state=0,
        )

    return factory


def build_loader(XA_tr, yA_tr, distiller_factory=None, loader_factory=None,
                 crystal_dir=None, fit_batch=FIT_BATCH):
    """Fit the crystal on the training split, save it, return ``(loader, path)``.

    ``distiller_factory``/``loader_factory`` exist so tests can inject stubs;
    the defaults are the real distiller and ``HologramCrystalLoader``.
    """
    fit_batch = int(fit_batch)
    if fit_batch < 1:
        raise ValueError("fit_batch must be >= 1")
    distiller_factory = distiller_factory or default_distiller_factory()
    loader_factory = loader_factory or HologramCrystalLoader

    d = distiller_factory()
    for i in range(0, len(XA_tr), fit_batch):
        d.partial_fit(XA_tr[i : i + fit_batch], yA_tr[i : i + fit_batch])
    d.finalize_crystal()

    if crystal_dir is None:
        crystal_dir = tempfile.mkdtemp(prefix="shd_latency_")
    crystal_dir = str(crystal_dir)
    os.makedirs(crystal_dir, exist_ok=True)
    path = os.path.join(crystal_dir, "crystal.npz")
    d.save_hologram(path, include_projector=True)
    return loader_factory(path), path


def default_data_provider(allow_fallback=False):
    """Load the Domain-A split the legacy benchmark used, without silent swaps.

    ``bench_seeds.load_superconduct()`` silently falls back to California housing
    when the OpenML fetch fails. That fallback is refused here by default: the
    latency artifact would then describe a different dataset than its label.
    """
    X, y, _, title = load_superconduct()
    if not allow_fallback and "fallback" in title.lower():
        raise RuntimeError(
            f"load_superconduct() fell back to {title!r}; latency numbers would not "
            "come from the dataset this benchmark documents. Re-run with "
            "--allow-dataset-fallback to proceed on that dataset anyway."
        )
    A = y < 40.0
    XA, yA = X[A], y[A]
    XA_tr, XA_te, yA_tr, _ = train_test_split(
        XA, yA, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    if len(XA_tr) == 0 or len(XA_te) == 0:
        raise RuntimeError(
            f"degenerate Domain-A split: train={len(XA_tr)}, held_out={len(XA_te)}"
        )
    return title, XA_tr, yA_tr, XA_te


def run_benchmark(XA_tr, yA_tr, XA_te, title, windows=DEFAULT_WINDOWS, repeats=REPEATS,
                  include_full=False, distiller_factory=None, loader_factory=None,
                  crystal_dir=None, fit_batch=FIT_BATCH, timer=time.perf_counter,
                  crystal_size=CRYSTAL_SIZE, verbose=True):
    """Time every honour-able window; return the truthfulness-checked payload."""
    available = len(XA_te)
    loader, _path = build_loader(
        XA_tr, yA_tr, distiller_factory=distiller_factory, loader_factory=loader_factory,
        crystal_dir=crystal_dir, fit_batch=fit_batch,
    )
    measured, skipped = plan_windows(windows, available, include_full=include_full)

    records = {}
    for m in measured:
        rows = XA_te[: m["actual_rows"]]
        if len(rows) != m["actual_rows"]:  # defensive: never time a truncated slice
            raise AssertionError(
                f"window slice for {m['label']} yielded {len(rows)} rows, "
                f"expected {m['actual_rows']}"
            )
        records[m["label"]] = dict(m, **measure_latency(loader, rows, repeats, timer))

    payload = {
        "dataset": title,
        "crystal_size": int(crystal_size),
        "repeats": int(repeats),
        "heldout_rows_available": int(available),
        "windows_requested": [int(b) for b in windows],
        "measured_windows": [int(r["actual_rows"]) for r in measured],
        "records": records,
        "skipped_windows": skipped,
        "range_ms": (
            [float(min(r["median_ms"] for r in records.values())),
             float(max(r["median_ms"] for r in records.values()))]
            if records else None
        ),
    }

    if verbose:
        for label, r in records.items():
            print(f"B={label:>5s}: median {r['median_ms']:.2f} ms "
                  f"(min {r['min_ms']:.2f}, max {r['max_ms']:.2f}) "
                  f"[requested {r['requested']}, actual {r['actual_rows']}]")
        for s in skipped:
            print(f"B={s['label']:>5s}: skipped -- {s['reason']}", file=sys.stderr)
        print(f"held-out pool: {available} rows; measured windows: "
              f"{payload['measured_windows']} of {len(windows)} requested")
    return payload


def write_json(payload, path):
    """Atomic write so an interrupted run never leaves a partial artifact."""
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def parse_windows(text):
    """'64,256,1024' -> [64, 256, 1024]; a junk token is a hard error."""
    values = []
    for part in str(text).replace(" ", ",").split(","):
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            raise SystemExit(f"--windows expects integers separated by commas, got {part!r}")
    return values


def main(argv=None, *, data_provider=None, distiller_factory=None, loader_factory=None,
         timer=None):
    ap = argparse.ArgumentParser(
        description="Radar latency benchmark on UCI Superconductivity "
                    "(records requested and actual window sizes)."
    )
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                    help="comma-separated requested window sizes (default: %(default)s). "
                         "Requests larger than the held-out pool are explicitly skipped.")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--full-heldout", action="store_true",
                    help="also time the entire available held-out pool as the 'full' entry, "
                         "labelled with its true row count (never padded)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="output JSON path (default: %(default)s; the legacy "
                         "bench_latency_results.json is never written implicitly)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --output file")
    ap.add_argument("--fit-batch", type=int, default=FIT_BATCH,
                    help="partial_fit chunk size on the training split (default: %(default)s)")
    ap.add_argument("--allow-dataset-fallback", action="store_true",
                    help="proceed even if load_superconduct() falls back to another dataset")
    args = ap.parse_args(argv)

    if args.repeats < 1:
        return _fail("--repeats must be >= 1")
    if args.fit_batch < 1:
        return _fail("--fit-batch must be >= 1")
    windows = parse_windows(args.windows)
    if not windows and not args.full_heldout:
        return _fail("no window sizes requested: pass --windows or --full-heldout")
    if os.path.isdir(args.output):
        return _fail(f"--output {args.output} is a directory")
    if os.path.exists(args.output) and not args.force:
        return _fail(f"{args.output} already exists; pass --force to overwrite it "
                     "(results artifacts are never overwritten implicitly)")

    provider = data_provider or (
        lambda: default_data_provider(allow_fallback=args.allow_dataset_fallback)
    )
    try:
        title, XA_tr, yA_tr, XA_te = provider()
    except Exception as e:
        return _fail(f"data load failed: {e}")

    try:
        payload = run_benchmark(
            XA_tr, yA_tr, XA_te, title, windows=windows, repeats=args.repeats,
            include_full=args.full_heldout, distiller_factory=distiller_factory,
            loader_factory=loader_factory, timer=timer or time.perf_counter,
            fit_batch=args.fit_batch,
        )
    except Exception as e:
        return _fail(f"benchmark failed: {e}")

    write_json(payload, args.output)
    if payload["records"]:
        print(f"range: {payload['range_ms']} -> {args.output}")
        return EXIT_OK
    print(f"no windows were measured; skipped: {payload['skipped_windows']} -> {args.output}",
          file=sys.stderr)
    return EXIT_NOTHING_MEASURED


if __name__ == "__main__":
    raise SystemExit(main())

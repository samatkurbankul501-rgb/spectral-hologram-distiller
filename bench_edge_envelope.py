"""Deployment envelope: what the crystal actually costs on one CPU core.

The strategic claim ("runs on an edge box with no GPU and no network") is only
credible as measured numbers. This reports synthetic Gaussian/sine workloads,
not measurements of the real superconductivity dataset. For each case:

  * artifact size on disk (KB),
  * resident-set increase of the Python process after loading + radar,
  * RSS of the process holding a crystal,
  * per-window radar latency (ms) for several batch sizes, single-threaded,
  * full-crystal inference latency (KRR predict on the crystal) if applicable,
  * training throughput (rows/s of partial_fit).

Single-core enforcement: BLAS/OpenMP thread caps are set before numpy imports,
so the numbers reflect one core rather than however many the host has.

Usage:
    python bench_edge_envelope.py --repeats 3
Writes bench_edge_envelope_v2_results.json (historical output is preserved).
"""

from __future__ import annotations

import os

# Must precede the numpy import: pin BLAS/OMP to one thread.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import platform
import time

import numpy as np

from shd import HologramCrystalLoader, SpectralHologramDistiller


def rss_mb():
    """Resident set of this process in MB.

    Uses explicit ctypes signatures: without ``restype = HANDLE`` on
    ``GetCurrentProcess`` the returned pseudo-handle is truncated on 64-bit and
    the call silently reports 0 bytes.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if ok:
            return counters.WorkingSetSize / (1024 * 1024)
        raise OSError("GetProcessMemoryInfo failed")
    except Exception:
        try:
            import resource
            import sys
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports ru_maxrss in KB, macOS in bytes.
            return rss / 1048576.0 if sys.platform == "darwin" else rss / 1024.0
        except Exception:
            return float("nan")


def build(X, y, m, seed, batch=2500):
    d = SpectralHologramDistiller(
        crystal_size=m, high_band_ratio=0.26, sketch_dim=256,
        tail_quantile=84.0, phase_kappa=3.2, random_state=seed,
    )
    t0 = time.perf_counter()
    n = 0
    for i in range(0, len(X), batch):
        d.partial_fit(X[i : i + batch], y[i : i + batch])
        n += len(X[i : i + batch])
    fit_s = time.perf_counter() - t0
    d.finalize_crystal()
    return d, n / fit_s


def timeit(fn, repeats=20):
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def measure_case(name, d_feat, n, m, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_feat))
    y = np.sin(X[:, 0]) + 0.3 * rng.normal(size=n)

    base_rss = rss_mb()
    d, throughput = build(X, y, m, seed)
    peak_rss = rss_mb()

    path = os.path.join(os.environ.get("TEMP", "."), "edge_{}.hologram.npz".format(name))
    kb = d.save_hologram(path)
    load_t0 = time.perf_counter()
    loader = HologramCrystalLoader(path)
    load_ms = (time.perf_counter() - load_t0) * 1000.0
    loaded_rss = rss_mb()

    lat = {}
    for b in (64, 256, 1024):
        window = rng.normal(size=(b, d_feat))
        lat[str(b)] = timeit(lambda w=window: loader.check_domain_shift(w))

    Xc, yc, wc = loader.get_training_data()
    os.remove(path)
    return {
        "case": name,
        "n_features": int(d_feat),
        "n_train_rows": int(n),
        "crystal_points": int(len(yc)),
        "artifact_kb": float(kb),
        "rss_base_mb": base_rss,
        "rss_after_fit_mb": peak_rss,
        "rss_after_load_mb": loaded_rss,
        "rss_delta_load_mb": loaded_rss - peak_rss,
        "crystal_load_ms": load_ms,
        "radar_ms": lat,
        "partial_fit_rows_per_s": float(throughput),
    }


def run(cases, m, seeds, repeats=3):
    """Median over `repeats` independent measurement passes (latency is noisy)."""
    out = {"dataset": "edge_envelope", "title": "Synthetic on-device cost on one CPU core",
           "data_source": "synthetic: X Gaussian, y=sin(X[:,0])+Gaussian noise",
           "m": int(m), "python": platform.python_version(),
           "platform": platform.platform(), "thread_env": 1,
           "repeats": int(repeats), "cases": [], "elapsed_sec": 0.0}
    t_all = time.perf_counter()

    for name, d_feat, n in cases:
        passes = [measure_case(name, d_feat, n, m, seed=i) for i in range(repeats)]
        med = lambda k: float(np.median([p[k] for p in passes]))
        case = {
            "case": name,
            "n_features": int(d_feat),
            "n_train_rows": int(n),
            "crystal_points": int(passes[0]["crystal_points"]),
            "artifact_kb": med("artifact_kb"),
            "rss_base_mb": med("rss_base_mb"),
            "rss_after_fit_mb": med("rss_after_fit_mb"),
            "rss_after_load_mb": med("rss_after_load_mb"),
            "rss_delta_load_mb": med("rss_delta_load_mb"),
            "crystal_load_ms": med("crystal_load_ms"),
            "radar_ms": {b: float(np.median([p["radar_ms"][b] for p in passes]))
                         for b in ("64", "256", "1024")},
            "partial_fit_rows_per_s": med("partial_fit_rows_per_s"),
            "radar_ms_spread": {b: [float(np.min([p["radar_ms"][b] for p in passes])),
                                    float(np.max([p["radar_ms"][b] for p in passes]))]
                                for b in ("64", "256", "1024")},
        }
        out["cases"].append(case)
        print("  {:<18s} {:.1f} KB | radar(med) 64/256/1024 = {:.2f}/{:.2f}/{:.2f} ms "
              "| RSS {:.1f}+{:.1f} MB | fit {:,.0f} rows/s".format(
                  name, case["artifact_kb"], case["radar_ms"]["64"],
                  case["radar_ms"]["256"], case["radar_ms"]["1024"],
                  case["rss_base_mb"], case["rss_delta_load_mb"],
                  case["partial_fit_rows_per_s"]), flush=True)

    out["elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", default="bench_edge_envelope_v2_results.json")
    args = ap.parse_args()
    cases = [("synthetic-sensor-4d", 4, 16000), ("synthetic-gaussian-sine-81d", 81, 17010)]
    res = run(cases, args.m, None, repeats=args.repeats)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\nSaved -> {} ({:.0f}s)".format(args.output, res["elapsed_sec"]))


if __name__ == "__main__":
    main()

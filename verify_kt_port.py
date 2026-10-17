"""Verify kt_baseline against the OFFICIAL Microsoft goodpoints X-path.

Skips cleanly when the goodpoints source is not available locally. To enable:
    python -m pip download goodpoints --no-deps --no-binary :all: -d _gp_src
    tar -xzf _gp_src/goodpoints-*.tar.gz -C _gp_src

The Cython extension (`goodpoints.ktc`) is only used by the K-matrix paths, so a
stub module lets the pure-numpy X-path (split_X/best_X/refine_X) be imported
and executed on Windows.

Usage: python verify_kt_port.py [--src PATH]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import types

import numpy as np

from kt_baseline import kt_thin, make_gauss_kernels, median_bandwidth_sq


def load_official(src_root):
    kt_path = os.path.join(src_root, "kt.py")
    if not os.path.exists(kt_path):
        return None
    pkg = types.ModuleType("goodpoints")
    pkg.__path__ = [src_root]
    sys.modules["goodpoints"] = pkg
    for name in ("tictoc", "util"):
        path = os.path.join(src_root, f"{name}.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(f"goodpoints.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"goodpoints.{name}"] = mod
        spec.loader.exec_module(mod)
    stub = types.ModuleType("goodpoints.ktc")
    sys.modules["goodpoints.ktc"] = stub
    setattr(pkg, "ktc", stub)
    spec = importlib.util.spec_from_file_location("goodpoints.kt", kt_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["goodpoints.kt"] = mod
    spec.loader.exec_module(mod)
    return mod


def official_kernels(var):
    def gauss(y, X):
        return np.exp(-0.5 * np.sum((y - X) ** 2, axis=1) / var)

    def gauss_rt(y, X):
        return np.exp(-0.5 * np.sum((y - X) ** 2, axis=1) / (var / 2))

    return gauss, gauss_rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_gp_src")
    args = ap.parse_args()

    candidates = [args.src] + glob.glob(os.path.join(args.src, "goodpoints-*", "goodpoints"))
    src_root = next((c for c in candidates if os.path.exists(os.path.join(c, "kt.py"))), None)
    if src_root is None:
        print("SKIP: goodpoints source not found (see module docstring to fetch it)")
        return 0

    official = load_official(src_root)
    if official is None:
        print("SKIP: goodpoints package incomplete")
        return 0
    print(f"official goodpoints X-path loaded from {src_root}")

    rng = np.random.default_rng(0)
    X2 = np.vstack([rng.normal(0, 1, (600, 2)), rng.normal(6, 1, (60, 2))])
    X4 = rng.normal(size=(800, 4))
    cases = [("2D mixture m=1", X2, 1), ("2D mixture m=2", X2, 2),
             ("4D gauss m=2", X4, 2), ("4D gauss m=3", X4, 3)]

    all_match = True
    for label, X, m in cases:
        var = median_bandwidth_sq(X, seed=0)
        swap_o, split_o = official_kernels(var)
        for seed in (7, 11, 23):
            off_idx = np.asarray(official.thin(X, m, split_kernel=split_o, swap_kernel=swap_o, seed=seed))
            my_idx = np.asarray(kt_thin(X, m, var, seed=seed))
            same = np.array_equal(off_idx, my_idx)
            all_match &= same
            print(f"  {label} seed={seed}: identical={same} (n={len(my_idx)})")

    print("VERDICT:", "PORT IS BIT-EXACT vs OFFICIAL CODE" if all_match else "PORT DIFFERS")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())

"""Command-line interface for Spectral Hologram Distiller (SHD).

Usage examples:
    python -m shd inspect artifacts/superconduct_v37.hologram.npz
    python -m shd distill --input stream.csv --target-col -1 --output out.hologram.npz
    python -m shd radar --crystal out.hologram.npz --input new_batch.csv --target-col -1
    python -m shd heal --crystal out.hologram.npz --input drifted.csv --target-col -1 --output healed.hologram.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .distiller import HologramCrystalLoader, SpectralHologramDistiller
from .presets import PRESETS as _PRESET_CHOICES


def _load_tabular(path: str, target_col: int | None = None, delimiter: str | None = None):
    """Load `.npy`, `.npz`, or `.csv`/`.tsv` without requiring pandas."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        X_raw = np.asarray(arr, dtype=np.float64)
        y_raw = None
    elif ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if "X" in data.files:
            X_raw = data["X"].astype(np.float64)
            y_raw = data["y"].astype(np.float64) if "y" in data.files else None
            # An explicit y array takes precedence over --target-col. Only fall
            # through to the target-column split when the NPZ has no y.
            if y_raw is not None or target_col is None:
                return X_raw, y_raw
        elif len(data.files) == 1:
            X_raw = data[data.files[0]].astype(np.float64)
            y_raw = None
        else:
            # Ambiguity is dangerous here: silently scoring e.g. an integer
            # "labels" array as the feature matrix produces nonsense reports.
            raise ValueError(
                f"{path} contains several arrays {sorted(data.files)} and no 'X' "
                "key; save the feature matrix under the key 'X' (optional 'y')"
            )
    else:
        sep = delimiter if delimiter is not None else ("," if ext != ".tsv" else "\t")
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        tokens = [t.strip() for t in first_line.split(sep)] if sep else first_line.split()
        has_header = False
        try:
            for tok in tokens:
                float(tok)
        except ValueError:
            has_header = True
        # ndmin=2: a single-row CSV would otherwise load as a 1-D array of
        # features and be silently reinterpreted as a one-column dataset.
        X_raw = np.loadtxt(
            path, delimiter=sep, skiprows=1 if has_header else 0, dtype=np.float64, ndmin=2
        )
        y_raw = None

    if X_raw.ndim == 1:
        X_raw = X_raw[:, None]
    if X_raw.size == 0 or X_raw.shape[0] == 0:
        raise ValueError(f"{path} contains no data rows")
    if target_col is not None and y_raw is None:
        idx = target_col if target_col >= 0 else (X_raw.shape[1] + target_col)
        if idx < 0 or idx >= X_raw.shape[1]:
            raise ValueError(f"target_col={target_col} out of bounds for shape {X_raw.shape}")
        y_raw = X_raw[:, idx].copy()
        keep = [j for j in range(X_raw.shape[1]) if j != idx]
        X_raw = X_raw[:, keep].copy()
        if X_raw.shape[1] == 0:
            raise ValueError(
                f"cannot strip target column {target_col}: the input has a single "
                "column, so no features would remain"
            )

    return X_raw, y_raw


def _npz_path(path: str) -> str:
    """Return the path numpy will actually write.

    np.savez_compressed appends .npz when the name lacks it, so a
    user-supplied -o out.hologram writes out.hologram.npz -- and the
    follow-up getsize/reload then fails on the ORIGINAL name
    (FileNotFoundError) while a stray file stays on disk. Normalising
    the name once keeps writing, measuring and reporting on one path.
    """
    p = str(path)
    if not p.lower().endswith(".npz"):
        p = p + ".npz"
    return p


def cmd_inspect(args: argparse.Namespace) -> int:
    loader = HologramCrystalLoader(
        args.crystal, include_deltas=True if args.include_deltas else None
    )
    info = loader.summary()
    print(json.dumps(info, indent=2))
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Re-save a legacy crystal with the current schema (v3.14.3).

    Loads the legacy artifact, then saves it with the current loader
    format (schema_version present, projector optional). The migrated
    artifact is bit-compatible with older loaders via the schema-1
    contract, but newer fields (schema_version, shd_version) travel with
    the file. Used for artifacts like superconduct_v37.hologram.npz that
    predate schema_version.
    """
    loader = HologramCrystalLoader(
        args.crystal, include_deltas=True if args.include_deltas else None
    )
    before = loader.summary()
    # Loader.save() persists the current crystals; optional projector is
    # kept if present (otherwise migration still yields a versioned file).
    out_path = _npz_path(args.output)
    size_kb = loader.save(out_path)
    after = HologramCrystalLoader(out_path).summary()
    out = {
        "input": args.crystal,
        "output": out_path,
        "file_size_kb": round(float(size_kb), 2),
        "before": {
            "schema_version": before["schema_version"],
            "shd_version": before["shd_version"],
            "n_medoids": before["n_medoids"],
            "has_projector": before["has_projector"],
        },
        "after": {
            "schema_version": after["schema_version"],
            "shd_version": after["shd_version"],
            "n_medoids": after["n_medoids"],
            "has_projector": after["has_projector"],
        },
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    X, y = _load_tabular(args.input, target_col=args.target_col)
    # v3.15.0: a named preset may set the operating point; an explicit CLI
    # flag always wins over the preset for the same knob.
    from .presets import apply_preset as _apply_preset

    preset_params = {}
    if getattr(args, "preset", None):
        preset_params = dict(_apply_preset(args.preset))
    # CLI flags are None when unset; preset value wins, else constructor
    # logic fills a concrete number. An explicit flag (even one equal to
    # the old default, e.g. 84.0) is a real flag and overrides the preset.
    tail_quantile = args.tail_quantile
    if "tail_quantile" in preset_params:
        if tail_quantile is None:
            tail_quantile = preset_params.pop("tail_quantile")
        else:
            preset_params.pop("tail_quantile", None)
    elif tail_quantile is None:
        tail_quantile = 84.0
    high_band_ratio = args.high_band_ratio
    if "high_band_ratio" in preset_params:
        if high_band_ratio is None:
            high_band_ratio = preset_params.pop("high_band_ratio")
        else:
            preset_params.pop("high_band_ratio", None)
    elif high_band_ratio is None:
        high_band_ratio = 0.26
    global_guard = bool(args.global_guard)
    if "global_guard" in preset_params and not args.global_guard:
        global_guard = bool(preset_params.pop("global_guard"))
    else:
        preset_params.pop("global_guard", None)
    residual_weight = args.residual_weight
    if residual_weight is None:
        residual_weight = preset_params.pop("residual_weight", 1.0)
    else:
        preset_params.pop("residual_weight", None)
    rff_mode = args.rff_mode
    if rff_mode is None:
        rff_mode = preset_params.pop("rff_mode", "dual")
    else:
        preset_params.pop("rff_mode", None)
    # Any remaining preset key is a knob the CLI does not expose yet; it is
    # still honoured, because the preset is a measured operating point.
    distiller = SpectralHologramDistiller(
        crystal_size=args.crystal_size,
        high_band_ratio=high_band_ratio,
        sketch_dim=args.sketch_dim,
        tail_quantile=tail_quantile,
        phase_kappa=args.phase_kappa,
        random_state=args.seed,
        max_init_buffer=args.max_init_buffer,
        include_deltas=args.include_deltas,
        tail=args.tail,
        adaptive_y_stats=args.adaptive_y_stats,
        on_nonfinite=args.on_nonfinite,
        passport_reference=args.passport_reference,
        global_guard=global_guard,
        residual_weight=residual_weight,
        rff_mode=rff_mode,
        **preset_params,
    )
    B = max(1, int(args.batch_size))
    for i in range(0, len(X), B):
        yb = None if y is None else y[i : i + B]
        distiller.partial_fit(X[i : i + B], yb)
    rel_err = distiller.finalize_crystal()
    # v3.15.2: numpy appends .npz to a name that lacks it; writing,
    # measuring and reporting must use the SAME name.
    out_path = _npz_path(args.output)
    size_kb = distiller.save_hologram(out_path, include_projector=True)
    out = {
        "output": out_path,
        "file_size_kb": round(float(size_kb), 2),
        "n_seen": int(distiller.n_seen),
        "crystal_size": int(len(distiller.crystal_X)),
        "unsupervised_mode": bool(distiller.unsupervised_mode),
        "rel_spectral_error": round(float(rel_err), 4),
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_radar(args: argparse.Namespace) -> int:
    loader = HologramCrystalLoader(
        args.crystal, include_deltas=True if args.include_deltas else None
    )
    X, _ = _load_tabular(args.input, target_col=args.target_col)
    report = loader.check_domain_shift(
        X, streaming=args.streaming, mean_shift=args.mean_shift, panel=args.panel
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_heal(args: argparse.Namespace) -> int:
    loader = HologramCrystalLoader(
        args.crystal, include_deltas=True if args.include_deltas else None
    )
    X, y = _load_tabular(args.input, target_col=args.target_col)
    res = loader.heal_domain_shift(
        X,
        y,
        n_inject=args.n_inject,
        max_crystal_size=args.max_crystal_size,
        force=args.force,
        random_state=args.seed,
        streaming=args.streaming,
    )
    if args.output:
        out_path = _npz_path(args.output)
    else:
        # v3.14.2 non-destructive default: never overwrite the input crystal.
        # c.hologram.npz -> c_healed.hologram.npz; c.npz -> c_healed.npz
        base, ext = os.path.splitext(args.crystal)
        if base.endswith(".hologram"):
            out_path = base[: -len(".hologram")] + "_healed.hologram" + ext
        else:
            out_path = base + "_healed" + ext
        # v3.15.2: the derived name may still lack .npz (a crystal named
        # c.hologram); normalise it like an explicit -o.
        out_path = _npz_path(out_path)
    if res["healed"]:
        size_kb = loader.save(out_path)
        res["output"] = out_path
        res["file_size_kb"] = round(float(size_kb), 2)
        res["n_medoids_total"] = int(len(loader.X))
    if res.get("warning"):
        print("WARNING: " + res["warning"], file=sys.stderr)
    print(json.dumps(res, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="shd", description="Spectral Hologram Distiller (SHD) CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect a .hologram.npz crystal")
    p_ins.add_argument("crystal", help="Path to .hologram.npz file")
    p_ins.add_argument(
        "--include-deltas", action="store_true",
        help="Treat a legacy (pre-3.9.1) artifact as a delta crystal",
    )
    p_ins.set_defaults(func=cmd_inspect)

    p_dis = sub.add_parser("distill", help="Distill a CSV/NPY/NPZ dataset into a .hologram.npz crystal")
    p_dis.add_argument("--input", "-i", required=True, help="Input CSV/TSV/NPY/NPZ path")
    p_dis.add_argument("--output", "-o", required=True, help="Output .hologram.npz path")
    p_dis.add_argument("--target-col", type=int, default=None, help="Target column index (omit for label-free mode)")
    p_dis.add_argument("--crystal-size", type=int, default=300)
    p_dis.add_argument("--high-band-ratio", type=float, default=None)
    p_dis.add_argument("--sketch-dim", type=int, default=256)
    p_dis.add_argument("--tail-quantile", type=float, default=None)
    p_dis.add_argument("--phase-kappa", type=float, default=3.2)
    p_dis.add_argument("--batch-size", type=int, default=2500)
    p_dis.add_argument("--max-init-buffer", type=int, default=8192)
    p_dis.add_argument("--include-deltas", action="store_true", help="Augment RFF sketch with streaming first differences")
    p_dis.add_argument(
        "--tail", choices=("high", "low", "both"), default="high",
        help="Which y-tail the phase-peak trigger protects (default: high)",
    )
    p_dis.add_argument(
        "--adaptive-y-stats", action="store_true",
        help="Keep y_mean/y_std calibrated after lattice build (target-drift streams)",
    )
    p_dis.add_argument(
        "--on-nonfinite", choices=("raise", "drop"), default="raise",
        help="NaN/Inf policy: fail loudly (default) or skip affected rows",
    )
    p_dis.add_argument(
        "--preset", default=None, choices=sorted(_PRESET_CHOICES),
        help=(
            "Named measured operating point from shd.presets (e.g. "
            "sparse_auto, critical_first_no_residual). An explicit flag "
            "for the same knob overrides the preset."
        ),
    )
    p_dis.add_argument(
        "--residual-weight", type=float, default=None,
        help=(
            "Weight of the residual term in the phase-peak score. "
            "0.0 keeps the y-z trigger only (critical-first; v3.15.0). "
            "Default: constructor default (1.0)."
        ),
    )
    p_dis.add_argument(
        "--rff-mode", default=None, choices=("dual", "single_low", "single_high"),
        help=(
            "Bochner frequency layout: 'dual' (published default), "
            "'single_low' or 'single_high' (ablation probes, v3.15.0)."
        ),
    )
    p_dis.add_argument(
        "--global-guard", action="store_true",
        help="Swap the lowest-weight rows for K-Means global medoids (PM2.5 guard).",
    )
    p_dis.add_argument("--seed", type=int, default=42)
    p_dis.add_argument(
        "--passport-reference", choices=("stream", "background"), default="stream",
        help=(
            "Population the spectral passport describes: 'stream' (default, "
            "reproduces the published artifacts) or 'background' (opt-in, "
            "experimental: passport over non-peak rows only; changes the "
            "radar's sensitivity, re-benchmark before trusting it)"
        ),
    )
    p_dis.set_defaults(func=cmd_distill)

    p_rad = sub.add_parser("radar", help="Run Bures-Wasserstein OOD radar against a crystal")
    p_rad.add_argument("--crystal", "-c", required=True, help="Path to .hologram.npz file")
    p_rad.add_argument("--input", "-i", required=True, help="New batch CSV/NPY/NPZ path")
    p_rad.add_argument("--target-col", type=int, default=None, help="Target column index to strip if present")
    p_rad.add_argument(
        "--streaming", action="store_true",
        help="Delta crystals: continue first differences from the previous radar call",
    )
    p_rad.add_argument(
        "--mean-shift", action="store_true",
        help="Also score the combined BW + mean-shift radar (v3.13.0)",
    )
    p_rad.add_argument(
        "--panel", action="store_true",
        help="Also score the panel (BW + mean-shift + MMD + energy) radar (v3.14.3)",
    )
    p_rad.add_argument(
        "--include-deltas", action="store_true",
        help="Treat a legacy (pre-3.9.1) artifact as a delta crystal",
    )
    p_rad.set_defaults(func=cmd_radar)

    p_heal = sub.add_parser("heal", help="Surgically inject new Voronoi medoids on domain shift")
    p_heal.add_argument("--crystal", "-c", required=True, help="Path to .hologram.npz file")
    p_heal.add_argument("--input", "-i", required=True, help="Drifted stream CSV/NPY/NPZ path")
    p_heal.add_argument("--output", "-o", default=None,
                        help="Output .hologram.npz path (default: <crystal>_healed.hologram.npz "
                             "next to the input; the input crystal is never overwritten)")
    p_heal.add_argument("--target-col", type=int, default=None, help="Target column index (omit for label-free)")
    p_heal.add_argument("--n-inject", type=int, default=35)
    p_heal.add_argument("--max-crystal-size", type=int, default=None, help="Cap total crystal size by pruning low-weight background medoids")
    p_heal.add_argument("--force", action="store_true")
    p_heal.add_argument("--streaming", action="store_true", help="Delta crystals: continue first differences from the previous call")
    p_heal.add_argument(
        "--include-deltas", action="store_true",
        help="Treat a legacy (pre-3.9.1) artifact as a delta crystal",
    )
    p_heal.add_argument("--seed", type=int, default=42)
    p_heal.set_defaults(func=cmd_heal)

    p_mig = sub.add_parser("migrate", help="Re-save a legacy crystal with the current schema (v3.14.3)")
    p_mig.add_argument("--crystal", "-c", required=True, help="Legacy .hologram.npz path")
    p_mig.add_argument("--output", "-o", required=True, help="Migrated .hologram.npz path")
    p_mig.add_argument(
        "--include-deltas", action="store_true",
        help="Treat a legacy (pre-3.9.1) artifact as a delta crystal",
    )
    p_mig.set_defaults(func=cmd_migrate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

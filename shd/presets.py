"""Named deployment presets for :class:`SpectralHologramDistiller`.

The published defaults (``high_band_ratio=0.26``, ``tail_quantile=84``,
``phase_kappa=3.2``) are the *protocol* constants: every §1–2 benchmark table
was generated with them, so they must not change silently. They are also
conservative, not per-dataset optimal -- that is measured, not asserted
(README "Constant sensitivity": bootstrap ablation puts
``high_band_ratio=0.50`` at 2.737 K critical RMSE vs 3.440 K for the shipped
defaults on the 4D reactor).

This module turns the measured operating points (README §6) into data, so a
caller never has to copy constructor arguments out of prose:

>>> from shd import apply_preset
>>> SpectralHologramDistiller(crystal_size=600, **apply_preset("critical_first"))

Presets never touch budget or reproducibility knobs (``crystal_size``,
``sketch_dim``, ``random_state``) -- the caller owns those.
"""

from __future__ import annotations

import inspect
from typing import Any

from .distiller import SpectralHologramDistiller

#: Measured operating points. Every value is backed by a committed artifact:
#: the frontier sweep (``bench_frontier_results.json``, 20 seeds, paired
#: Wilcoxon vs equal-budget K-Means), the PM2.5 rehab run
#: (``bench_pm25_rehab_results.json``) or the bootstrap ablation
#: (``bench_ablation_ci_summary.json``).
PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "params": {},
        "source": "constructor defaults; the protocol constants behind every "
                  "README section 1-2 table",
    },
    "global_first": {
        "params": {"high_band_ratio": 0.26, "tail_quantile": 84.0},
        "source": "README section 6 preset table; identical to the defaults, "
                  "spelled out for explicit configs",
        "explicit_defaults": True,
    },
    "pm25_global": {
        "params": {
            "global_guard": True,
        },
        "source": "bench_global_guard_results.json (20 seeds, shipped path: "
                  "save_hologram() with global_guard=True). Without the guard SHD "
                  "loses global regression on PM2.5 (KRR-all 101.15, R2 -0.15); "
                  "the guard (n_global=150, weight=0.35) fixes the global fit: "
                  "KRR-all 91.52 (20/20, p=1.9e-6), GBM-all 85.86 (20/20, "
                  "p=1.9e-6), R2 +0.06. HONEST LIMITS: it only TIES K-Means on "
                  "global KRR (91.52 vs 90.40, 8/20, p=0.14 n.s.) and LOSES on "
                  "global GBM (85.86 vs 77.38, 3/20, p=0.024); its win is the "
                  "CRITICAL regime (209.15 vs K-Means 239.15, 15/20, p=0.0014). "
                  "It also COSTS rare rows (26.5 -> 11.5 captured), because the "
                  "swapped rows are the low-weight ones. Use it when global fit "
                  "matters and the critical regime survives; without it, treat "
                  "PM2.5 as rare-episode monitoring only.",
    },
    "balanced": {
        "params": {"high_band_ratio": 0.50},
        "source": "README section 6: reactor crit -26.9% vs equal-budget "
                  "K-Means (20/20), global fit within +5.5% (n.s.); "
                  "tail_quantile stays at its default 84",
    },
    "critical_first": {
        "params": {"high_band_ratio": 0.65},
        "source": "README section 6: reactor crit -33.2% (20/20), "
                  "superconductivity KRR+GBM crit -20.9%/-40.0% (20/20); "
                  "the near-failure-accuracy operating point; "
                  "tail_quantile stays at its default 84",
    },
    "critical_rebuild": {
        "params": {"rebuild_at_finalize": True},
        "source": "2026-11-14 suppressor fix (bench_rebuild_at_finalize_results.json, "
                  "5 seeds): the default interval=0 online-only lattice cannot "
                  "create critical cells for post-warm-up rows. A single "
                  "finalize-time rebuild from the whole-stream reservoir "
                  "recovers critical accuracy: reactor crit 3.346 -> 2.495 "
                  "(-25%, beats the 2.78 oracle), SC 19.14 -> 17.585 (-8.1%, "
                  "matches the 17.59 oracle); rare capture 18 -> 35 (reactor), "
                  "64 -> 73 (SC). Cost: global +3-4%, fit time +30-85% (still "
                  "seconds). Opt-in; default interval=0 stays bit-for-bit.",
    },
    "sparse_stream_top3pct": {
        "params": {"tail_quantile": 97.0},
        "source": "README section 6 / bench_pm25_rehab: on a top-3% critical "
                  "regime, -11.9% KRR crit (18/20, p=2.1e-4) vs the default "
                  "quantile, at the cost of global fit; high_band_ratio "
                  "stays at its default 0.26",
    },
    "sparse_auto": {
        "params": {
            "tail_quantile": 97.0,
            "global_guard": True,
            "high_band_ratio": 0.26,
        },
        "explicit_defaults": True,
        "source": "Auto-engine for sparse critical regimes (crit_frac <= 5%): "
                  "tail_quantile=97 + global_guard. Measured on the shipped path "
                  "(bench_global_guard_results.json, 20 seeds): global R2 goes "
                  "from -0.15 (no guard) to +0.06 (with guard), KRR-all 91.52 vs "
                  "101.15 (20/20, p=1.9e-6), critical RMSE 209.15 vs 212.36 "
                  "(unchanged, 10/20, p=0.50 n.s.). COST: rare rows drop from "
                  "26.5 to 11.5 captured, and global GBM still trails K-Means "
                  "(85.86 vs 77.38). Opt-in: use apply_preset('sparse_auto') "
                  "explicitly. bit-for-bit: defaults untouched.",
    },
    "critical_first_no_residual": {
        "params": {
            "residual_weight": 0.0,
            "high_band_ratio": 0.65,
        },
        "source": "Residual-trigger tradeoff (bench_residual_tradeoff "
                  "10 seeds, Reactor+SC): rw_0 crit 3.316 vs default 3.440 "
                  "(Reactor 9/10, p=0.0098) and 17.94 vs 19.17 (SC 10/10, "
                  "p=0.0020); rw_0.5 on SC worse 19.42 (3/10). residual_"
                  "weight=0.0 keeps the y_z trigger only. cf_nores crit "
                  "2.32 (Reactor) / 16.30 (SC), 10/10 both. Use when crit "
                  "utility dominates global utility.",
    },
    "single_scale_low": {
        "params": {"rff_mode": "single_low"},
        "source": "Dual-scale RFF ablation (bench_rff): single_low 3.21 vs "
                  "dual 3.44 React crit (7/10, p=0.037) - single_low wins "
                  "on this reactor; dual is design choice, not theorem. "
                  "Kept as opt-in projection. rff_mode='dual' default untouched.",
    },
    "single_scale_high": {
        "params": {"rff_mode": "single_high"},
        "source": "bench_rff: single_high 3.28 vs dual 3.44 React crit. "
                  "Opt-in scale probe; rff_mode='dual' default untouched.",
    },
}

#: Caller-owned knobs a preset must never set.
_RESERVED = ("crystal_size", "sketch_dim", "random_state")


def apply_preset(name: str, **overrides: Any) -> dict[str, Any]:
    """Return constructor kwargs for preset ``name`` with caller overrides.

    Raises ``KeyError`` with the available names for an unknown preset.
    """
    try:
        preset = PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; available: {sorted(PRESETS)}"
        ) from None
    params = dict(preset["params"])
    unknown = set(overrides) & set(_RESERVED)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are caller-owned budget/reproducibility "
            "arguments; pass them to the constructor directly"
        )
    params.update(overrides)
    return params


def validate_presets() -> None:
    """Fail loudly when a preset drifts from the constructor it must feed.

    Called by the test-suite: every preset must be a subset of the
    ``SpectralHologramDistiller`` signature with non-default values (except
    the intentionally-default presets) and must actually build a distiller.
    """
    sig = inspect.signature(SpectralHologramDistiller.__init__)
    accepted = set(sig.parameters) - {"self"}
    for name, preset in PRESETS.items():
        params = preset["params"]
        unknown = set(params) - accepted
        if unknown:
            raise AssertionError(
                f"preset {name!r} uses unknown arguments: {sorted(unknown)}"
            )
        reserved = set(params) & set(_RESERVED)
        if reserved:
            raise AssertionError(
                f"preset {name!r} sets caller-owned arguments: {sorted(reserved)}"
            )
        defaults = {
            k: p.default for k, p in sig.parameters.items()
            if k in params and p.default is not inspect.Parameter.empty
        }
        if not preset.get("explicit_defaults"):
            stale = {k: v for k, v in defaults.items() if defaults[k] == params[k]}
            if stale:
                raise AssertionError(
                    f"preset {name!r} redundantly restates constructor "
                    f"defaults: {stale}; drop the keys, flag it explicitly "
                    "with 'explicit_defaults', or change the defaults "
                    "on purpose"
                )
        SpectralHologramDistiller(crystal_size=32, **params)  # must construct

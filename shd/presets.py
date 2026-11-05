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
        "source": "README §10 Known limitations / bench_pm25_rehab: without "
                  "the guard SHD loses global regression to K-Means (+6.9% "
                  "KRR-all p=4.8e-4); the guard (default n_global=150, weight=0.35) "
                  "swaps the 150 lowest-weight rows for K-Means global medoids "
                  "and wins BOTH all/crit 5/5 without hurting crit. PM2.5 global "
                  "without guard = negative result; ship the guard or call it "
                  "rare-episode monitoring.",
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
                  "tail_quantile=97 + global_guard. bench_pm25_rehab: "
                  "tq97+guard recovers global R2 to +0.04 KRR-all while "
                  "keeping crit ~205 K (vs default global R2 -0.13). "
                  "Opt-in: pass y with crit_frac<=0.05 detection, or use "
                  "apply_preset('sparse_auto') explicitly. bit-for-bit: "
                  "defaults untouched.",
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

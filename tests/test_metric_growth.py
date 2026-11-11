"""2026-11-11 metric-growth round: optimizations that raise SHD's real metrics.

Each test pins a verified improvement, not a tuned number. The published
default path (lattice_rebuild_interval=0) stays bit-for-bit identical; these
fixes act on the non-default rebuild path or on benchmark configuration.
"""

import os
import sys

import numpy as np

from shd import SpectralHologramDistiller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_finalize_resets_rebuild_counter_when_stream_ends_mid_interval():
    # With rebuilds enabled and rows absorbed since the last rebuild, finalize
    # must rebuild from the reservoir (which resets the counter). Before the fix
    # finalize committed the prefix-only lattice and left the counter > 0.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4))
    y = X.sum(axis=1)
    d = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, random_state=0,
        lattice_rebuild_interval=1000, max_init_buffer=1500)
    for i in range(0, 3000, 250):
        d.partial_fit(X[i:i + 250], y[i:i + 250])
    assert d._rows_since_rebuild > 0
    d.finalize_crystal()
    assert d._rows_since_rebuild == 0


def test_finalize_captures_late_regime_via_reservoir_rebuild():
    # Background first, a hot regime LAST, and no rebuild during streaming
    # (interval huge). The reservoir's peak stratum holds the late hot rows, so
    # the finalize-time rebuild must give them medoids. Before the fix the
    # crystal was the background warm-up lattice and captured zero hot rows.
    rng = np.random.default_rng(1)
    X_bg = rng.normal(size=(1000, 4))
    y_bg = rng.normal(size=1000) * 0.1
    X_hot = rng.normal(size=(500, 4)) + 4.0
    y_hot = rng.normal(size=500) + 12.0
    X = np.vstack([X_bg, X_hot])
    y = np.concatenate([y_bg, y_hot])
    d = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, random_state=0,
        lattice_rebuild_interval=10000, max_init_buffer=1000)
    for i in range(0, len(X), 250):
        d.partial_fit(X[i:i + 250], y[i:i + 250])
    d.finalize_crystal()
    rare = int(np.sum(d.crystal_Y >= 6.0))
    assert rare > 0


def test_default_interval_zero_does_not_rebuild_at_finalize():
    # Parity guard: the default interval=0 path never collects a reservoir and
    # must not rebuild at finalize (the counter stays as the online count).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4))
    y = X.sum(axis=1)
    d = SpectralHologramDistiller(
        crystal_size=40, sketch_dim=32, random_state=0,
        lattice_rebuild_interval=0, max_init_buffer=1500)
    for i in range(0, 3000, 250):
        d.partial_fit(X[i:i + 250], y[i:i + 250])
    before = d._rows_since_rebuild
    d.finalize_crystal()
    assert d._rows_since_rebuild == before


def test_bank_accepts_if_any_domain_accepts_and_routes_by_margin(tmp_path):
    # Mixed-rank bank: domain 1's crystal is rank-128 (wider threshold), domain
    # 0 is rank-32. A domain-1 window must be accepted (domain 1 accepts it) and
    # routed to domain 1 — not flagged as a shift because the low-rank crystal
    # happens to have a smaller raw BW distance.
    from shd import DomainBank
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(2000, 4))
    X1 = rng.normal(size=(2000, 4)) + 0.25
    paths = []
    for i, (X, rank) in enumerate([(X0, 32), (X1, 128)]):
        d = SpectralHologramDistiller(crystal_size=60, sketch_dim=128, random_state=i)
        for j in range(0, len(X), 500):
            d.partial_fit(X[j:j + 500], X[j:j + 500].sum(axis=1))
        d.finalize_crystal()
        p = str(tmp_path / f"d{i}.hologram.npz")
        d.save_hologram(p, rank_passport=rank, include_projector=True)
        paths.append(p)
    bank = DomainBank(paths)
    rep = bank.check_domain_shift(X1[:200])
    assert rep["domain_index"] == 1
    assert rep["is_shifted"] is False
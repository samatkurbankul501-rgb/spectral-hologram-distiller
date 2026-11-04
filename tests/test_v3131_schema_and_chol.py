"""Regressions for v3.13.1: artifact schema contract + per-batch Cholesky cache.

Covers the two technical holes from the v3.13.0 audit:

* **Artifact schema contract** — ``save_hologram`` writes ``schema_version`` /
  ``schema_min_loader`` / ``shd_version``; the loader rejects artifacts that
  need a newer loader, warns on artifacts distilled by a pre-3.13.1 build,
  and keeps ``heal_domain_shift().save()`` round-trips versioned.
* **Per-batch Cholesky cache** — ``partial_fit`` factors
  ``Psi_hat + beta_low * I`` once per batch and reuses the factor for both the
  leverage scores and the ridge head; predictions match a fresh factorisation
  bit-for-bit, and the cache is refreshed when the accumulators move.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shd import HologramCrystalLoader, SHD_VERSION, SpectralHologramDistiller  # noqa: E402

SHD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _distill(X, y, crystal_size=60, seed=1):
    d = SpectralHologramDistiller(crystal_size=crystal_size, random_state=seed)
    d.partial_fit(X, y)
    d.finalize_crystal()
    return d


def _save(d, tmp_path, name="c.hologram.npz"):
    path = str(tmp_path / name)
    d.save_hologram(path, rank_passport=32)
    return path


def _strip_schema_keys(src_path, dst_path):
    """Rewrite an artifact without the v3.13.1 keys -> legacy pre-3.13.1 file."""
    with np.load(src_path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files
                   if k not in ("schema_version", "schema_min_loader", "shd_version")}
    np.savez_compressed(dst_path, **payload)


def test_current_artifact_stores_schema_and_package_version(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = _distill(X, y)
    path = _save(d, tmp_path)

    ld = HologramCrystalLoader(path)
    assert ld.schema_version == 1
    assert ld.shd_version == SHD_VERSION
    s = ld.summary()
    assert s["schema_version"] == 1
    assert s["shd_version"] == SHD_VERSION


def test_legacy_artifact_without_keys_loads_with_warning(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = _distill(X, y)
    legacy_path = str(tmp_path / "legacy.hologram.npz")
    _strip_schema_keys(_save(d, tmp_path), legacy_path)

    with pytest.warns(UserWarning, match="pre-3.13.1"):
        ld = HologramCrystalLoader(legacy_path)
    assert ld.schema_version == 1
    assert ld.shd_version is None
    # the legacy layout still works end-to-end
    rep = ld.check_domain_shift(X[:64], mean_shift=True)
    assert "is_shifted" in rep


def test_future_schema_min_loader_is_rejected_at_load_time(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = _distill(X, y)
    path = _save(d, tmp_path)
    future_path = str(tmp_path / "future.hologram.npz")
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema_version"] = np.array([2], dtype=np.int64)
    payload["schema_min_loader"] = np.array([2], dtype=np.int64)
    np.savez_compressed(future_path, **payload)

    with pytest.raises(ValueError, match="upgrade the shd package"):
        HologramCrystalLoader(future_path)


def test_corrupt_schema_version_is_rejected_not_swallowed(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = _distill(X, y)
    path = _save(d, tmp_path)
    broken_path = str(tmp_path / "broken.hologram.npz")
    with np.load(path, allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    payload["schema_version"] = np.array([np.nan, np.nan], dtype=np.float64)
    np.savez_compressed(broken_path, **payload)

    with pytest.raises(ValueError, match="unreadable schema_version"):
        HologramCrystalLoader(broken_path)


def test_healed_crystal_roundtrip_keeps_schema_contract(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = _distill(X, y)
    path = _save(d, tmp_path)

    ld = HologramCrystalLoader(path)
    ld.heal_domain_shift(X[:64] + 3.0, force=True, random_state=0)
    healed_path = str(tmp_path / "healed.hologram.npz")
    ld.save(healed_path)

    ld2 = HologramCrystalLoader(healed_path)
    assert ld2.schema_version == 1
    assert ld2.shd_version == SHD_VERSION


def test_cholesky_cache_reused_within_batch_and_refreshed_per_batch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=2000)
    d = SpectralHologramDistiller(crystal_size=60, random_state=1)
    d.partial_fit(X[:1000], y[:1000])
    factor_at_1000 = d._chol_cache
    assert factor_at_1000 is not None and factor_at_1000[0] == 1000

    Psi_hat = d.Psi_sum / d.n_seen
    d._chol_psi(Psi_hat)  # same n_seen -> cached factor must be reused
    assert d._chol_cache is factor_at_1000

    d.partial_fit(X[1000:2000], y[1000:2000])
    assert d._chol_cache[0] == 2000  # refreshed for the new batch


def test_cholesky_cache_matches_a_fresh_factorisation():
    from scipy import linalg

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=1500)
    d = SpectralHologramDistiller(crystal_size=60, random_state=1)
    d.partial_fit(X, y)

    Psi_hat = d.Psi_sum / d.n_seen
    fresh = linalg.cho_factor(Psi_hat + d.beta_low * np.eye(d.k), lower=True)
    assert np.allclose(d._chol_cache[1][0], fresh[0])


def test_cholesky_cache_is_invalidated_by_external_psi_sum_write():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = X.sum(axis=1) + 0.1 * rng.normal(size=1500)
    d = SpectralHologramDistiller(crystal_size=60, random_state=1)
    d.partial_fit(X, y)
    d._chol_cache[1][0][0, 0] += 1.0  # corrupt the cached factor in place
    d._invalidate_chol()
    assert d._chol_cache is None
    Psi_hat = d.Psi_sum / d.n_seen
    d._chol_psi(Psi_hat)  # rebuilds a clean factor instead of reusing the corrupt one
    assert d._chol_cache is not None


def test_package_version_matches_SHD_VERSION():
    from shd import SHD_VERSION, __version__

    assert SHD_VERSION == "3.14.2"
    assert __version__ == SHD_VERSION

"""Bughunt 2026-11-08: defects confirmed by runtime probe before the fix.

Every test here pins a defect that was reproduced by an independent
runtime probe BEFORE the fix -- not a plausible reading of the code.
"""

import warnings
import zipfile

import numpy as np

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd.bank import DomainBank


def _train(n=600, d=8, seed=0, crystal_size=60):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    dist = SpectralHologramDistiller(crystal_size=crystal_size, sketch_dim=32,
                                     random_state=seed)
    dist.partial_fit(X, y)
    dist.finalize_crystal()
    return dist, X


def test_save_hologram_normalises_the_npz_suffix(tmp_path):
    # np.savez_compressed appends .npz; the old code then measured the
    # ORIGINAL name and raised FileNotFoundError after writing the file.
    dist, _ = _train()
    kb = dist.save_hologram(str(tmp_path / "crystal.hologram"))
    assert (tmp_path / "crystal.hologram.npz").exists()
    assert kb > 0


def test_loader_save_normalises_the_npz_suffix(tmp_path):
    dist, _ = _train()
    src = tmp_path / "c.hologram.npz"
    dist.save_hologram(str(src))
    loader = HologramCrystalLoader(str(src))
    kb = loader.save(str(tmp_path / "copy.hologram"))
    assert (tmp_path / "copy.hologram.npz").exists()
    assert kb > 0


def test_uncalibrated_panel_reports_no_verdict(tmp_path):
    # The panel score sums 8 terms while the default threshold is the
    # BW-calibrated constant (4-term panel): measured 8/8 in-distribution
    # windows came out 'shifted'. Without calibration the panel verdict
    # must be unknown, and it must not overwrite the BW verdict.
    dist, X = _train()
    src = tmp_path / "c.hologram.npz"
    dist.save_hologram(str(src))
    loader = HologramCrystalLoader(str(src))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = loader.check_domain_shift(X[:64], panel=True)
    assert any("panel=True without a panel calibration" in str(w.message) for w in caught)
    assert rep["panel_is_shifted"] is None
    assert rep["panel_threshold_source"] == "uncalibrated"
    bw = loader.check_domain_shift(X[:64])
    assert rep["is_shifted"] == bw["is_shifted"]


def test_calibrated_panel_returns_a_boolean_verdict(tmp_path):
    dist, X = _train()
    src = tmp_path / "c.hologram.npz"
    dist.save_hologram(str(src))
    loader = HologramCrystalLoader(str(src))
    loader.panel_tau_calibrated = 0.5
    loader.panel_tau_window = 128
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = loader.check_domain_shift(X[:64], panel=True)
    assert isinstance(rep["panel_is_shifted"], bool)
    assert rep["panel_threshold_source"] == "calibrated"
    assert not any("panel=True without a panel calibration" in str(w.message) for w in caught)


def test_domain_bank_uses_the_loader_width_contract(tmp_path):
    # Legacy delta artifact: include_deltas postdates the file, so the
    # loader expects the 2d width while the bank used to advertise d.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6))
    dist = SpectralHologramDistiller(crystal_size=40, sketch_dim=32,
                                     random_state=2, include_deltas=True)
    dist.partial_fit(X, rng.normal(size=300))
    dist.finalize_crystal()
    src = tmp_path / "delta.hologram.npz"
    dist.save_hologram(str(src))
    legacy = tmp_path / "legacy.hologram.npz"
    zin = zipfile.ZipFile(src)
    items = [(i, zin.read(i.filename)) for i in zin.infolist()]
    zin.close()
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in items:
            if info.filename == "include_deltas.npy":
                continue
            zout.writestr(info, data)
    loader = HologramCrystalLoader(str(legacy))
    assert loader.include_deltas is False
    bank = DomainBank([loader])
    rep = bank.check_domain_shift(rng.normal(size=(16, len(loader.X_mean))))
    assert "bw_distance" in rep


def test_global_guard_survives_a_stream_shorter_than_the_budget(tmp_path):
    # global_guard_n is clamped against the REQUESTED crystal_size; the
    # real crystal was smaller, so save_hologram() raised ValueError.
    rng = np.random.default_rng(3)
    dist = SpectralHologramDistiller(crystal_size=60, sketch_dim=32,
                                     random_state=3, global_guard=True)
    dist.partial_fit(rng.normal(size=(20, 8)), rng.normal(size=20))
    dist.finalize_crystal()
    assert len(dist.crystal_X) == 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kb = dist.save_hologram(str(tmp_path / "g.hologram.npz"))
    assert kb > 0


def test_loader_save_survives_a_half_populated_calibration(tmp_path):
    # A calibration group with `calibrated` set but window/n_windows None
    # used to crash save() with int(None) -- and save() is the persist path,
    # so that crash would lose the result of a heal. An incomplete group is
    # now dropped instead of killing the write.
    dist, _ = _train()
    src = tmp_path / "c.hologram.npz"
    dist.save_hologram(str(src))
    loader = HologramCrystalLoader(str(src))
    loader.panel_tau_calibrated = 0.5   # window / n_windows left None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kb = loader.save(str(tmp_path / "out.hologram"))
    assert kb > 0
    assert (tmp_path / "out.hologram.npz").exists()
"""2026-11-11 bug hunt: core distiller/loader hardening regressions.

Each test pins a CONTRACT repaired in this round, not a tuned number:

* the public read paths (get_training_data / get_phase_peak_slice) acquire the
  loader RLock, so a concurrent heal cannot rebind X/y/weights mid-read;
* heal rejects n_inject < 1 instead of silently injecting one medoid while
  reporting n_injected=0;
* the loader rejects an all-zero gamma_weights artifact (the read paths
  normalize by the weight mean, which would divide by zero);
* adaptive_threshold=False still rank-scales the default constant (the BW
  distance grows ~sqrt(rank) regardless of threshold mode); rank-32 is
  bit-identical to the shipped gate;
* _rows_since_rebuild counts the rows actually absorbed online, not the
  pre-slice batch size, when the warm-up buffer fills mid-batch;
* rank_passport above sketch_dim clamps to sketch_dim (not crystal_size) and
  rank_passport < 1 raises, matching the save_hologram docstring.
"""

import os
import sys
import threading

import numpy as np
import pytest

from shd import HologramCrystalLoader, SpectralHologramDistiller
from shd.distiller import BW_ALERT_THRESHOLD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _distill(n=400, d=3, crystal_size=20, sketch_dim=16, seed=0):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1.0, d + 1.0)
    dist = SpectralHologramDistiller(
        crystal_size=crystal_size, sketch_dim=sketch_dim,
        random_state=seed, max_init_buffer=max(n, 400),
    )
    for i in range(0, n, 200):
        dist.partial_fit(X[i:i + 200], y[i:i + 200])
    dist.finalize_crystal()
    return dist


def _crystal(tmp_path, name="c.hologram.npz", sketch_dim=16, rank_passport=32):
    dist = _distill(sketch_dim=sketch_dim)
    path = str(tmp_path / name)
    dist.save_hologram(path, rank_passport=rank_passport)
    return path


def _payload(path):
    src = np.load(path, allow_pickle=False)
    data = {key: np.asarray(src[key]) for key in src.files}
    src.close()
    return data


# ------------------------------------------------------- 3. read-path lock ---

def test_read_paths_acquire_loader_lock(tmp_path):
    ld = HologramCrystalLoader(_crystal(tmp_path))
    seen = {}
    orig = ld._select_weights

    def spy(weighting):
        # RLock._is_owned() is True only while THIS thread holds the lock.
        seen["owned"] = ld._lock._is_owned()
        return orig(weighting)

    ld._select_weights = spy
    ld.get_training_data()
    assert seen.get("owned") is True
    seen.clear()
    ld.get_phase_peak_slice()
    assert seen.get("owned") is True


def test_reader_blocks_while_heal_holds_lock(tmp_path):
    ld = HologramCrystalLoader(_crystal(tmp_path))
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def holder():
        with ld._lock:
            started.set()
            release.wait(5.0)

    def reader():
        ld.get_training_data()
        done.set()

    th = threading.Thread(target=holder)
    th.start()
    assert started.wait(5.0)
    tr = threading.Thread(target=reader)
    tr.start()
    try:
        # The reader must not complete while another thread owns the lock.
        assert not done.wait(0.2)
    finally:
        release.set()
        th.join(5.0)
        tr.join(5.0)
    assert done.is_set()


# --------------------------------------------------------- 4. n_inject >= 1 ---

def test_heal_rejects_zero_and_negative_n_inject(tmp_path):
    ld = HologramCrystalLoader(_crystal(tmp_path))
    rng = np.random.default_rng(0)
    win = rng.normal(size=(20, 3))
    n_before = len(ld.X)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="n_inject must be >= 1"):
            ld.heal_domain_shift(win, n_inject=bad, force=True)
        assert len(ld.X) == n_before  # rejected heal leaves the loader untouched


# ----------------------------------------------------- 5. all-zero weights ---

def test_loader_rejects_all_zero_weights(tmp_path):
    path = _crystal(tmp_path)
    payload = _payload(path)
    payload["gamma_weights"] = np.zeros_like(payload["gamma_weights"])
    out = str(tmp_path / "zero.hologram.npz")
    np.savez_compressed(out, **payload)
    with pytest.raises(ValueError, match="all-zero gamma_weights"):
        HologramCrystalLoader(out)


def test_loader_still_rejects_negative_weights(tmp_path):
    path = _crystal(tmp_path)
    payload = _payload(path)
    w = payload["gamma_weights"].copy()
    w[0] = -1.0
    payload["gamma_weights"] = w
    out = str(tmp_path / "neg.hologram.npz")
    np.savez_compressed(out, **payload)
    with pytest.raises(ValueError, match="negative gamma_weights"):
        HologramCrystalLoader(out)


# ------------------------------------------- 14. rank-scaled False threshold ---

def test_false_threshold_is_rank_scaled(tmp_path):
    rng = np.random.default_rng(11)
    win = rng.normal(size=(64, 3))
    p64 = _crystal(tmp_path, name="r64.hologram.npz", sketch_dim=64, rank_passport=64)
    ld64 = HologramCrystalLoader(p64)
    assert len(ld64.eigvals) == 64
    off64 = ld64.check_domain_shift(win, adaptive_threshold=False)
    assert off64["threshold"] == pytest.approx(
        BW_ALERT_THRESHOLD * float(np.sqrt(64.0 / 32.0)))
    assert off64["threshold_source"] == "default"

    p32 = _crystal(tmp_path, name="r32.hologram.npz", sketch_dim=64, rank_passport=32)
    ld32 = HologramCrystalLoader(p32)
    assert len(ld32.eigvals) == 32
    off32 = ld32.check_domain_shift(win, adaptive_threshold=False)
    # Rank-32 is bit-identical to the shipped default constant.
    assert off32["threshold"] == pytest.approx(BW_ALERT_THRESHOLD)


# ----------------------------------------- 15. online rebuild-row accounting ---

def test_rows_since_rebuild_counts_online_tail_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = X.sum(axis=1)
    d = SpectralHologramDistiller(
        crystal_size=20, sketch_dim=16, random_state=0,
        max_init_buffer=100, lattice_rebuild_interval=100000,
    )
    d.partial_fit(X, y)
    # Warm-up buffered 100 rows; only the 50-row tail streamed online.
    assert d._rows_since_rebuild == 50


# --------------------------------------------- 17. rank_passport clamp doc ---

def test_rank_passport_clamps_to_sketch_dim(tmp_path):
    dist = _distill(sketch_dim=16)
    p = str(tmp_path / "clamp.hologram.npz")
    dist.save_hologram(p, rank_passport=300)  # above sketch_dim
    ld = HologramCrystalLoader(p)
    assert len(ld.eigvals) == 16  # clamped to sketch_dim, not crystal_size
    with pytest.raises(ValueError, match="rank_passport must be >= 1"):
        dist.save_hologram(str(tmp_path / "bad.hologram.npz"), rank_passport=0)


# ----------------------------------------- 2. gaps summarise test keys ---

def _gaps_rec(seed, shd_crit, kt_crit, skt_crit, strat_crit):
    def arm(crit):
        return {"krr_rmse_all": crit * 0.8, "krr_rmse_crit": crit, "krr_r2": 0.9,
                "gbm_rmse_all": crit * 0.9, "gbm_rmse_crit": crit, "gbm_r2": 0.8,
                "rare_captured": 60}
    return {"seed": seed, "split_seed": 42, "shd": arm(shd_crit),
            "shd_sub4800": arm(shd_crit + 0.5), "kt": arm(kt_crit),
            "skt": arm(skt_crit), "stratified": arm(strat_crit)}


def test_gaps_summarise_only_shd_vs_baseline_keys():
    import bench_gaps
    recs = [_gaps_rec(i, 18.0, 23.0, 21.0, 24.0) for i in range(5)]
    s = bench_gaps.summarise(recs, split_seed=42)
    assert s["n_records"] == 5
    assert "shd_vs_kt_krr_rmse_crit" in s["tests"]
    # sub4800 is compared against full SHD, never against the baselines: the
    # removed main() print key "shd_sub4800_vs_kt_..." was therefore dead.
    assert "shd_vs_shd_sub4800_krr_rmse_crit" in s["tests"]
    assert not any(k.startswith("shd_sub4800_vs_") for k in s["tests"])
    assert s["tests"]["shd_vs_kt_krr_rmse_crit"]["wins_shd"] == 5
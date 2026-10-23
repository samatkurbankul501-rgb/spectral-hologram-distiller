"""Live smoke for the v3.13.0 guard + combined radar integration.

Builds a small crystal on the reactor-style stream, runs build_global_guard
with an explicit pool and the mean-shift radar path, and prints every report
field so nothing is eyeballed.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from shd import HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402


def make_stream(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 4))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    return X, y


def main():
    X, y = make_stream()
    d = SpectralHologramDistiller(
        crystal_size=80, high_band_ratio=0.26, sketch_dim=64,
        tail_quantile=84.0, random_state=3,
    )
    B = 1500
    for i in range(0, len(X), B):
        d.partial_fit(X[i:i + B], y[i:i + B])
    d.finalize_crystal()

    w_before = d.gamma_weights.copy()
    X_before = d.crystal_X.copy()
    rep = d.build_global_guard(n_global=20, weight=0.35, X_pool=X, y_pool=y,
                               random_state=11)
    print("guard report:", rep)
    assert rep["swapped"] == 20
    assert rep["n_pool"] == len(X)
    assert np.isfinite(d.gamma_weights).all() and (d.gamma_weights > 0).all()
    assert not np.allclose(w_before, d.gamma_weights)
    n_changed = int(np.sum(np.any(X_before != d.crystal_X, axis=1)))
    print("rows changed:", n_changed)
    assert n_changed >= 15  # most swapped slots must actually differ
    # crystal_U must match the swapped rows' sketches
    _Z, U_chk = d._spectral_transform(d.crystal_X[:5], update_state=False)
    assert np.allclose(U_chk, d.crystal_U[:5], atol=1e-9)
    print("crystal_U consistent: OK")

    # residual stats idempotent + explicit pool
    st = d.fit_residual_stats(X, y)
    print("resid stats:", st)
    assert st["n_pool"] == len(X)
    st2 = d.fit_residual_stats(X[:100], y[:100])
    assert st2["n_pool"] == 100

    # save/load roundtrip with guard applied
    d.save_hologram("tools/_guard_smoke.hologram.npz")
    loader = HologramCrystalLoader("tools/_guard_smoke.hologram.npz")
    print("loader loaded:", loader.X.shape)

    # mean-shift radar: healthy window vs shifted window
    rng = np.random.default_rng(5)
    Xw = rng.normal(size=(64, 4))
    rep0 = loader.check_domain_shift(Xw, mean_shift=True)
    print("healthy:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in rep0.items()})
    Xw2 = Xw + 3.0
    rep1 = loader.check_domain_shift(Xw2, mean_shift=True)
    print("shifted:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in rep1.items()})
    assert rep1["mean_shift_distance"] > rep0["mean_shift_distance"]
    assert rep1["combined_distance"] > rep0["combined_distance"]
    assert rep1["is_shifted"] is True
    # legacy path unchanged
    rep_legacy = loader.check_domain_shift(Xw)
    assert "mean_shift_distance" not in rep_legacy
    assert rep_legacy["is_shifted"] == (rep_legacy["bw_distance"] > rep_legacy["threshold"])

    # calibrate_threshold_ms on healthy reference + roundtrip persistence
    Xref = rng.normal(size=(2000, 4))
    crep = loader.calibrate_threshold_ms(Xref, window=64, fa_rate=0.05)
    print("ms calib:", crep)
    assert crep["n_windows"] == 31
    kb = loader.save("tools/_guard_smoke.hologram.npz")
    loader2 = HologramCrystalLoader("tools/_guard_smoke.hologram.npz")
    assert loader2.ms_tau_calibrated == crep["tau"]
    rep_ms2 = loader2.check_domain_shift(Xw, mean_shift=True)
    assert rep_ms2["combined_threshold_source"] == "calibrated"
    print("ms tau persisted:", loader2.ms_tau_calibrated)
    # heal invalidates the ms calibration together with the BW one
    loader2.heal_domain_shift(Xw2, force=True, random_state=0)
    assert loader2.ms_tau_calibrated is None and loader2.tau_calibrated is None
    print("heal invalidated both calibrations: OK")
    print("ALL GUARD/RADAR SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()

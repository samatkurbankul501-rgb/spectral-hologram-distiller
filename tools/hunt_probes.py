"""Live reproduction of the v3.11.0 defects found by the 2026-10-21 review.

Each probe REPRODUCES a defect on the v3.11.0 tree (or demonstrates that a
suspicion was wrong). Run it from anywhere:

    python tools/hunt_probes.py

Exit code 0 means every probe behaved as expected on the repaired tree.
These are the scripts that produced the numbers quoted in the v3.11.1 repair
notes, so the findings stay reproducible instead of being folklore.
"""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shd.distiller import SpectralHologramDistiller, HologramCrystalLoader  # noqa: E402

rng = np.random.default_rng(7)
tmp = tempfile.mkdtemp()
report = []


def sec(name):
    print("\n" + "=" * 78)
    print("PROBE:", name)
    print("=" * 78)


def stream(n, d, crit_frac=0.03, seed_batch=500):
    for i in range(0, n, seed_batch):
        B = min(seed_batch, n - i)
        X = rng.uniform(-2, 2, size=(B, d))
        y = 1.6 * np.sin(1.1 * X[:, 0]) * np.cos(X[:, 1]) + 0.5 * X[:, 2] ** 2 - 0.8 * X[:, 3]
        mask = rng.random(B) < crit_frac
        if mask.any():
            y[mask] += 8.0 * np.exp(-np.sum((X[mask] - 0.5) ** 2, axis=1))
        yield X, y


# ------------------------------------------------- P1: anytime budget keeps the band
def probe_p1():
    sec("P1: get_training_data(budget) must keep the phase-peak band")
    d = SpectralHologramDistiller(crystal_size=200, high_band_ratio=0.30, max_init_buffer=4096)
    for Xb, yb in stream(12000, 4):
        d.partial_fit(Xb, yb)
    d.finalize_crystal()
    p = os.path.join(tmp, "p1.hologram.npz")
    d.save_hologram(p, include_projector=True)
    l = HologramCrystalLoader(p)
    peak_idx = set(l._peak_rank_idx(len(l.y), l.high_band_ratio).tolist())
    n_in_first100 = len(peak_idx & set(range(100)))
    print(f"crystal M={len(l.y)}, peak band={len(peak_idx)} rows (tail={l.tail})")
    print(f"peak-band rows inside the first 100 stored rows: {n_in_first100}")
    X100, y100, w100 = l.get_training_data(budget_points=100)
    print(f"max y in budget slice: {y100.max():.2f} | max y in full crystal: {l.y.max():.2f}")
    print(f"weights mean after renormalisation: {w100.mean():.4f}")
    # The contract: a budget of 100 must carry the WHOLE 60-row peak band.
    peak_y = np.sort(l.y)[-len(peak_idx):]
    kept = sum(1 for v in peak_y if np.isclose(y100, v).any())
    print(f"peak-band rows kept by the budget slice: {kept}/{len(peak_idx)}")
    report.append(("P1 budget keeps the peak band", kept == len(peak_idx)))


# ------------------------------------------------- P2: slice follows the artifact
def probe_p2():
    sec("P2: get_phase_peak_slice must follow the persisted band ratio")
    l = HologramCrystalLoader(os.path.join(tmp, "p1.hologram.npz"))
    print(f"persisted high_band_ratio = {l.high_band_ratio}")
    X_def, _, _ = l.get_phase_peak_slice()
    X_own, _, _ = l.get_phase_peak_slice(high_band_ratio=l.high_band_ratio)
    print(f"default slice rows={len(X_def)} | explicit persisted-ratio rows={len(X_own)}")
    report.append(("P2 slice follows persisted ratio", len(X_def) == len(X_own)))


# ------------------------------------------------- P3: label-free heal stays honest
def probe_p3():
    sec("P3: label-free heal must not poison the target column")
    l = HologramCrystalLoader(os.path.join(tmp, "p1.hologram.npz"))
    X_ood = rng.uniform(4.0, 6.0, size=(400, 4))
    res = l.heal_domain_shift(X_ood, n_inject=10, force=True)
    print(f"healed={res['healed']}, injected={res['n_injected']}, "
          f"unlabelled={res.get('n_unlabelled_injected')}")
    print(f"warning: {res.get('warning')}")
    print(f"real y median before heal: {float(np.median(l.y[:-10])):.3f}")
    print(f"injected y values: {np.round(l.y[-10:], 3)}")
    n_train = len(l.get_training_data()[1])
    print(f"training rows with the default drop_unlabelled=True: {n_train} "
          f"(crystal holds {len(l.y)})")
    report.append(("P3 unlabelled rows are dropped by default", n_train == len(l.y) - 10))


# ------------------------------------------------- P4: float16 passport fidelity
def probe_p4():
    sec("P4: float16 passport vs exact float64 re-decomposition")
    l2 = HologramCrystalLoader(os.path.join(tmp, "p1.hologram.npz"))
    V16, e16 = l2.eigvecs, l2.eigvals
    orth_err = float(np.linalg.norm(V16.T @ V16 - np.eye(V16.shape[1]), ord="fro"))
    print(f"||V^T V - I||_F for float16-roundtripped eigvecs: {orth_err:.4f}")
    Psi_stored = (V16 * e16) @ V16.T
    ev, evec = np.linalg.eigh(0.5 * (Psi_stored + Psi_stored.T))
    Vx, ex = evec[:, -32:], ev[-32:]
    X_win = next(stream(2000, 4))[0]
    _, U_win = HologramCrystalLoader(os.path.join(tmp, "p1.hologram.npz"))._prepare_stream(X_win)
    Psi_new = (U_win.T @ U_win) / len(U_win)

    def bw(V, e):
        A = np.diag(np.maximum(e, 1e-7))
        B = V.T @ Psi_new @ V + 1e-7 * np.eye(len(e))
        sA = np.diag(np.sqrt(np.diag(A)))
        inner = sA @ B @ sA
        w = np.linalg.eigvalsh(0.5 * (inner + inner.T))
        return float(np.sqrt(max(0.0, np.trace(A) + np.trace(B)
                                 - 2 * np.sum(np.sqrt(np.maximum(w, 0))))))

    b16, bx = bw(V16, e16), bw(Vx, ex)
    print(f"BW(float16 passport) = {b16:.5f} | BW(exact re-decomp) = {bx:.5f} "
          f"| diff = {abs(b16 - bx):.2e}")
    print(f"alert threshold at B={len(U_win)}: "
          f"{0.085 * np.sqrt(1024 / len(U_win)):.3f}")
    report.append(("P4 float16 passport is faithful", abs(b16 - bx) < 1e-3))


# ------------------------------------------------- P5: critical_threshold vs tail
def probe_p5():
    sec("P5: critical_threshold + tail='low' must be rejected")
    try:
        SpectralHologramDistiller(crystal_size=60, high_band_ratio=0.10, tail="low",
                                  critical_threshold=1.0, max_init_buffer=4096)
    except ValueError as exc:
        print("rejected as designed:", exc)
        report.append(("P5 contradictory combination rejected", True))
    else:
        print("ACCEPTED -- the contradiction is back")
        report.append(("P5 contradictory combination rejected", False))


# ------------------------------------------------- P6: passport reference default
def probe_p6():
    sec("P6: passport reference default must reproduce the published radar")
    d = SpectralHologramDistiller(crystal_size=20)
    print(f"default passport_reference = {d.passport_reference!r}")
    d_bg = SpectralHologramDistiller(crystal_size=20, passport_reference="background")
    print(f"opt-in passport_reference = {d_bg.passport_reference!r}")
    report.append(("P6 default is stream", d.passport_reference == "stream"))


if __name__ == "__main__":
    probe_p1()
    probe_p2()
    probe_p3()
    probe_p4()
    probe_p5()
    probe_p6()
    print("\n" + "=" * 78)
    print("PROBE SUMMARY")
    failed = 0
    for k, v in report:
        print(f"  {k}: {v}")
        if not v:
            failed += 1
    print(f"\n{len(report) - failed}/{len(report)} probes behaved as expected")
    sys.exit(1 if failed else 0)

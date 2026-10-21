"""Adversarial probe suite for the SHD core (hunt for real defects).

Run:  python hunt_core.py
Every probe prints PASS/FAIL/WARN with the numbers behind the verdict.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")

from shd import DomainBank, HologramCrystalLoader, SpectralHologramDistiller  # noqa: E402

FINDINGS = []


def finding(sev, probe, msg):
    FINDINGS.append((sev, probe, msg))
    print(f"  [{sev}] {probe}: {msg}")


def ok(probe, msg):
    print(f"  [ ok ] {probe}: {msg}")


def section(name):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)


# --------------------------------------------------------------------------- data
def make_stream(n=4000, d=8, seed=0, crit_frac=0.18, outlier_scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (2.0 * X[:, 0] + 0.7 * X[:, 1] * X[:, 2]
         + np.sin(1.3 * X[:, 3]) + 0.05 * rng.normal(size=n))
    hot = rng.random(n) < crit_frac
    y[hot] += 8.0 + 2.0 * rng.normal(size=hot.sum())
    X[hot] += 0.9
    if outlier_scale != 1.0:
        X[0] *= outlier_scale
    return X, y


def build(X, y, **kw):
    d = SpectralHologramDistiller(crystal_size=kw.pop("M", 120), random_state=7, **kw)
    for i in range(0, len(X), 512):
        d.partial_fit(X[i:i + 512], y[i:i + 512])
    d.finalize_crystal()
    return d


# =========================================================== P1: float16 passport
section("P1  float16 spectral passport: does 16-bit storage move the BW distance?")
X, y = make_stream()
d = build(X, y)
tmp = tempfile.mkdtemp(prefix="hunt_")
path = os.path.join(tmp, "c.hologram.npz")
d.save_hologram(path)

ld = HologramCrystalLoader(path)
# float32 truncation of the stored projector is the other half of the story.
# Recompute the exact (float64) passport from the distiller for comparison.
from scipy import linalg as sla

Psi_hat = d.Psi_sum / d.n_seen
w64, V64 = sla.eigh(Psi_hat)
r = len(w64)
A64 = np.diag(np.maximum(w64[-r:], 1e-7))
V32_f16 = ld.eigvecs.astype(np.float64)

V_new = ld.eigvecs
orth_err = np.abs(V32_f16.T @ V32_f16 - np.eye(V32_f16.shape[1])).max()
ok("P1", f"float16 eigenvector orthogonality defect = {orth_err:.3e}")

for nrows in (64, 512, 4000):
    Xin = X[:nrows]
    # stored (float16 passport + float32 projector)
    rep = ld.check_domain_shift(Xin)
    # exact reference: float64 passport, float64 projector, same normalisation
    Z = np.clip((Xin - d.X_mean) / d.X_std, -8.0, 8.0)
    U = np.sqrt(2.0 / d.k) * np.cos(Z @ d.Omega + d.phase)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    Psi_new = (U.T @ U) / len(U)
    B = V64[:, -r:].T @ Psi_new @ V64[:, -r:] + 1e-7 * np.eye(r)
    B = 0.5 * (B + B.T)
    sA = np.diag(np.sqrt(np.maximum(w64[-r:], 1e-7)))
    inner = sA @ B @ sA
    inner = 0.5 * (inner + inner.T)
    ev = np.maximum(sla.eigvalsh(inner), 0.0)
    bw_ref = float(np.sqrt(max(0.0, np.trace(np.diag(np.maximum(w64[-r:], 1e-7)))
                               + np.trace(B) - 2.0 * np.sum(np.sqrt(ev)))))
    err = abs(rep["bw_distance"] - bw_ref)
    rel = err / max(bw_ref, 1e-12)
    tag = "OK" if rel < 0.02 else "DRIFT"
    print(f"  [ {tag}] P1 B={nrows:5d}: stored BW={rep['bw_distance']:.6f} "
          f"reference BW={bw_ref:.6f} abs.err={err:.2e} rel={rel:.2%}")
    if rel >= 0.02:
        finding("HIGH", "P1", f"float16 passport shifts BW by {rel:.2%} at B={nrows} "
                            f"(stored {rep['bw_distance']:.5f} vs exact {bw_ref:.5f})")

# decision flip: how close does the stored distance sit to the threshold?
thr = rep["threshold"]
margin = abs(rep["bw_distance"] - thr)
print(f"  [info] P1 threshold={thr:.5f} stored BW={rep['bw_distance']:.5f} "
      f"margin={margin:.5f}")
if margin < 0.02:
    finding("MED", "P1", f"in-domain BW sits {margin:.4f} from the alert threshold; "
                        f"float16 drift of this size would flip the decision")

# ======================================================= P2: save without finalize
section("P2  contract: save_hologram() before finalize_crystal()")
try:
    d2 = SpectralHologramDistiller(crystal_size=20)
    d2.partial_fit(X[:200], y[:200])
    d2.save_hologram(os.path.join(tmp, "unfinalized.hologram.npz"))
    finding("MED", "P2", "save_hologram() before finalize_crystal() silently succeeded")
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    if "NoneType" in msg:
        finding("LOW", "P2", f"save before finalize raises an opaque error -> {msg}")
    else:
        ok("P2", f"save before finalize raises a clear error -> {msg}")

# ================================================ P3: determinism + batch invariance
section("P3  determinism and batch-size invariance of the streaming crystal")
d_a = build(X, y)
d_b = build(X, y)
same = np.array_equal(d_a.crystal_X, d_b.crystal_X) and np.array_equal(d_a.gamma_weights, d_b.gamma_weights)
ok("P3", f"same seed -> identical crystal: {same}")

d_c = SpectralHologramDistiller(crystal_size=120, random_state=7)
for i in range(0, len(X), 250):
    d_c.partial_fit(X[i:i + 250], y[i:i + 250])
d_c.finalize_crystal()
inter = np.mean([np.isclose(np.sort(d_a.crystal_X[:, 0]), np.sort(d_c.crystal_X[:, 0]),
                            atol=1e-6).mean()])
ok("P3", f"batch 512 vs 250 -> first-coordinate agreement {inter:.3f} (stream order matters by design)")

# ==================================================== P4: heal cap contract fuzz
section("P4  heal_domain_shift: fuzz the max_crystal_size hard-cap contract")
ld2 = HologramCrystalLoader(path)
drift = X[:400] * 2.5 + 4.0
yd = y[:400] + 3.0
cap_violations = []
for cap in (None, 200, 140, 130, 121, 120, 119, 100, 60, 40, 20, 5, 1):
    l = HologramCrystalLoader(path)
    try:
        res = l.heal_domain_shift(drift, yd, n_inject=35, max_crystal_size=cap, random_state=1)
        if cap is not None and len(l.X) > cap:
            cap_violations.append((cap, len(l.X)))
    except ValueError:
        pass
    except Exception as e:
        finding("HIGH", "P4", f"cap={cap} raised an unexpected {type(e).__name__}: {e}")
if cap_violations:
    finding("HIGH", "P4", f"hard cap exceeded: {cap_violations}")
else:
    ok("P4", "hard cap never exceeded across caps 1..200 (ValueError raised where required)")

# cap that is *just* enough: does the caller learn what cap is needed?
l = HologramCrystalLoader(path)
try:
    l.heal_domain_shift(drift, yd, n_inject=35, max_crystal_size=130, random_state=1)
except ValueError as e:
    print(f"  [info] P4 cap=130 error text: {e}")
    if "use max_crystal_size >=" not in str(e):
        finding("LOW", "P4", "cap error message does not state the minimum viable cap")

# ============================================ P5: repeated heal / forgetting
section("P5  repeated heal: does the crystal keep growing past the cap?")
l = HologramCrystalLoader(path)
for i in range(6):
    l.heal_domain_shift(drift * (1 + 0.2 * i) + i, yd + i, n_inject=35, random_state=i)
ok("P5", f"after 6 heals crystal size = {len(l.X)} (no cap set -> growth is expected)")

# ================================================ P6: single-row / edge inputs
section("P6  edge inputs: single row, constant features, huge outliers")
for label, Xe, ye in [
    ("single row", X[:1], y[:1]),
    ("constant features", np.ones((300, 4)), np.linspace(0, 1, 300)),
    ("huge outlier in batch 1", make_stream(n=1500, outlier_scale=1e6)[0], make_stream(n=1500, outlier_scale=1e6)[1]),
]:
    try:
        dd = SpectralHologramDistiller(crystal_size=30, random_state=1)
        for i in range(0, len(Xe), 256):
            dd.partial_fit(Xe[i:i + 256], ye[i:i + 256])
        dd.finalize_crystal()
        p2 = os.path.join(tmp, "edge.hologram.npz")
        dd.save_hologram(p2)
        l2 = HologramCrystalLoader(p2)
        rep = l2.check_domain_shift(Xe[:64])
        ok("P6", f"{label}: crystal built, medoids={len(l2.X)}, radar BW={rep['bw_distance']:.4f}")
    except Exception as e:
        finding("MED", "P6", f"{label} -> {type(e).__name__}: {e}")

# ================================================ P7: medoid contract (real rows)
section("P7  contract: are crystal rows REAL observations?")
d7 = build(X, y)
hits = 0
for row in d7.crystal_X:
    dist = np.linalg.norm(X - row, axis=1).min()
    hits += 1 if dist < 1e-9 else 0
if hits == len(d7.crystal_X):
    ok("P7", f"all {hits} medoids are verbatim dataset rows")
else:
    finding("HIGH", "P7", f"only {hits}/{len(d7.crystal_X)} medoids are real dataset rows")

# ================================================ P8: phase-peak band occupancy
section("P8  contract: does the crystal actually reserve the phase-peak band?")
d8 = build(X, y, high_band_ratio=0.26, tail_quantile=84.0)
thr84 = np.percentile(y, 84.0)
frac_crit = float(np.mean(y >= thr84))
frac_in_crystal = float(np.mean(d8.crystal_Y >= thr84))
ok("P8", f"stream critical fraction={frac_crit:.3f}, crystal critical fraction={frac_in_crystal:.3f} "
        f"(target {d8.M_high}/{d8.M} = {d8.M_high / d8.M:.3f})")
if frac_in_crystal < 0.15:
    finding("HIGH", "P8", f"crystal holds only {frac_in_crystal:.1%} critical rows "
                         f"despite a {d8.M_high}/{d8.M} band reservation")

# ================================================ P9: DomainBank mixed widths
section("P9  DomainBank routing with a mixed-width bank")
l_a = HologramCrystalLoader(path)
l_b = HologramCrystalLoader(path)
bank = DomainBank([l_a, l_b])
rep = bank.check_domain_shift(X[:128])
ok("P9", f"bank of 2 identical crystals -> domain_index={rep['domain_index']}, "
        f"distances={['%.4f' % v if v is not None else None for v in rep['bw_distances']]}")

# ================================================ P10: in-domain radar distance
section("P10  in-distribution radar distance on the training stream itself")
rep_train = ld.check_domain_shift(X[:2000])
rep_hold = ld.check_domain_shift(X[2000:3000])
ok("P10", f"train rows BW={rep_train['bw_distance']:.5f} (thr {rep_train['threshold']:.5f}), "
          f"held-out BW={rep_hold['bw_distance']:.5f}")
if rep_hold["is_shifted"]:
    finding("HIGH", "P10", f"held-out IN-DOMAIN rows are flagged as shifted "
                          f"(BW={rep_hold['bw_distance']:.4f} > {rep_hold['threshold']:.4f})")

# ================================================ P11: y semantics after label-free heal
section("P11  label-free heal: what lands in crystal_Y?")
l11 = HologramCrystalLoader(path)
res = l11.heal_domain_shift(drift, None, n_inject=20, random_state=0, force=True)
ok("P11", f"unsupervised heal injected {res['n_injected']} medoids; "
          f"crystal_Y now mixes real targets and novelty scores "
          f"(range {l11.y.min():.3f}..{l11.y.max():.3f}, real y range {y.min():.3f}..{y.max():.3f})")

# ================================================ P12: get_training_data weighting
section("P12  get_training_data weighting / budget semantics")
Xc, yc, wc = l_a.get_training_data()
Xu, yu, wu = l_a.get_training_data(weighting="uniform")
ok("P12", f"gamma weights: mean={wc.mean():.4f} (should be 1.0), min={wc.min():.3f}, max={wc.max():.3f}")
if abs(wc.mean() - 1.0) > 1e-6:
    finding("MED", "P12", f"gamma weights are not mean-normalised (mean={wc.mean():.6f})")
if not np.allclose(wu, 1.0):
    finding("LOW", "P12", "uniform weights are not all ones")
sub_X, sub_y, sub_w = l_a.get_training_data(budget_points=30)
ok("P12", f"budget=30 -> {sub_X.shape}, weight mean={sub_w.mean():.4f}")

# ================================================ P13: loader save round-trip
section("P13  loader.save() round-trip fidelity")
l13 = HologramCrystalLoader(path)
p13 = os.path.join(tmp, "resaved.hologram.npz")
l13.save(p13)
l13b = HologramCrystalLoader(p13)
same = (np.allclose(l13.X, l13b.X) and np.allclose(l13.y, l13b.y)
        and np.allclose(l13.weights, l13b.weights)
        and np.allclose(l13.eigvals, l13b.eigvals, atol=1e-6))
ok("P13", f"round-trip preserves X/y/weights/passport: {same}")
if not same:
    finding("MED", "P13", "loader.save() -> load does not reproduce the crystal exactly")

# ================================================ P14: timestamp / artifact hygiene
section("P14  artifact hygiene: does save_hologram leak anything unexpected?")
with np.load(path, allow_pickle=False) as z:
    keys = sorted(z.files)
    sizes = {k: z[k].shape for k in keys}
ok("P14", f"stored arrays: {sizes}")

print("\n" + "=" * 72)
print(f"SUMMARY: {len(FINDINGS)} findings")
for sev, probe, msg in FINDINGS:
    print(f"  {sev:5s} {probe}: {msg}")
print("=" * 72)

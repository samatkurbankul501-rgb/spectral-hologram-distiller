"""Probe set 5: last contracts before the write-up."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="h5_")
rng = np.random.default_rng(41)
N, d = 6000, 12
X = rng.normal(size=(N, d))
y = 2.0 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.normal(size=N)
hot = rng.random(N) < 0.18
y[hot] += 8.0; X[hot] += 0.9

dist = SpectralHologramDistiller(crystal_size=150, random_state=7)
for i in range(0, N, 512):
    dist.partial_fit(X[i:i+512], y[i:i+512])
dist.finalize_crystal()
p = os.path.join(tmp, "c.npz"); dist.save_hologram(p)
ld = HologramCrystalLoader(p)

print("A) radar with a SINGLE row (B=1):")
for B in (1, 2, 4, 16, 64):
    r = ld.check_domain_shift(X[:B])
    print("   B=%3d: BW=%.5f thr=%.5f shifted=%s" % (B, r["bw_distance"], r["threshold"], r["is_shifted"]))
print("   => the adaptive threshold widens as 0.085*sqrt(1024/B); a 1-row window")
print("      needs 0.85 to be quiet, so single-row scoring is nearly useless.")

print("\nB) get_phase_peak_slice vs the crystal's own tail mode:")
Xp, yp, wp = ld.get_phase_peak_slice()
print("   slice size = %d (26%% of %d)" % (len(Xp), len(ld.X)))
print("   y range of the slice: [%.3f .. %.3f], crystal y range [%.3f .. %.3f]"
      % (yp.min(), yp.max(), ld.y.min(), ld.y.max()))
print("   are these the top-y rows? ", bool(np.all(np.diff(yp) >= -1e-9)))

print("\nC) the phase-peak band vs the CRITICAL band (they are not the same set):")
crit = np.percentile(y, 97.0)
in_crit = int((ld.y >= crit).sum())
print("   crystal rows above the stream's 97th percentile: %d" % in_crit)
print("   rows in get_phase_peak_slice():                %d" % len(Xp))
print("   overlap: %d" % len(set(np.where(ld.y >= crit)[0]) & set(np.where(ld.y >= np.sort(ld.y)[-len(Xp)])[0])))

print("\nD) empty / single-column inputs:")
for label, Xe, ye in [("one feature", X[:, :1], y), ("one row", X[:1], y[:1])]:
    try:
        dd = SpectralHologramDistiller(crystal_size=10, random_state=0)
        dd.partial_fit(Xe, ye)
        dd.finalize_crystal()
        print("   %-12s -> OK, crystal %d" % (label, len(dd.crystal_X)))
    except Exception as e:
        print("   %-12s -> %s: %s" % (label, type(e).__name__, str(e)[:70]))

print("\nE) does the loader validate that the artifact's own X matches its own X_mean?")
print("   crystal_X rows are raw feature values (not normalised): "
      "crystal_X[0][:3] = %s" % np.round(ld.X[0][:3], 3))
print("   X_mean[:3] = %s  X_std[:3] = %s" % (np.round(ld.X_mean[:3], 3), np.round(ld.X_std[:3], 3)))
print("   -> the radar normalises the STREAM with (X_mean, X_std) but the crystal")
print("      rows stay raw; mixing the two in one array is a silent trap.")

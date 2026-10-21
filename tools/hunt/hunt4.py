"""Probe set 4: internal consistency of the passport / heal / rank."""
import os, sys, tempfile
import numpy as np
from scipy import linalg as sla
sys.path.insert(0, "/sessions/affectionate-gallant-archimedes/mnt/New folder/_release/spectral-hologram-distiller")
from shd import HologramCrystalLoader, SpectralHologramDistiller

tmp = tempfile.mkdtemp(prefix="h4_")
rng = np.random.default_rng(21)
N, d = 8000, 20
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

print("=" * 72)
print("A) how much spectral mass does the 32-mode passport keep?")
print("=" * 72)
Psi = dist.Psi_sum / dist.n_seen
w, V = sla.eigh(Psi)
print("  trace(Psi_hat)            = %.6f" % np.trace(Psi))
print("  top-32 eigvalue mass       = %.6f  (%.1f%% of the trace)"
      % (w[-32:].sum(), 100 * w[-32:].sum() / np.trace(Psi)))
print("  DISCARDED mass             = %.6f  (%.1f%%)"
      % (w[:-32].sum(), 100 * w[:-32].sum() / np.trace(Psi)))
print("  data rank in sketch space  <= min(d, k) = %d ; passport keeps 32" % min(d, dist.k))

print()
print("=" * 72)
print("B) does the radar see the crystal's medoids at all?")
print("=" * 72)
# a window made of the crystal's own medoids
r_med = ld.check_domain_shift(ld.X[:64])
r_raw = ld.check_domain_shift(X[:64])
print("  window = crystal's own medoids : BW=%.5f shifted=%s" % (r_med["bw_distance"], r_med["is_shifted"]))
print("  window = raw stream rows        : BW=%.5f shifted=%s" % (r_raw["bw_distance"], r_raw["is_shifted"]))
print("  => the radar scores the PASSPORT, never the medoids: the crystal's")
print("     content is invisible to it.")

print()
print("=" * 72)
print("C) after a heal, is the passport consistent with n_seen?")
print("=" * 72)
drift = X[:400] * 2.0 + 2.0
ld2 = HologramCrystalLoader(p)
before_n = ld2.n_seen
before_trace = np.trace(ld2.Psi_passport)
res = ld2.heal_domain_shift(drift, y[:400] + 3.0, n_inject=35, random_state=0, force=True)
after_n = ld2.n_seen
after_trace = np.trace(ld2.Psi_passport)
print("  n_seen      : %d -> %d (the injected window was counted as 'seen')" % (before_n, after_n))
print("  tr(passport): %.6f -> %.6f" % (before_trace, after_trace))
print("  the passport is a BLEND of the old passport and the new window's")
print("  second moment; it is NOT the second moment of the full stream, yet")
print("  n_seen now claims the stream grew by the injected rows.")

print()
print("=" * 72)
print("D) does the heal actually move the radar toward the new domain?")
print("=" * 72)
ld3 = HologramCrystalLoader(p)
r0 = ld3.check_domain_shift(drift[:512])
for i in range(6):
    ld3.heal_domain_shift(drift[:512], y[:400] + 3.0, n_inject=35, random_state=i)
r1 = ld3.check_domain_shift(drift[:512])
print("  BW on the drifted domain: before=%.5f after 6 heals=%.5f  (threshold %.5f)"
      % (r0["bw_distance"], r1["bw_distance"], r1["threshold"]))
print("  crystal size now: %d" % len(ld3.X))
print("  => each heal re-blends the passport toward the new window, but the")
print("     blend coefficient is capped at 0.50, so 6 heals are needed to move")
print("     it, and every heal also injects 35 more medoids (unbounded growth).")

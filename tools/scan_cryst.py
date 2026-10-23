"""Grep crystal_X/crystal_Y/gamma_weights in distiller.py."""
out = []
for i, line in enumerate(open("shd/distiller.py", encoding="utf-8"), 1):
    s = line
    if any(k in s for k in ("crystal_X", "crystal_Y", "gamma_weights",
                            "def finalize_crystal", "def partial_fit",
                            "_seed_kmeans_pp", "def save_hologram")):
        out.append(f"{i}: {s.rstrip()[:140]}")
open("tools/_cryst.txt", "w", encoding="utf-8").write("\n".join(out))
print("lines:", len(out))

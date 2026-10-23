"""Grep attribute names in distiller.py (eigvec/eigen/bw/psi/spectrum)."""
out = []
for i, line in enumerate(open("shd/distiller.py", encoding="utf-8"), 1):
    s = line.lower()
    if any(k in s for k in ("eigvec", "eigenval", "self.eig", "psi", "spectrum", "bw_")):
        out.append(f"{i}: {line.rstrip()[:140]}")
open("tools/_attrs.txt", "w", encoding="utf-8").write("\n".join(out))
print("lines:", len(out))

"""List def lines and cell/band keywords in distiller.py -> tools/_defs.txt."""
out = []
for i, line in enumerate(open("shd/distiller.py", encoding="utf-8"), 1):
    s = line.strip()
    low = s.lower()
    if s.startswith("def ") or s.startswith("    def ") or "band_ratio" in low or "tail" in low:
        out.append(f"{i}: {line.rstrip()[:130]}")
open("tools/_defs.txt", "w", encoding="utf-8").write("\n".join(out))
print("lines:", len(out))

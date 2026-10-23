"""List lines mentioning 'cell' in distiller.py with line numbers."""
import io

src = open("shd/distiller.py", encoding="utf-8").read()
out = []
for i, line in enumerate(src.splitlines(), 1):
    if "cell" in line.lower():
        out.append(f"{i}: {line[:120]}")
open("tools/_cells.txt", "w", encoding="utf-8").write("\n".join(out))
print("lines:", len(out))

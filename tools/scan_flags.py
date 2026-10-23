"""Grep supervised/label flags and finalize_crystal signature in distiller.py."""
out = []
for i, line in enumerate(open("shd/distiller.py", encoding="utf-8"), 1):
    s = line.lower()
    if any(k in s for k in ("supervised", "labels_given", "def finalize_crystal",
                            "has_labels", "y_labelled", "_synthetic", "novelty")):
        out.append(f"{i}: {line.rstrip()[:140]}")
open("tools/_flags.txt", "w", encoding="utf-8").write("\n".join(out))
print("lines:", len(out))

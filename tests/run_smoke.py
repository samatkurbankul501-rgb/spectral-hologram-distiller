"""Dependency-free runner for test_smoke.py (works without pytest, e.g. Colab).

Usage: python tests/run_smoke.py
"""

import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import test_smoke as ts  # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="shd_smoke_")

    class TmpPathFactory:
        def mktemp(self, name):
            import tempfile as _tf
            from pathlib import Path

            return Path(_tf.mkdtemp(prefix=f"{name}_", dir=tmp))

    distilled = ts.make_distilled(TmpPathFactory())

    failures = []
    for name in [
        "test_version",
        "test_distiller_shapes",
        "test_save_load_roundtrip",
        "test_ood_radar",
        "test_legacy_artifact_without_projector",
    ]:
        fn = getattr(ts, name)
        try:
            if name == "test_version":
                fn()
            elif name == "test_legacy_artifact_without_projector":
                fn()
            else:
                fn(distilled)
            print(f"PASS  {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()

    print(f"\n{5 - len(failures)}/5 passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

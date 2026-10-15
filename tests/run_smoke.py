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

    no_fixture = {
        "test_version",
        "test_legacy_artifact_without_projector",
        "test_unsupervised_label_free_mode",
        "test_max_init_buffer_cap",
    }
    test_names = [
        "test_version",
        "test_distiller_shapes",
        "test_save_load_roundtrip",
        "test_ood_radar",
        "test_legacy_artifact_without_projector",
        "test_unsupervised_label_free_mode",
        "test_max_init_buffer_cap",
        "test_heal_domain_shift",
    ]
    failures = []
    for name in test_names:
        fn = getattr(ts, name)
        try:
            if name in no_fixture:
                fn()
            else:
                fn(distilled)
            print(f"PASS  {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()

    total = len(test_names)
    print(f"\n{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

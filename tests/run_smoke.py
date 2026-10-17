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
        "test_include_deltas_streaming_augmentation",
        "test_two_tailed_trigger",
        "test_adaptive_y_stats",
        "test_loader_streaming_delta_continuity",
        "test_heal_protects_low_tail_band",
        "test_input_validation_and_degenerate_data",
        "test_architecture_constants_are_wired",
        "test_partial_fit_copies_input_buffers",
        "test_mode_switch_is_rejected",
        "test_delta_width_ambiguity_is_rejected",
        "test_cli_npz_target_col",
        "test_golden_crystal_numbers",
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
        "test_loader_save_and_cli_commands",
        "test_adaptive_threshold_and_small_batch",
        "test_heal_max_crystal_size_prunes_background",
        "test_heal_max_crystal_size_is_hard_cap",
        "test_cli_npz_target_col",
        "test_golden_crystal_numbers",
        "test_include_deltas_streaming_augmentation",
        "test_two_tailed_trigger",
        "test_adaptive_y_stats",
        "test_loader_streaming_delta_continuity",
        "test_heal_protects_low_tail_band",
        "test_input_validation_and_degenerate_data",
        "test_architecture_constants_are_wired",
        "test_partial_fit_copies_input_buffers",
        "test_mode_switch_is_rejected",
        "test_delta_width_ambiguity_is_rejected",
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

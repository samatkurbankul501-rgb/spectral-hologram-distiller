"""Regression tests for bench_latency: honest window sizes, no phantom windows.

The legacy artifact labelled every measurement with the requested B, so
``XA_te[:5780]`` -- a slice shorter than 5780 on this dataset -- was recorded
as a 5780-row timing. These tests pin the fix:

* requested and actual row counts are both recorded;
* oversize requests are explicitly skipped, never silently truncated or padded;
* only honour-able windows land in ``records``/``measured_windows``/``range_ms``;
* the full held-out pool is measurable only as its own ``full`` entry;
* the legacy ``bench_latency_results.json`` name is never a default output.

Stubs/synthetic data only -- no dataset download, no heavy run. Artifacts live
under the repo's ``.test-artifacts/latency`` (never the system temp dir).
"""
import atexit
import json
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_latency as bl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / ".test-artifacts" / "latency" / f"run_{uuid.uuid4().hex[:8]}"
atexit.register(shutil.rmtree, RUN_DIR, ignore_errors=True)


def make_dir(name):
    d = RUN_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def synthetic(n=1200, seed=3, d_feat=6):
    """Small synthetic regression stream (rare hot band in y)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d_feat))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
    hot = np.abs(X[:, 0]) > 1.8
    y[hot] += 6.0 + 1.5 * X[hot, 3]
    return X, y


class StubRadar:
    """Fake HologramCrystalLoader: records the row counts it was handed."""

    def __init__(self):
        self.seen = []
        self.calls = 0

    def check_domain_shift(self, rows, **kwargs):
        self.seen.append(len(rows))
        self.calls += 1
        return {"bw_distance": 0.0, "threshold": 1.0, "latency_ms": 0.0, "is_shifted": False}


class RecordingLoader:
    """Wraps the real HologramCrystalLoader and records row counts."""

    def __init__(self, inner):
        self.inner = inner
        self.seen = []

    def check_domain_shift(self, rows, **kwargs):
        self.seen.append(len(rows))
        return self.inner.check_domain_shift(rows, **kwargs)


class StubDistiller:
    """Minimal stand-in for SpectralHologramDistiller."""

    def __init__(self):
        self.fitted = 0

    def partial_fit(self, X_batch, y_batch=None):
        self.fitted += len(X_batch)

    def finalize_crystal(self):
        return 0.0

    def save_hologram(self, path, include_projector=True):
        np.savez(path, stub=np.array([self.fitted]))
        return 0.0


def stub_distiller_factory():
    return lambda: StubDistiller()


def real_small_factory():
    """Real distiller with tiny hyper-parameters so the pipeline test stays fast."""
    return bl.default_distiller_factory(crystal_size=40, sketch_dim=32)


def fake_timer():
    """Deterministic clock: each timer() call costs 1 ms, so each timed window
    measures exactly 1.0 ms (bench_latency converts perf_counter deltas to ms)."""
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 0.001  # 1 ms in perf_counter units
        return clock["t"]

    return tick


def splits(n=1200, seed=3):
    """Same 80/20 split recipe the benchmark uses, on synthetic data."""
    X, y = synthetic(n=n, seed=seed)
    return X[: int(n * 0.8)], y[: int(n * 0.8)], X[int(n * 0.8) :]


# --- window planning ------------------------------------------------------------


def test_plan_windows_skips_oversize_instead_of_truncating():
    measured, skipped = bl.plan_windows((64, 256, 1024, 5780), 1330)
    assert [(m["requested"], m["actual_rows"]) for m in measured] == [(64, 64), (256, 256), (1024, 1024)]
    # 5780 must not be measured: the legacy bug labelled XA_te[:5780] as 5780.
    assert 5780 not in [m["actual_rows"] for m in measured]
    over = {s["requested"] for s in skipped}
    assert over == {5780}
    entry = skipped[0]
    assert "1330" in entry["reason"] and "exceeds" in entry["reason"]


def test_plan_windows_skips_oversize_when_pool_is_small():
    # 256 fits in a 375-row pool; only 1024 does not.
    measured, skipped = bl.plan_windows((64, 256, 1024), 375)
    assert [m["requested"] for m in measured] == [64, 256]
    assert [m["actual_rows"] for m in measured] == [64, 256]
    assert [s["requested"] for s in skipped] == [1024]
    assert "375" in skipped[0]["reason"]


def test_plan_windows_rejects_nonpositive_and_duplicates():
    measured, skipped = bl.plan_windows((64, 64, 0, -5), 100)
    assert [m["requested"] for m in measured] == [64]
    reasons = {s["reason"] for s in skipped}
    assert {int(64)} <= {s["requested"] for s in skipped}
    assert any("positive" in r for r in reasons)
    assert any("duplicate" in r for r in reasons)


def test_full_heldout_only_exists_when_requested():
    measured, skipped = bl.plan_windows((64,), 375, include_full=False)
    assert [m["label"] for m in measured] == ["64"]
    assert skipped == []
    measured, skipped = bl.plan_windows((64,), 375, include_full=True)
    full = measured[-1]
    assert full == {"label": "full", "requested": None, "actual_rows": 375}
    assert skipped == []


def test_full_heldout_not_invented_on_empty_pool():
    measured, skipped = bl.plan_windows((), 0, include_full=True)
    assert measured == []
    assert skipped == [{"label": "full", "requested": None, "reason": "held-out pool is empty"}]


# --- measurement ----------------------------------------------------------------


def test_measure_latency_is_warm_and_deterministic_with_fake_timer():
    radar = StubRadar()
    rows = np.arange(64 * 3, dtype=float).reshape(64, 3)
    stats = bl.measure_latency(radar, rows, repeats=5, timer=fake_timer())
    assert stats["median_ms"] == pytest.approx(1.0)
    assert stats["min_ms"] == pytest.approx(1.0)
    assert stats["max_ms"] == pytest.approx(1.0)
    assert stats["n"] == 5
    assert radar.calls == 6  # 1 warm-up + 5 repeats
    assert radar.seen == [64] * 6


def test_measure_latency_warmup_can_be_disabled():
    radar = StubRadar()
    stats = bl.measure_latency(radar, np.zeros((8, 3)), repeats=3,
                               timer=fake_timer(), warmup=False)
    assert stats["n"] == 3
    assert radar.calls == 3


def test_measure_latency_rejects_zero_repeats():
    with pytest.raises(ValueError):
        bl.measure_latency(StubRadar(), np.zeros((8, 3)), repeats=0)


# --- full benchmark (real distiller pipeline on synthetic data) ------------------


def test_run_benchmark_records_requested_and_actual_with_real_pipeline():
    XA_tr, yA_tr, XA_te = splits()
    radar = []
    payload = bl.run_benchmark(
        XA_tr, yA_tr, XA_te, "synthetic (test)", windows=(64, 1024), repeats=2,
        include_full=True, distiller_factory=real_small_factory(),
        loader_factory=lambda path: radar.append(RecordingLoader(bl.HologramCrystalLoader(path))) or radar[-1],
        crystal_dir=make_dir("real_crystal"), timer=fake_timer(), verbose=False,
    )
    available = len(XA_te)
    assert payload["heldout_rows_available"] == available
    assert payload["measured_windows"] == [64, available]
    assert "256" not in payload["records"]

    small = payload["records"]["64"]
    assert small["requested"] == 64 and small["actual_rows"] == 64
    assert small["median_ms"] == pytest.approx(1.0)  # fake timer, real path verified below

    full = payload["records"]["full"]
    assert full["requested"] is None and full["actual_rows"] == available

    over = {s["requested"] for s in payload["skipped_windows"]}
    assert over == {1024}
    assert payload["records"].keys() == {"64", "full"}

    # range_ms must span only the measured medians.
    medians = [r["median_ms"] for r in payload["records"].values()]
    assert payload["range_ms"] == [min(medians), max(medians)]

    # The real loader really ran: windows of exactly 64 then 'available' rows.
    assert radar, "loader_factory was never used"
    assert radar[-1].seen == [64] * (2 + 1) + [available] * (2 + 1)


def test_run_benchmark_loader_never_sees_an_oversize_window():
    XA_tr, yA_tr, XA_te = splits()
    payload = bl.run_benchmark(
        XA_tr, yA_tr, XA_te, "synthetic (test)", windows=(64, 5780), repeats=2,
        distiller_factory=stub_distiller_factory(), loader_factory=lambda p: StubRadar(),
        crystal_dir=make_dir("stub_crystal"), timer=fake_timer(), verbose=False,
    )
    assert payload["measured_windows"] == [64]
    assert payload["skipped_windows"] == [
        {"label": "5780", "requested": 5780,
         "reason": f"requested 5780 rows exceeds the {len(XA_te)} held-out rows available"}
    ]
    assert "5780" not in payload["records"]
    assert payload["range_ms"] == pytest.approx([1.0, 1.0])


def test_empty_payload_when_every_window_is_oversize():
    XA_tr, yA_tr, XA_te = splits(n=80)  # held-out pool ~16 rows
    payload = bl.run_benchmark(
        XA_tr, yA_tr, XA_te, "tiny", windows=(64, 5780), repeats=1,
        distiller_factory=stub_distiller_factory(), loader_factory=lambda p: StubRadar(),
        crystal_dir=make_dir("tiny_crystal"), timer=fake_timer(), verbose=False,
    )
    assert payload["records"] == {} and payload["measured_windows"] == []
    assert payload["range_ms"] is None
    assert len(payload["skipped_windows"]) == 2


# --- CLI -------------------------------------------------------------------------


def test_default_output_is_not_the_legacy_artifact():
    assert bl.DEFAULT_OUTPUT == "bench_latency_v2_results.json"
    assert bl.DEFAULT_OUTPUT != "bench_latency_results.json"


def test_main_writes_v2_json_and_never_touches_legacy_file(tmp_path, monkeypatch, capsys):
    XA_tr, yA_tr, XA_te = splits()
    guard = make_dir("legacy_guard")
    monkeypatch.chdir(guard)
    out = guard / "out.json"
    rc = bl.main(
        ["--windows", "64,5780", "--repeats", "2", "--output", str(out)],
        data_provider=lambda: ("synthetic (test)", XA_tr, yA_tr, XA_te),
        distiller_factory=stub_distiller_factory(),
        loader_factory=lambda p: StubRadar(), timer=fake_timer(),
    )
    assert rc == bl.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dataset"] == "synthetic (test)"
    assert payload["windows_requested"] == [64, 5780]
    assert list(payload["records"]) == ["64"]
    assert payload["measured_windows"] == [64]
    assert payload["records"]["64"]["requested"] == 64
    assert payload["records"]["64"]["actual_rows"] == 64
    assert payload["skipped_windows"][0]["requested"] == 5780
    # nothing in the cwd directory list is the legacy artifact name
    assert not (guard / "bench_latency_results.json").exists()
    assert capsys.readouterr().out  # progress lines printed

    # Refuse to overwrite an existing output without --force.
    before = out.read_bytes()
    rc = bl.main(
        ["--windows", "64", "--repeats", "1", "--output", str(out)],
        data_provider=lambda: ("synthetic (test)", XA_tr, yA_tr, XA_te),
        distiller_factory=stub_distiller_factory(),
        loader_factory=lambda p: StubRadar(), timer=fake_timer(),
    )
    assert rc == bl.EXIT_FAILED
    assert out.read_bytes() == before  # untouched
    assert "--force" in capsys.readouterr().err

    # --force rewrites it.
    rc = bl.main(
        ["--windows", "64", "--repeats", "1", "--output", str(out), "--force"],
        data_provider=lambda: ("synthetic (test)", XA_tr, yA_tr, XA_te),
        distiller_factory=stub_distiller_factory(),
        loader_factory=lambda p: StubRadar(), timer=fake_timer(),
    )
    assert rc == bl.EXIT_OK
    assert out.exists() and json.loads(out.read_text(encoding="utf-8"))["repeats"] == 1


def test_main_default_output_name_lands_in_cwd(tmp_path, monkeypatch):
    XA_tr, yA_tr, XA_te = splits()
    guard = make_dir("default_out")
    monkeypatch.chdir(guard)
    rc = bl.main(
        ["--windows", "64", "--repeats", "1"],
        data_provider=lambda: ("synthetic (test)", XA_tr, yA_tr, XA_te),
        distiller_factory=stub_distiller_factory(),
        loader_factory=lambda p: StubRadar(), timer=fake_timer(),
    )
    assert rc == bl.EXIT_OK
    created = sorted(p.name for p in guard.iterdir())
    assert "bench_latency_v2_results.json" in created
    assert "bench_latency_results.json" not in created


def test_main_reports_nothing_measured_when_all_oversize():
    out = make_dir("nothing") / "out.json"
    rc = bl.main(
        ["--windows", "4096", "--repeats", "1", "--output", str(out)],
        data_provider=lambda: ("synthetic (test)", *splits()),
        distiller_factory=stub_distiller_factory(),
        loader_factory=lambda p: StubRadar(), timer=fake_timer(),
    )
    assert rc == bl.EXIT_NOTHING_MEASURED
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["records"] == {} and payload["range_ms"] is None
    assert payload["skipped_windows"][0]["requested"] == 4096


def test_parse_windows_rejects_junk():
    assert bl.parse_windows("64, 256 ,1024") == [64, 256, 1024]
    with pytest.raises(SystemExit):
        bl.parse_windows("64,abc")


def test_cli_validates_positive_repeats_and_batch(capsys):
    argv = ["--windows", "64", "--repeats", "0"]
    assert bl.main(argv, data_provider=lambda: ("synthetic (test)", *splits()),
                   distiller_factory=stub_distiller_factory(),
                   loader_factory=lambda p: StubRadar(), timer=fake_timer()) == bl.EXIT_FAILED
    assert "--repeats" in capsys.readouterr().err
    argv = ["--windows", "64", "--fit-batch", "0"]
    assert bl.main(argv, data_provider=lambda: ("synthetic (test)", *splits()),
                   distiller_factory=stub_distiller_factory(),
                   loader_factory=lambda p: StubRadar(), timer=fake_timer()) == bl.EXIT_FAILED
    assert "--fit-batch" in capsys.readouterr().err


def test_main_requires_at_least_one_window(capsys):
    rc = bl.main(["--windows", "", "--repeats", "1"],
                 data_provider=lambda: ("synthetic (test)", *splits()),
                 distiller_factory=stub_distiller_factory(),
                 loader_factory=lambda p: StubRadar(), timer=fake_timer())
    assert rc == bl.EXIT_FAILED
    assert "no window sizes requested" in capsys.readouterr().err


# --- dataset fallback policy -----------------------------------------------------


def test_dataset_fallback_is_refused_by_default(monkeypatch):
    monkeypatch.setattr(bl, "load_superconduct",
                        lambda: (np.zeros((100, 6)), np.zeros(100), 77.0,
                                 "California housing (fallback)"))
    with pytest.raises(RuntimeError, match="fallback"):
        bl.default_data_provider()

    monkeypatch.setattr(bl, "load_superconduct",
                        lambda: (np.zeros((100, 6)), np.zeros(100), 77.0, "UCI Superconductivity"))
    title, XA_tr, yA_tr, XA_te = bl.default_data_provider(allow_fallback=True)
    assert title == "UCI Superconductivity"
    assert len(XA_te) > 0 and len(XA_tr) > 0 and len(yA_tr) == len(XA_tr)


def test_main_exits_cleanly_on_dataset_fallback(monkeypatch, capsys):
    monkeypatch.setattr(bl, "load_superconduct",
                        lambda: (np.zeros((100, 6)), np.zeros(100), 77.0,
                                 "California housing (fallback)"))
    out = make_dir("fallback") / "out.json"
    rc = bl.main(["--windows", "16", "--repeats", "1", "--output", str(out)],
                 distiller_factory=stub_distiller_factory(),
                 loader_factory=lambda p: StubRadar(), timer=fake_timer())
    assert rc == bl.EXIT_FAILED
    assert "fallback" in capsys.readouterr().err
    assert not out.exists()


def test_main_allow_dataset_fallback_flag_proceeds(monkeypatch, capsys):
    X, y = synthetic(n=1200, seed=3)
    A = y < 40.0
    monkeypatch.setattr(bl, "load_superconduct",
                        lambda: (X[A], y[A], 77.0, "California housing (fallback)"))
    out = make_dir("allowed") / "out.json"
    rc = bl.main(["--windows", "64", "--repeats", "1", "--output", str(out),
                  "--allow-dataset-fallback", "--full-heldout"],
                 distiller_factory=stub_distiller_factory(),
                 loader_factory=lambda p: StubRadar(), timer=fake_timer())
    assert rc == bl.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dataset"] == "California housing (fallback)"
    assert payload["measured_windows"] == [64, payload["heldout_rows_available"]]
    assert payload["records"]["full"]["requested"] is None

"""Protocol-v2 regressions: real time, bounded alerts, identical replay."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shd_watch as watch
from bench_metro_watch import exposure_months, score_feature_z, score_iforest, unmatched_fa


def axis(n, step=30):
    return np.datetime64("2020-03-01T00:00", "m") + np.arange(n) * np.timedelta64(step, "m")


def test_constant_alert_cannot_pass_budget():
    ts = axis(40 * 48)
    scores = np.random.default_rng(123).uniform(.1, .5, len(ts))
    tau, count = watch.calibrate_tau("2020-03-01", "2020-04-10", ts, scores, 1)
    assert tau > scores.min()
    assert np.mean(scores >= tau) <= watch.MAX_ALERT_FRACTION
    assert count <= 1


def test_zero_budget_has_finite_never_alert_threshold():
    ts = axis(100)
    scores = np.ones(len(ts))
    tau, count = watch.calibrate_tau("2020-03-01", "2020-03-04", ts, scores, 0)
    assert np.isfinite(tau) and tau > 1 and count == 0


def test_candidate_selection_replays_frozen_baseline():
    ts = axis(28 * 48)
    ts[400:] += np.timedelta64(24, "h")
    ts[800:] += np.timedelta64(24, "h")
    scores = np.cumsum(np.random.default_rng(0).normal(0, .02, len(ts))) + 2
    tau, count = watch.calibrate_tau("2020-03-01", "2020-04-02", ts, scores, 2,
                                    halflife_days=7, step_min=30)
    replay = watch.detrended_scores(ts, scores, 7, 30, tau=tau)
    assert count == len(watch.episodes(ts, replay, tau))
    assert count <= 1  # 28 observed days at 2/month: floor(1.84)
    assert np.mean(replay >= tau) <= watch.MAX_ALERT_FRACTION


def test_short_slice_does_not_round_budget_up():
    ts = axis(26 * 48)
    scores = np.linspace(0, 1, len(ts))
    tau, count = watch.calibrate_tau("2020-03-01", "2020-03-27", ts, scores, 1)
    assert count == 0
    assert tau > scores.max()


def test_gaps_are_not_counted_as_healthy_exposure():
    ts = axis(100)
    ts[50:] += np.timedelta64(28, "D")
    scores = np.linspace(0, 1, len(ts))
    tau, count = watch.calibrate_tau("2020-03-01", "2020-04-10", ts, scores, 1)
    assert count == 0 and tau > scores.max()


class DummyRadar:
    def check_domain_shift(self, X, **kwargs):
        return {"bw_distance": float(X.mean())}


class DummyForest:
    def decision_function(self, X):
        return X[:, 0]


def test_windows_never_bridge_missing_minutes_and_arms_align():
    ts = axis(180, 1)
    ts[80:] += np.timedelta64(1300, "m")
    X = np.arange(360, dtype=float).reshape(180, 2)
    bounds = list(watch.window_bounds(ts, 60, 30))
    assert bounds == [(0, 60), (80, 140), (110, 170)]
    for lo, hi in bounds:
        assert ts[hi - 1] - ts[lo] == np.timedelta64(59, "m")
    idx, _ = watch.window_scores(DummyRadar(), X, 60, 30, minutes=ts)
    zi, _ = score_feature_z(X, X, 60, 30, minutes=ts)
    fi, _ = score_iforest(DummyForest(), X, 60, 30, minutes=ts)
    np.testing.assert_array_equal(idx, zi)
    np.testing.assert_array_equal(idx, fi)


@pytest.mark.parametrize("win,step", [(0, 30), (60, 0), (60, -1)])
def test_invalid_windows_fail_fast(win, step):
    with pytest.raises(ValueError):
        list(watch.window_bounds(axis(180, 1), win, step))


def test_ewma_uses_elapsed_time_and_baseline_timestamp():
    ts = np.array(["2020-03-01T00:00", "2020-03-02T00:00"], dtype="datetime64[m]")
    dev, baseline = watch.detrended_scores(ts, np.array([0., 10.]), 1, 30,
                                          return_baseline=True)
    np.testing.assert_allclose(dev, [0, 10])
    assert baseline == pytest.approx(5.)
    _, baseline = watch.detrended_scores(ts[1:], np.array([10.]), 1, 30,
                                         b_init=0, baseline_time=ts[0],
                                         return_baseline=True)
    assert baseline == pytest.approx(5.)


def test_learn_accepts_npz_without_csv(tmp_path, monkeypatch, capsys):
    p = tmp_path / "minutes.npz"
    np.savez(p, minutes=axis(10, 1).astype("int64"), X=np.ones((10, 2)), features=["a", "b"])
    class Crystal:
        def save_hologram(self, path):
            return 1.0
    monkeypatch.setattr(watch, "build_crystal", lambda X, m: Crystal())
    watch.main(["learn", "--npz", str(p), "--from", "2020-03-01", "--to", "2020-03-02",
                "--out", str(tmp_path / "crystal.npz")])
    assert json.loads(capsys.readouterr().out)["rows"] == 10


def test_csv_requires_named_columns(capsys):
    with pytest.raises(SystemExit) as exc:
        watch.main(["learn", "--csv", "unused.csv", "--from", "2020-03-01", "--to", "2020-03-02", "--out", "unused.npz"])
    assert exc.value.code == 2
    assert "--tcol and --columns" in capsys.readouterr().err


def test_legacy_threshold_requires_recalibration(tmp_path, monkeypatch):
    cfg = tmp_path / "old.json"
    cfg.write_text('{"tau": 0.5}', encoding="utf-8")
    monkeypatch.setattr(watch, "HologramCrystalLoader", lambda p: DummyRadar())
    with pytest.raises(SystemExit, match="re-run calibrate"):
        watch.main(["watch", "--npz", "unused.npz", "--crystal", "unused.npz",
                    "--threshold", str(cfg), "--from", "2020-03-01", "--to", "2020-03-02"])


def test_calibrate_then_watch_same_contract(tmp_path, monkeypatch, capsys):
    p = tmp_path / "minutes.npz"
    ts = axis(4 * 24 * 60, 1)
    np.savez(p, minutes=ts.astype("int64"), X=np.ones((len(ts), 2)), features=["a", "b"])
    monkeypatch.setattr(watch, "HologramCrystalLoader", lambda p: DummyRadar())
    cfg = tmp_path / "threshold.json"
    common = ["--npz", str(p), "--crystal", "unused.npz"]
    watch.main(["calibrate", *common, "--from", "2020-03-01", "--to", "2020-03-03", "--out", str(cfg)])
    c = json.loads(capsys.readouterr().out)
    assert c["protocol_version"] == 2 and c["calibration_episodes"] == 0
    watch.main(["watch", *common, "--threshold", str(cfg), "--from", "2020-03-03", "--to", "2020-03-05"])
    result = json.loads(capsys.readouterr().out)
    assert result["episodes"] == 0 and result["windows"] > 0
    with pytest.raises(SystemExit, match="re-running calibrate"):
        watch.main(["watch", *common, "--threshold", str(cfg), "--from", "2020-03-03", "--to", "2020-03-05", "--step-min", "15"])


def test_real_learn_calibrate_watch_roundtrip(tmp_path, capsys):
    ts = axis(5 * 24 * 60, 1)
    rng = np.random.default_rng(83)
    p = tmp_path / "real.npz"
    np.savez(p, minutes=ts.astype("int64"), X=rng.normal(size=(len(ts), 2)),
             features=["pressure", "temperature"])
    crystal, cfg = tmp_path / "real_crystal.npz", tmp_path / "real_threshold.json"
    watch.main(["learn", "--npz", str(p), "--from", "2020-03-01", "--to", "2020-03-02",
                "--m", "8", "--out", str(crystal)])
    assert json.loads(capsys.readouterr().out)["rows"] == 1440
    assert crystal.is_file()
    common = ["--npz", str(p), "--crystal", str(crystal)]
    watch.main(["calibrate", *common, "--from", "2020-03-02", "--to", "2020-03-04", "--out", str(cfg)])
    calibration = json.loads(capsys.readouterr().out)
    assert calibration["calibration_episodes"] == 0
    assert np.isfinite(calibration["baseline_end"])
    watch.main(["watch", *common, "--threshold", str(cfg), "--from", "2020-03-04", "--to", "2020-03-06"])
    report = json.loads(capsys.readouterr().out)
    assert report["windows"] == 95
    assert report["tau"] == calibration["tau"]
    assert report["episodes"] == len(report["alerts"])
    with pytest.raises(SystemExit, match="after the calibration"):
        watch.main(["watch", *common, "--threshold", str(cfg), "--from", "2020-03-03", "--to", "2020-03-06"])


def test_read_csv_minutes_rejects_nonfinite_features(tmp_path):
    p = tmp_path / "nan.csv"
    p.write_text("time,a,b\n2020-03-01 00:00:00,1,nan\n2020-03-01 00:01:00,2,2\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        watch.read_csv_minutes(str(p), "time", ["a", "b"])


def test_read_csv_minutes_rejects_header_only_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("time,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        watch.read_csv_minutes(str(p), "time", ["a"])


def test_read_csv_minutes_rejects_unparseable_timestamp(tmp_path):
    p = tmp_path / "badts.csv"
    p.write_text("time,a\nnot-a-date,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timestamp"):
        watch.read_csv_minutes(str(p), "time", ["a"])


def test_read_csv_minutes_rejects_unordered_timestamps(tmp_path):
    p = tmp_path / "unordered.csv"
    p.write_text("time,a\n2020-03-01 00:02:00,5\n2020-03-01 00:00:00,1\n2020-03-01 00:01:00,3\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="non-decreasing"):
        watch.read_csv_minutes(str(p), "time", ["a"])


def test_watch_rejects_threshold_missing_v2_keys(tmp_path, monkeypatch):
    cfg = tmp_path / "partial.json"
    cfg.write_text(json.dumps({"protocol_version": 2, "tau": 0.5}), encoding="utf-8")
    monkeypatch.setattr(watch, "HologramCrystalLoader", lambda p: DummyRadar())
    with pytest.raises(SystemExit, match="re-run calibrate"):
        watch.main(["watch", "--npz", "unused.npz", "--crystal", "unused.npz",
                    "--threshold", str(cfg), "--from", "2020-03-01",
                    "--to", "2020-03-02"])


def test_unmatched_fa_rate_uses_budget_exposure_denominator():
    ends = (np.datetime64("2020-04-10T00:59", "m")
            + np.array([0, 30, 60, 420, 450, 480], dtype="timedelta64[m]"))
    gaps = np.diff(ends) / np.timedelta64(1, "m")
    expected = min(24 * 60, 30 + float(np.minimum(gaps, 30).sum())) / (30.4375 * 24 * 60)
    assert exposure_months(ends, 30, "2020-04-10", "2020-04-11") == pytest.approx(expected)
    incidents = [{"name": "far_leak", "start": "2020-06-01T10:00",
                  "end": "2020-06-01T12:00"}]
    eps = [(np.datetime64("2020-04-10T02:00", "m"),
            np.datetime64("2020-04-10T02:30", "m"), 1.0)]
    un, rates = unmatched_fa(incidents, eps, "2020-04-10", "2020-04-11", ends, 30)
    assert un == 1
    assert rates["per_month_observed"] > rates["per_month_calendar"]
    assert rates["exposure_months"] < rates["calendar_months"]


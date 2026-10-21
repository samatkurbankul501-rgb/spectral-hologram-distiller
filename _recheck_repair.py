"""Independent recheck of the v3.11.0 repair (2026-10-20).

Reprobes the ORIGINAL audit triggers for all 14 defects against the current
tree (not the workers' own test files), then cross-checks every number that
was hand-written into README/PITCH/EMAIL_ENGINEER/paper against the fresh
JSON artifacts, and confirms the historical artifacts were not rewritten.

Exit code 0 = all checks pass. ASCII output only.
"""

import io
import json
import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("  {}  {}{}".format(tag, name, ("  | " + str(detail)) if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def section(title):
    print("\n" + "=" * 90 + "\n" + title + "\n" + "=" * 90)


def axis(n, step=30):
    return (np.datetime64("2020-03-01T00:00", "m")
            + np.arange(n) * np.timedelta64(step, "m"))


def in_memory_load(d):
    b = io.BytesIO()
    real = np.savez_compressed
    with patch("shd.distiller.np.savez_compressed",
               side_effect=lambda path, **kw: real(b, **kw)), \
         patch("shd.distiller.os.path.getsize", return_value=0):
        d.save_hologram("memory_recheck.npz")
    b.seek(0)
    from shd import HologramCrystalLoader
    return HologramCrystalLoader(b)


# ---------------------------------------------------------------- watch (D1-D3)
section("Watch protocol v2: original triggers D1-D3")

import shd_watch as watch

# D1: a sustained constant alert used to merge into ONE episode and pass the
# monthly budget; budget=0 used to return inf.
ts = axis(26 * 48)
scores = np.ones(len(ts))
tau1, count1 = watch.calibrate_tau("2020-03-01", "2020-03-27", ts, scores, 1.0)
replay1 = scores  # raw (no-halflife) arm: replay is the scores themselves
check("D1 constant-alert threshold finite", np.isfinite(tau1), "tau={}".format(tau1))
check("D1 budget-1 calibration gives 0 episodes", count1 == 0, "count={}".format(count1))
check("D1 frozen-replay episode count matches", count1 == len(watch.episodes(ts, replay1, tau1)))
check("D1 alert fraction respects the 5% cap",
      float(np.mean(replay1 >= tau1)) <= watch.MAX_ALERT_FRACTION + 1e-12)
tau0, count0 = watch.calibrate_tau("2020-03-01", "2020-03-27", ts, scores, 0.0)
check("D1 zero-budget returns a finite never-alert threshold",
      np.isfinite(tau0) and tau0 > 1.0 and count0 == 0, "tau={}".format(tau0))

# D2: calibration replay must use the SAME freeze-on-alert EWMA as watch.
ts2 = axis(28 * 48)
ts2[400:] += np.timedelta64(24, "h")
ts2[800:] += np.timedelta64(24, "h")
scores2 = np.cumsum(np.random.default_rng(0).normal(0, .02, len(ts2))) + 2
tau2, count2 = watch.calibrate_tau("2020-03-01", "2020-04-02", ts2, scores2, 2,
                                   halflife_days=7, step_min=30)
replay2 = watch.detrended_scores(ts2, scores2, 7, 30, tau=tau2)
check("D2 frozen replay equals calibration episode count",
      count2 == len(watch.episodes(ts2, replay2, tau2)), "cal={} replay={}".format(
          count2, len(watch.episodes(ts2, replay2, tau2))))
check("D2 replay respects the 5% cap",
      float(np.mean(replay2 >= tau2)) <= watch.MAX_ALERT_FRACTION + 1e-12)

# D3: row-based windows used to cross telemetry gaps.
ts3 = axis(180, 1)
ts3[80:] += np.timedelta64(1300, "m")
bounds = list(watch.window_bounds(ts3, 60, 30))
check("D3 windows never bridge a gap", bounds == [(0, 60), (80, 140), (110, 170)],
      "bounds={}".format(bounds))
check("D3 every window spans exactly 59 minutes",
      all(ts3[hi - 1] - ts3[lo] == np.timedelta64(59, "m") for lo, hi in bounds))

# ------------------------------------------------------------------ core (D4-D10)
section("Core distiller/CLI: original triggers D4-D10")

from shd import SpectralHologramDistiller

rng = np.random.default_rng(13)
XB, yB = rng.normal(size=(2000, 4)), rng.normal(size=2000)

# D4: a 2000-row first batch with max_init_buffer=40 used to hand the lattice
# builder all 2000 rows.
d4 = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                               max_init_buffer=40)
sizes = []
cls = type(d4)
orig_build = cls._build_phase_manifold_lattice


def spy(self):
    sizes.append(sum(len(a) for a in self.init_X))
    return orig_build(self)


cls._build_phase_manifold_lattice = spy
try:
    d4.partial_fit(XB, yB)
finally:
    cls._build_phase_manifold_lattice = orig_build
check("D4 lattice builder never saw more than cap rows",
      bool(sizes) and max(sizes) <= 40, "seen buffer sizes={}".format(sorted(set(sizes))))
check("D4 split tail still streamed (no rows lost)", d4.n_seen == 2000,
      "n_seen={}".format(d4.n_seen))
d4.finalize_crystal()
check("D4 crystal size respects M=20", len(d4.crystal_X) == 20)

# D5: adaptive y-stats used to leave every centre in the old coordinates.
rng5 = np.random.default_rng(2)
X5 = rng5.normal(size=(1200, 4))
y5 = X5[:, 0] * 2.0 + rng5.normal(scale=.05, size=1200)
d5 = SpectralHologramDistiller(crystal_size=24, sketch_dim=16, random_state=2,
                               adaptive_y_stats=True, high_band_ratio=0.3,
                               center_lr=0.0, phase_kappa=3.2, max_init_buffer=1000)
for i in range(0, 1200, 200):
    d5.partial_fit(X5[i:i + 200], y5[i:i + 200])
assert d5.lattice_ready
c_before = np.asarray(d5.centers_ZY).copy()
mean_b, std_b = float(d5.y_mean), float(d5.y_std)
d5.partial_fit(X5[:200] + 3.0, y5[:200] + 3.0)
c_after = np.asarray(d5.centers_ZY)
mean_a, std_a = float(d5.y_mean), float(d5.y_std)
kappa = 3.2
y_implied_before = c_before[:, -1] * std_b / kappa + mean_b
y_implied_after = c_after[:, -1] * std_a / kappa + mean_a
check("D5 adaptive rebase keeps centres in a consistent coordinate frame",
      np.allclose(y_implied_before, y_implied_after, atol=1e-8),
      "max |delta| = {:.3e}".format(float(np.max(np.abs(y_implied_before - y_implied_after)))))

# D6: one 81-D observation decided differently as a vector vs a matrix.
Z81 = np.zeros((400, 81))
Z81[300:] = 1.0
y81 = np.arange(400, dtype=np.float64)
d6 = SpectralHologramDistiller(crystal_size=20, sketch_dim=32, random_state=0,
                               max_init_buffer=40)
d6.partial_fit(Z81, y81)
d6.finalize_crystal()
loader6 = in_memory_load(d6)
rep_v = loader6.check_domain_shift(np.zeros(81), adaptive_threshold=False)
rep_m = loader6.check_domain_shift(np.zeros((1, 81)), adaptive_threshold=False)
check("D6 same BW distance for vector and (1,d) query",
      abs(rep_v["bw_distance"] - rep_m["bw_distance"]) < 1e-12,
      "{} vs {}".format(rep_v["bw_distance"], rep_m["bw_distance"]))
check("D6 identical verdict for both encodings",
      rep_v["is_shifted"] == rep_m["is_shifted"],
      "vector={} matrix={}".format(rep_v["is_shifted"], rep_m["is_shifted"]))

# D7: a singleton crystal used to crash on roundtrip/peak-slice/heal.
d7 = SpectralHologramDistiller(crystal_size=1, sketch_dim=8, random_state=0,
                               max_init_buffer=40)
d7.partial_fit(np.array([[0.0, 1.0, 2.0]]), np.array([5.0]))
d7.finalize_crystal()
check("D7 crystal_Y stays (1,) for a singleton", np.shape(d7.crystal_Y) == (1,),
      "shape={}".format(np.shape(d7.crystal_Y)))
loader7 = in_memory_load(d7)
Xc7, yc7, wc7 = loader7.get_training_data()
check("D7 roundtrip shapes", Xc7.shape == (1, 3) and yc7.shape == (1,)
      and wc7.shape == (1,), "X{} y{} w{}".format(Xc7.shape, yc7.shape, wc7.shape))
ok7 = True
try:
    loader7.get_phase_peak_slice()
    loader7.heal_domain_shift(np.array([[3.0, 1.0, 0.0]]), n_inject=1,
                             random_state=0)
except Exception as exc:  # pragma: no cover - the original defect raised here
    ok7 = False
    print("      singleton exception:", type(exc).__name__, exc)
check("D7 peak-slice and heal survive a singleton", ok7)

# D8: seeding with a zero-size band used to add one stray centre.
for m, br in ((20, 0.0), (20, 0.01), (20, 0.1), (1, 0.26)):
    d8 = SpectralHologramDistiller(crystal_size=m, sketch_dim=16, random_state=0,
                                   high_band_ratio=br, max_init_buffer=500)
    d8.partial_fit(XB[:300], yB[:300])
    d8.finalize_crystal()
    check("D8 M={} br={} yields exactly {} centres".format(m, br, m),
          len(d8.crystal_X) == m, "got {}".format(len(d8.crystal_X)))

# D9: heal precheck used to SUM overlapping protected sets instead of union.
rng9 = np.random.default_rng(0)
X9 = rng9.normal(size=(200, 4))
d9 = SpectralHologramDistiller(crystal_size=20, sketch_dim=16, random_state=0,
                               high_band_ratio=0.25, max_init_buffer=500)
d9.partial_fit(X9, np.arange(200, dtype=np.float64))
d9.finalize_crystal()
loader9 = in_memory_load(d9)
n_old = len(loader9.X)
band = max(1, int(round(loader9.high_band_ratio * n_old)))
peak_idx = np.argsort(loader9.y)[-band:]
weights = np.zeros(n_old)
weights[peak_idx] = 1.0 + np.arange(band) * 0.1
loader9.weights = weights
res9 = loader9.heal_domain_shift(rng9.normal(size=(64, 4)), n_inject=2,
                                 max_crystal_size=7, protect_old_weight_frac=0.25,
                                 force=True, random_state=1)
check("D9 overlapping protections fit the union cap",
      res9 is not None and len(loader9.X) <= 7, "len={}".format(len(loader9.X)))
loader9b = in_memory_load(d9)
loader9b.weights = weights
raised = False
try:
    loader9b.heal_domain_shift(rng9.normal(size=(64, 4)), n_inject=2,
                              max_crystal_size=6, protect_old_weight_frac=0.25,
                              force=True, random_state=1)
except ValueError:
    raised = True
check("D9 one slot short of the union is still rejected", raised)

# D10: the reported spectral error used to be the pure-herding intermediate,
# not the exported blended weights.
d10 = SpectralHologramDistiller(crystal_size=16, sketch_dim=16, random_state=53,
                                max_init_buffer=1000)
d10.partial_fit(XB[:600], yB[:600])
returned10 = d10.finalize_crystal()
check("D10 herding intermediate reported under an explicit name",
      hasattr(d10, "herding_rel_error") and np.isfinite(d10.herding_rel_error))
check("D10 exported-blend error differs from the herding intermediate",
      abs(returned10 - d10.herding_rel_error) > 1e-9,
      "blend={} herding={}".format(returned10, d10.herding_rel_error))

# D11: single-row CSV used to load as a one-COLUMN dataset.
from shd.cli import _load_tabular
tmp = tempfile.mkdtemp(prefix="shd_recheck_")
p11 = os.path.join(tmp, "one_row.csv")
with open(p11, "w", encoding="utf-8") as f:
    f.write("1,2,3\n")
X11, y11 = _load_tabular(p11, target_col=-1)
check("D11 single-row CSV stays (1,2) with target -1",
      X11.shape == (1, 2) and y11 is not None and float(y11[0]) == 3.0,
      "X{} y{}".format(X11.shape, None if y11 is None else y11.shape))
with open(p11, "w", encoding="utf-8") as f:
    f.write("1,2,3\n")
X11b, _ = _load_tabular(p11)
check("D11 single-row CSV without target stays (1,3)", X11b.shape == (1, 3),
      "X{}".format(X11b.shape))

# --------------------------------------------------------------- CLI/watch (D12)
section("shd_watch CLI: original trigger D12")

import importlib
cap_out = []
saved_stdout = sys.stdout


class _Cap:
    def write(self, s):
        cap_out.append(s)

    def flush(self):
        pass


npz12 = os.path.join(tmp, "minutes12.npz")
np.savez(npz12, minutes=axis(2 * 48).astype("int64"),
         X=np.random.default_rng(5).normal(size=(2 * 48, 2)),
         features=["a", "b"])
crystal12 = os.path.join(tmp, "crystal12.npz")
ok12 = True
err12 = ""
sys.stdout = _Cap()
try:
    watch.main(["learn", "--npz", npz12, "--from", "2020-03-01",
                "--to", "2020-03-02", "--m", "8", "--out", crystal12])
except SystemExit as exc:
    ok12 = False
    err12 = "SystemExit {}".format(exc.code)
except Exception as exc:  # pragma: no cover
    ok12 = False
    err12 = "{}: {}".format(type(exc).__name__, exc)
finally:
    sys.stdout = saved_stdout
check("D12 NPZ-only learn works without --csv", ok12 and os.path.exists(crystal12),
      err12 or "crystal written")
rep12 = json.loads("".join(cap_out))
check("D12 learn report row count (30-min cadence over one day)",
      rep12.get("rows") == 48, "rows={}".format(rep12.get("rows")))

# --------------------------------------------------------------- latency (D13)
section("Latency benchmark: original trigger D13")

from bench_latency import plan_windows
measured, skipped = plan_windows([5780], available=1330)
check("D13 oversize request is skipped, never silently truncated",
      measured == [] and len(skipped) == 1 and "exceeds" in skipped[0]["reason"],
      "skipped={}".format(skipped))
measured2, skipped2 = plan_windows([256], available=375)
check("D13 a fitting window is measured at its true size",
      [m["actual_rows"] for m in measured2] == [256] and skipped2 == [],
      "measured={}".format(measured2))

# ----------------------------------------------------------------- edge (D14)
section("Edge envelope: original trigger D14 (labels)")

with open("bench_edge_envelope_v2_results.json", encoding="utf-8") as f:
    edge = json.load(f)
check("D14 data source labelled synthetic",
      str(edge.get("data_source", "")).startswith("synthetic"),
      edge.get("data_source"))
check("D14 cases named synthetic-*",
      all(str(c["case"]).startswith("synthetic-") for c in edge["cases"]),
      [c["case"] for c in edge["cases"]])
check("D14 no superconductivity claim in the v2 artifact",
      "superconduct" not in json.dumps(edge).lower())

# --------------------------------------------- fresh artifacts vs published text
section("Fresh artifacts vs hand-written numbers in docs")

with open("bench_metro_watch_v2_results.json", encoding="utf-8") as f:
    metro = json.load(f)
with open("bench_scania_cost_official_results.json", encoding="utf-8") as f:
    scania = json.load(f)
with open("bench_latency_v2_results.json", encoding="utf-8") as f:
    latency = json.load(f)

arms = {a["arm"]: a for a in metro["arms"]}
z = arms["z_detrend"]
check("metro: SHD detected 0/4",
      sum(i["detected"] for i in arms["shd_detrend"]["incidents"]) == 0)
check("metro: z-score detected 4/4",
      sum(i["detected"] for i in z["incidents"]) == 4)
leads = [i["lead_hours"] for i in z["incidents"]]
check("metro: z leads match README row",
      all(abs(l - r) < .05 for l, r in zip(leads, [131.3, 392.9, 415.4, 163.7])),
      "leads={}".format([round(l, 1) for l in leads]))
check("metro: z unmatched rate 1.743/month (README: 1.74)",
      abs(z["unmatched_fa_per_month"] - 1.743) < 1e-9)
check("metro: exposure 2.869 months (README: 2.87)",
      abs(z["watch_exposure_months"] - 2.869) < 1e-9)
check("metro: 4192 watch windows / 994 calibration windows",
      z["watch_windows"] == 4192 and z["calibration_windows"] == 994)
check("metro: crystal 54.8 KB", abs(arms["shd_detrend"]["crystal_kb"] - 54.8) < .05)

s = scania["summary"]
check("scania: shd_hgb cost/fn/fp medians",
      (s["shd_hgb"]["cost"], s["shd_hgb"]["fn"], s["shd_hgb"]["fp"]) == (18645.0, 17.5, 969.0),
      (s["shd_hgb"]["cost"], s["shd_hgb"]["fn"], s["shd_hgb"]["fp"]))
check("scania: kmeans_hgb 31340.0 / 41.5 / 828.5",
      (s["kmeans_hgb"]["cost"], s["kmeans_hgb"]["fn"], s["kmeans_hgb"]["fp"]) == (31340.0, 41.5, 828.5))
check("scania: random_hgb 32355.0 / 34.0 / 1074.5",
      (s["random_hgb"]["cost"], s["random_hgb"]["fn"], s["random_hgb"]["fp"]) == (32355.0, 34.0, 1074.5))
check("scania: logit medians",
      s["shd_logit"]["cost"] == 53330.0 and s["kmeans_logit"]["cost"] == 23080.0
      and s["random_logit"]["cost"] == 32545.0)
check("scania: oracle / never-alert",
      scania["oracle_fulltrain"]["hgb"]["cost"] == 10460
      and scania["oracle_fulltrain"]["logit"]["cost"] == 16800
      and scania["never_alert"]["cost"] == 187500)
w = scania["wilcoxon"]
check("scania: hgb significance 9/10 p=0.0039 and 8/10 p=0.0098",
      w["shd_vs_random_hgb_cost"]["wins_shd"] == 9
      and abs(w["shd_vs_random_hgb_cost"]["p_value"] - 0.00390625) < 1e-12
      and w["shd_vs_kmeans_hgb_cost"]["wins_shd"] == 8
      and abs(w["shd_vs_kmeans_hgb_cost"]["p_value"] - 0.009765625) < 1e-12)
check("scania: crystal positives 7.0 / 7.5 / 25.0",
      scania["crystal_positives"] == {"random": 7.0, "kmeans": 7.5, "shd": 25.0})
check("scania: cost model is the official 500/10",
      scania["cost_model"] == {"fn": 500, "fp": 10})

recs = latency["records"]
check("latency: medians 2.21 / 8.37 / 20.11 / 62.35 ms",
      abs(recs["64"]["median_ms"] - 2.21) < .005
      and abs(recs["256"]["median_ms"] - 8.37) < .005
      and abs(recs["1024"]["median_ms"] - 20.11) < .005
      and abs(recs["full"]["median_ms"] - 62.35) < .005,
      {k: round(v["median_ms"], 2) for k, v in recs.items()})
check("latency: every record carries requested and actual_rows",
      all(("requested" in r and "actual_rows" in r) for r in recs.values()))
check("latency: full pool measured at its true size 2808",
      recs["full"]["actual_rows"] == 2808
      and latency["heldout_rows_available"] == 2808)
check("latency: nothing skipped in this run", latency["skipped_windows"] == [])

docs = {}
for name in ("README.md", "PITCH.md", "EMAIL_ENGINEER.md", "paper/main.tex",
             "ARTIFACTS.md"):
    with open(name, encoding="utf-8") as f:
        docs[name] = f.read()
r, p, e, tex, art = (docs["README.md"], docs["PITCH.md"], docs["EMAIL_ENGINEER.md"],
                     docs["paper/main.tex"], docs["ARTIFACTS.md"])

check("README quotes official Scania numbers",
      all(t in r for t in ("18645.0", "31340.0", "32355.0", "10460", "187500",
                           "p=0.0039", "p=0.0098", "17.5", "34.0", "41.5")))
check("README quotes metro v2 numbers",
      all(t in r for t in ("131.3 / 392.9 / 415.4 / 163.7", "1.74", "2.87",
                           "54.8 KB", "994", "4,192")))
check("README quotes edge v2 numbers",
      all(t in r for t in ("28.9 KB", "184.9 KB", "1.22 / 3.75 / 14.55",
                           "1.24 / 4.73")))
check("PITCH quotes official Scania + metro negative + latency",
      all(t in p for t in ("18645", "31340", "32355", "0/4", "5.5–17.3", "1.74",
                          "2.21–62.35")))
check("EMAIL quotes official Scania + metro negative",
      all(t in e for t in ("18645", "31340", "0/4", "4/4")))
check("paper quotes corrected latency numbers",
      all(t in tex for t in ("2.21--62.35", "2{,}808", "62.35")))
check("ARTIFACTS lists all four new artifacts",
      all(t in art for t in ("bench_metro_watch_v2_results.json",
                             "bench_edge_envelope_v2_results.json",
                             "bench_latency_v2_results.json",
                             "bench_scania_cost_official_results.json")))

# --------------------------------------------- historical artifacts untouched
section("Historical artifacts not rewritten by the repair")

cutoff = datetime(2026, 10, 20).timestamp()
for name in ("bench_latency_results.json", "bench_metro_watch_results.json",
             "bench_scania_cost_results.json", "bench_edge_envelope_results.json",
             "bench_seeds_results.json", "metro_demo_crystal.hologram.npz"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.exists(path):
        check("historical {} exists".format(name), False, "missing")
        continue
    mtime = os.path.getmtime(path)
    when = datetime.fromtimestamp(mtime).isoformat(timespec="minutes")
    check("historical {} predates today's repair".format(name), mtime < cutoff,
          "mtime={}".format(when))

for name in ("bench_metro_watch_v2_results.json", "metro_demo_v2_crystal.hologram.npz",
             "bench_edge_envelope_v2_results.json", "bench_latency_v2_results.json",
             "bench_scania_cost_official_results.json"):
    check("fresh artifact {} exists".format(name),
          os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), name)))

# ------------------------------------------------------------------- conclusion
print("\n" + "=" * 90)
print("RESULT: {} failures".format(len(FAILURES)))
if FAILURES:
    for f in FAILURES:
        print("  FAIL:", f)
print("=" * 90)
sys.exit(1 if FAILURES else 0)

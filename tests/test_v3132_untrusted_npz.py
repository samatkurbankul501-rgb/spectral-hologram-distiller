"""v3.13.2: thread-model for untrusted .hologram.npz artifacts.

The loader is the package's boundary with the outside world -- crystals travel
by email, pull requests and model marketplaces. This file pins the structural
validation contract added on top of ``np.load(allow_pickle=False)``:

* a poisoned artifact must fail HERE, in the loader, with an "artifact is
  inconsistent" ValueError -- not deep inside a downstream fit with a cryptic
  shape/NaN error;
* every leg of the passport / crystal contract is checked once, with the exact
  wording of the error pinned (tests fail loudly if the message drifts);
* legit artifacts (plain, delta, legacy) still load: the checks must reject
  lies, not the truth;
* the archive handle is released inside ``__init__`` (on Windows an open
  ``NpzFile`` keeps the .npz locked, which broke TemporaryDirectory cleanup).
"""

import os
import sys
import tempfile

import numpy as np
import pytest

from shd import SpectralHologramDistiller, HologramCrystalLoader


def _crystal(include_deltas=False, n=200, d=3):
    """Distill a small legit crystal and save it to a temp .hologram.npz."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n, d))
    y = X @ np.arange(1.0, d + 1.0)
    dist = SpectralHologramDistiller(
        crystal_size=20,
        sketch_dim=16,
        random_state=0,
        include_deltas=include_deltas,
        max_init_buffer=max(n, 400),
    )
    for i in range(0, n, 200):
        dist.partial_fit(X[i : i + 200], y[i : i + 200])
    dist.finalize_crystal()
    path = os.path.join(
        tempfile.mkdtemp(prefix="shd_untrusted_"), "c.hologram.npz"
    )
    dist.save_hologram(path)
    return path


def _payload(path):
    """Read the full payload of an .npz as a plain dict (handle closed)."""
    src = np.load(path, allow_pickle=False)
    data = {key: np.asarray(src[key]) for key in src.files}
    src.close()
    return data


def _modified(path, key, value, name="mod.hologram.npz"):
    """Re-save the artifact with one payload array replaced (or dropped)."""
    payload = _payload(path)
    if value is None:
        payload.pop(key, None)
    else:
        payload[key] = value
    out = os.path.join(tempfile.mkdtemp(prefix="shd_untrusted_"), name)
    np.savez_compressed(out, **payload)
    return out


def _assert_rejected(path, match):
    with pytest.raises(ValueError, match=match):
        HologramCrystalLoader(path)


# --------------------------------------------------------------------------
# Legit crystals still load: the validation rejects lies, not the truth.
# --------------------------------------------------------------------------


def test_plain_and_delta_and_legacy_still_load():
    plain = HologramCrystalLoader(_crystal())
    delta = HologramCrystalLoader(_crystal(include_deltas=True))
    assert plain.X.shape[1] == 3
    assert plain.X.shape[0] >= 1
    assert delta.include_deltas is True
    # a legacy pre-3.13.1 crystal (no schema/shd version keys) still loads, with
    # the provenance warning the schema contract prescribes
    stripped = _modified(
        _modified(_crystal(), "schema_version", None), "shd_version", None
    )
    with pytest.warns(UserWarning, match="no schema_version key"):
        legacy = HologramCrystalLoader(stripped)
    assert legacy.schema_version == 1
    assert legacy.shd_version is None


def test_loader_releases_npz_handle():
    """The archive handle is released inside __init__.

    Observable on Windows: an OPEN NpzFile keeps the .npz locked and
    ``os.replace`` fails with WinError 32. The first block proves the lock
    mechanism actually fires on this platform (guards the test itself), the
    second proves the loader is not holding it.
    """
    path = _crystal()
    moved = path + ".moved"
    if sys.platform == "win32":
        src = np.load(path, allow_pickle=False)
        with pytest.raises(OSError):
            os.replace(path, moved)
        src.close()
        os.replace(path, moved)  # now it moves -- the lock was the NpzFile
        os.replace(moved, path)

    HologramCrystalLoader(path)
    os.replace(path, moved)
    os.replace(moved, path)


def test_loader_releases_npz_handle_on_validation_error():
    """A rejected artifact must not leak the handle either."""
    path = _modified(
        _crystal(), "gamma_weights", -np.ones(20, dtype=np.float32)
    )
    moved = path + ".moved"
    with pytest.raises(ValueError, match="inconsistent"):
        HologramCrystalLoader(path)
    os.replace(path, moved)
    os.replace(moved, path)


# --------------------------------------------------------------------------
# Poisoned artifacts must fail in the loader, with a pinned message.
# --------------------------------------------------------------------------


def test_rejects_nan_in_crystal_x():
    path = _crystal()
    payload = _payload(path)
    payload["crystal_X"] = payload["crystal_X"].copy()
    payload["crystal_X"][0, 0] = np.nan
    _assert_rejected(
        _modified(path, "crystal_X", payload["crystal_X"]),
        "non-finite values in crystal_X",
    )


def test_rejects_mismatched_y_length():
    path = _crystal()
    payload = _payload(path)
    _assert_rejected(
        _modified(path, "crystal_Y", payload["crystal_Y"][:-1]),
        "crystal_Y of shape",
    )


def test_rejects_mismatched_weights_length():
    _assert_rejected(
        _modified(_crystal(), "gamma_weights", np.ones(3, dtype=np.float32)),
        "gamma_weights of shape",
    )


def test_rejects_negative_weights_and_std():
    _assert_rejected(
        _modified(
            _crystal(), "gamma_weights", -np.ones(20, dtype=np.float32)
        ),
        "negative gamma_weights",
    )
    path = _crystal()
    payload = _payload(path)
    payload["X_std"] = payload["X_std"].copy()
    payload["X_std"][0] = -1.0
    _assert_rejected(
        _modified(path, "X_std", payload["X_std"]), "negative X_std"
    )


def test_rejects_wrong_passport_shape():
    _assert_rejected(
        _modified(_crystal(), "passport_eigvals", np.ones(5)),
        "passport must be eigvals \\(rank,\\) \\+ eigvecs \\(sketch_dim, rank\\)",
    )
    _assert_rejected(
        _modified(_crystal(), "passport_eigvecs", np.ones((16, 6))),
        "passport must be eigvals \\(rank,\\) \\+ eigvecs \\(sketch_dim, rank\\)",
    )


def test_rejects_wrong_mean_width():
    _assert_rejected(
        _modified(_crystal(), "X_mean", np.ones(7, dtype=np.float32)),
        "expected 3 for a plain crystal or 6 for a delta one",
    )


def test_rejects_zero_rows():
    _assert_rejected(
        _modified(_crystal(), "crystal_X", np.empty((0, 3), dtype=np.float32)),
        "a crystal must be a non-empty \\(rows, features\\) matrix",
    )


def test_rejects_degenerate_n_seen():
    _assert_rejected(
        _modified(_crystal(), "n_seen", np.array([0], dtype=np.int64)),
        "n_seen=0",
    )
    _assert_rejected(
        _modified(_crystal(), "n_seen", np.array([-5], dtype=np.int64)),
        "n_seen=-5",
    )

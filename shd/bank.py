"""Domain bank: route windows to one of several per-domain SHD crystals.

A single crystal healed on a new regime must trade old-domain retention against
recognizing the new domain: a blended spectral passport can leave *both* domains
above the 0.085 alert threshold, and a replaced passport flags the old domain.
A bank keeps one crystal (and one passport) per regime, so every domain remains
in-distribution and no medoids are overwritten. Routing uses the same
Bures-Wasserstein distance as the single-crystal radar: pick the crystal with
the smallest distance; the window is shifted only if even the closest crystal
is above its threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from .distiller import HologramCrystalLoader


def _loader_widths(loader: HologramCrystalLoader) -> set[int]:
    """Feature widths a loader accepts for a stream window.

    The loader's own contract is the width of ``X_mean`` (d for a plain
    crystal, 2d for a delta one) -- NOT ``loader.X.shape[1]``, which is
    the raw medoid width even for a delta crystal. v3.15.1: the bank used
    the raw width, so for a legacy delta artifact (include_deltas
    postdates the file) it accepted windows the loader rejects and
    rejected the width the loader actually takes.
    """
    widths = {int(len(loader.X_mean))}
    if loader.include_deltas:
        d_raw = int(loader.X.shape[1])
        widths |= {d_raw, 2 * d_raw}
    return widths


class DomainBank:
    """A collection of per-domain crystals with nearest-passport routing.

    Parameters
    ----------
    crystals:
        Optional initial crystals, each either a :class:`HologramCrystalLoader`
        or a path to a ``.hologram.npz`` file.

    Examples
    --------
    >>> bank = DomainBank(["a.hologram.npz", "b.hologram.npz"])
    >>> loader = bank.select(X_new)          # nearest spectral passport
    >>> X, y, w = loader.get_training_data() # that domain's crystal
    """

    def __init__(self, crystals: Iterable[HologramCrystalLoader | str] | None = None):
        self.loaders: list[HologramCrystalLoader] = []
        for crystal in crystals or ():
            self.add(crystal)

    def add(self, crystal: HologramCrystalLoader | str) -> HologramCrystalLoader:
        """Add a loader (or path) and return the stored loader."""
        loader = crystal if isinstance(crystal, HologramCrystalLoader) else HologramCrystalLoader(crystal)
        self.loaders.append(loader)
        return loader

    def __len__(self) -> int:
        return len(self.loaders)

    def __iter__(self) -> Iterator[HologramCrystalLoader]:
        return iter(self.loaders)

    def __getitem__(self, index: int) -> HologramCrystalLoader:
        return self.loaders[index]

    def check_domain_shift(self, X_new_stream, streaming: bool = False) -> dict:
        """Return the nearest-passport radar result.

        The returned ``is_shifted`` is True only when the *closest* domain is
        above its own threshold, i.e. the window matches no known regime.
        ``domain_index`` is the selected crystal; ``bw_distances`` exposes the
        full bank for diagnostics (``None`` for crystals whose feature width is
        incompatible with the window).

        Crystals whose expected feature width does not match the window are
        excluded from the ranking instead of aborting the whole bank: a mixed
        bank (e.g. a delta crystal stored next to plain ones) still routes any
        window that at least one crystal can score.
        """
        if not self.loaders:
            raise ValueError("DomainBank is empty; add at least one crystal")
        X = np.asarray(X_new_stream, dtype=np.float64)
        if X.ndim == 1:
            w = int(X.shape[0])
        elif X.ndim == 2:
            w = int(X.shape[1])
        else:
            w = None  # let each loader's own validation raise
        candidates = []
        for i, loader in enumerate(self.loaders):
            if w is None or w in _loader_widths(loader):
                candidates.append(i)
        if not candidates:
            widths = sorted({w_ for l in self.loaders
                             for w_ in _loader_widths(l)})
            raise ValueError(
                f"no crystal in the bank can score a {w}-feature window; "
                f"feature widths the bank understands: {widths}"
            )
        reports = {i: self.loaders[i].check_domain_shift(X, streaming=streaming)
                   for i in candidates}
        # Prefer a domain that ACCEPTS the window; only if none accepts, fall
        # back to the closest. Rank by relative margin bw/threshold, not raw bw:
        # BW distance grows ~sqrt(rank) and each threshold is rank-scaled or
        # calibrated, so raw distances are incomparable across a mixed-rank bank
        # (a low-rank crystal used to win on raw distance and report a shift even
        # though another domain accepted the window).
        accepting = [i for i in reports if not reports[i]["is_shifted"]]
        pool = accepting if accepting else list(reports)
        best = min(pool, key=lambda i: reports[i]["bw_distance"]
                   / max(reports[i]["threshold"], 1e-12))
        result = dict(reports[best])
        result["domain_index"] = best
        result["is_shifted"] = not accepting
        result["bw_distances"] = [reports[i]["bw_distance"] if i in reports else None
                                  for i in range(len(self.loaders))]
        return result

    def select(self, X_new_stream, streaming: bool = False) -> HologramCrystalLoader:
        """Return the crystal whose passport is closest to the window."""
        idx = self.check_domain_shift(X_new_stream, streaming=streaming)["domain_index"]
        return self.loaders[idx]

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

from .distiller import HologramCrystalLoader


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

    def check_domain_shift(self, X_new_stream) -> dict:
        """Return the nearest-passport radar result.

        The returned ``is_shifted`` is True only when the *closest* domain is
        above its own threshold, i.e. the window matches no known regime.
        ``domain_index`` is the selected crystal; ``bw_distances`` exposes the
        full bank for diagnostics.
        """
        if not self.loaders:
            raise ValueError("DomainBank is empty; add at least one crystal")
        reports = [loader.check_domain_shift(X_new_stream) for loader in self.loaders]
        best = min(range(len(reports)), key=lambda i: reports[i]["bw_distance"])
        result = dict(reports[best])
        result["domain_index"] = best
        result["bw_distances"] = [r["bw_distance"] for r in reports]
        return result

    def select(self, X_new_stream) -> HologramCrystalLoader:
        """Return the crystal whose passport is closest to the window."""
        return self.loaders[self.check_domain_shift(X_new_stream)["domain_index"]]

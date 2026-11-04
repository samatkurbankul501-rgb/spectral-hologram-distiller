from .distiller import (
    SHD_VERSION,
    HologramCrystalLoader,
    SpectralHologramDistiller,
)
from .bank import DomainBank
from .portfolio import CoresetPortfolio, ConformalCrystal
from .presets import PRESETS, apply_preset, validate_presets

#: Single source of truth lives in ``shd.distiller.SHD_VERSION``; it is
#: recorded into every .hologram.npz artifact at save time (v3.13.1).
__version__ = SHD_VERSION

__all__ = [
    "SpectralHologramDistiller",
    "HologramCrystalLoader",
    "DomainBank",
    "CoresetPortfolio",
    "ConformalCrystal",
    "PRESETS",
    "apply_preset",
    "validate_presets",
    "SHD_VERSION",
]

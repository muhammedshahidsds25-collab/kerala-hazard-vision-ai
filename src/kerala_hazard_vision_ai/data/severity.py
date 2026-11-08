"""Severity labels computed from the spectral change between before and after images.

Severity is the fraction of valid pixels in a tile that changed in the way the hazard implies:
vegetation loss (NDVI drop) for landslides, new surface water (NDWI crossing the water
threshold) for floods. Using a
measured label means no manual annotation, and the same function can score generated images.
"""

from __future__ import annotations

import numpy as np

from kerala_hazard_vision_ai.config import SeverityConfig

BANDS: tuple[str, ...] = ("B2", "B3", "B4", "B8")
HAZARD_CODES: dict[str, int] = {"flood": 0, "landslide": 1}

_GREEN, _RED, _NIR = BANDS.index("B3"), BANDS.index("B4"), BANDS.index("B8")


def _normalized_difference(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return (a - b) / (a + b + eps)


def ndvi(bands: np.ndarray, eps: float) -> np.ndarray:
    """Normalized Difference Vegetation Index, (NIR - red) / (NIR + red).

    Args:
        bands: Array of shape (4, H, W) in ``BANDS`` order.
        eps: Small constant that keeps the division finite for zero reflectance.

    Returns:
        Array of shape (H, W).
    """
    return _normalized_difference(bands[_NIR], bands[_RED], eps)


def ndwi(bands: np.ndarray, eps: float) -> np.ndarray:
    """McFeeters Normalized Difference Water Index, (green - NIR) / (green + NIR).

    Args:
        bands: Array of shape (4, H, W) in ``BANDS`` order.
        eps: Small constant that keeps the division finite for zero reflectance.

    Returns:
        Array of shape (H, W).
    """
    return _normalized_difference(bands[_GREEN], bands[_NIR], eps)


def change_mask(
    before: np.ndarray, after: np.ndarray, hazard: str, cfg: SeverityConfig
) -> np.ndarray:
    """Boolean (H, W) mask of pixels whose change matches the hazard.

    Pixels that are NaN in either image are never flagged.

    Raises:
        ValueError: If ``hazard`` is not one of ``HAZARD_CODES``.
    """
    if hazard == "landslide":
        return ndvi(before, cfg.eps) - ndvi(after, cfg.eps) > cfg.ndvi_drop
    if hazard == "flood":
        # Require a crossing into water: a plain NDWI rise also happens on vegetation loss.
        return (ndwi(before, cfg.eps) <= cfg.water_ndwi) & (ndwi(after, cfg.eps) > cfg.water_ndwi)
    raise ValueError(f"Unknown hazard '{hazard}', expected one of {list(HAZARD_CODES)}")


def severity(before: np.ndarray, after: np.ndarray, hazard: str, cfg: SeverityConfig) -> float:
    """Fraction of valid pixels flagged by :func:`change_mask`, in [0, 1].

    Args:
        before: Pre-event bands, shape (4, H, W).
        after: Post-event (real or generated) bands, shape (4, H, W).
        hazard: ``"flood"`` or ``"landslide"``.
        cfg: Change thresholds.

    Returns:
        Severity in [0, 1]; 0.0 when the tile has no valid pixels.
    """
    changed = change_mask(before, after, hazard, cfg)
    valid = np.isfinite(before).all(axis=0) & np.isfinite(after).all(axis=0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0
    return float((changed & valid).sum() / n_valid)

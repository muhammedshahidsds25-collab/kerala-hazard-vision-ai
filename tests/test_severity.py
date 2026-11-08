"""Tests for change-based severity labels."""

import numpy as np
import pytest

from kerala_hazard_vision_ai.data.severity import BANDS, HAZARD_CODES, ndvi, ndwi, severity

# Reflectance-like values in BANDS order (B2, B3, B4, B8).
VEGETATION = (0.05, 0.08, 0.05, 0.40)  # low red, high NIR: NDVI ~0.78
BARE_SOIL = (0.15, 0.18, 0.22, 0.25)  # NIR ~ red: NDVI ~0.06
DRY_LAND = (0.05, 0.08, 0.05, 0.40)  # NDWI ~ -0.67
WATER = (0.08, 0.10, 0.06, 0.03)  # green > NIR: NDWI ~0.54


def tile(values, size=8):
    return np.stack([np.full((size, size), v, dtype=np.float32) for v in values])


def test_band_order_and_hazard_codes():
    assert BANDS == ("B2", "B3", "B4", "B8")
    assert HAZARD_CODES == {"flood": 0, "landslide": 1}


def test_no_change_is_zero(cfg):
    veg = tile(VEGETATION)
    assert severity(veg, veg, "landslide", cfg.severity) == 0.0


def test_full_vegetation_loss_is_one(cfg):
    sev = severity(tile(VEGETATION), tile(BARE_SOIL), "landslide", cfg.severity)
    assert sev == pytest.approx(1.0)


def test_flood_uses_ndwi_rise(cfg):
    assert severity(tile(DRY_LAND), tile(WATER), "flood", cfg.severity) == pytest.approx(1.0)
    # Vegetation loss alone is not a flood.
    assert severity(tile(VEGETATION), tile(BARE_SOIL), "flood", cfg.severity) == 0.0


def test_half_tile_change_is_half(cfg):
    after = tile(VEGETATION)
    after[:, :, :4] = tile(BARE_SOIL)[:, :, :4]
    assert severity(tile(VEGETATION), after, "landslide", cfg.severity) == pytest.approx(0.5)


def test_nan_pixels_are_ignored(cfg):
    after = tile(BARE_SOIL)
    after[:, :, :4] = np.nan
    assert severity(tile(VEGETATION), after, "landslide", cfg.severity) == pytest.approx(1.0)


def test_zero_reflectance_no_nan(cfg):
    zeros = np.zeros((4, 8, 8), dtype=np.float32)
    assert np.isfinite(ndvi(zeros, cfg.severity.eps)).all()
    assert np.isfinite(ndwi(zeros, cfg.severity.eps)).all()


def test_unknown_hazard_raises(cfg):
    veg = tile(VEGETATION)
    with pytest.raises(ValueError, match="tsunami"):
        severity(veg, veg, "tsunami", cfg.severity)

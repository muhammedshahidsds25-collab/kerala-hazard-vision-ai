"""Tests for the pure (Earth Engine-free) parts of the export module."""

import sys
from datetime import date

import pytest

from disastergen.config import DataConfig
from disastergen.data.gee_export import composite_windows, dry_season_before


def test_composite_windows_are_matching_dry_seasons(cfg):
    before, after = composite_windows(date(2020, 8, 6), cfg.data)  # Pettimudi, monsoon
    assert before == (date(2019, 12, 1), date(2020, 4, 30))
    assert after == (date(2020, 12, 1), date(2021, 4, 30))


def test_composite_windows_october_event(cfg):
    before, after = composite_windows(date(2021, 10, 16), cfg.data)  # Kokkayar
    assert before == (date(2020, 12, 1), date(2021, 4, 30))
    assert after == (date(2021, 12, 1), date(2022, 4, 30))


def test_composite_windows_event_inside_dry_season(cfg):
    before, after = composite_windows(date(2021, 2, 10), cfg.data)
    assert before[1] < date(2021, 2, 10) < after[0]


def test_dry_season_before(cfg):
    assert dry_season_before(date(2026, 11, 8), cfg.data) == (date(2025, 12, 1), date(2026, 4, 30))
    assert dry_season_before(date(2026, 3, 1), cfg.data) == (date(2024, 12, 1), date(2025, 4, 30))


def test_dry_season_must_cross_new_year():
    with pytest.raises(ValueError, match="New Year"):
        DataConfig(dry_season_start_month=1, dry_season_end_month=4)


def test_module_imports_without_earth_engine():
    # CI has no earthengine-api: importing the module must not import `ee`.
    assert "ee" not in sys.modules

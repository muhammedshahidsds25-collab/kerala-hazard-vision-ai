"""Tests for the pure (Earth Engine-free) parts of the export module."""

import sys
from datetime import date

from disastergen.data.gee_export import composite_windows


def test_composite_windows(cfg):
    (ps, pe), (qs, qe) = composite_windows(date(2020, 8, 6), cfg.data)
    assert pe < date(2020, 8, 6) < qs
    assert (pe - ps).days == cfg.data.pre_window_days - 1
    assert (qs - date(2020, 8, 6)).days == cfg.data.post_offset_days
    assert (qe - qs).days == cfg.data.post_window_days


def test_module_imports_without_earth_engine():
    # CI has no earthengine-api: importing the module must not import `ee`.
    assert "ee" not in sys.modules

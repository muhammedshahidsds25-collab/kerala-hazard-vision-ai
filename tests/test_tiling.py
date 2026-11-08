"""Tests for tiling event stacks into training samples."""

from datetime import date

import numpy as np
import pytest

from kerala_hazard_vision_ai.config import Event
from kerala_hazard_vision_ai.data.tiling import (
    STACK_CHANNELS,
    fill_missing,
    iter_tiles,
    split_stack,
    write_event_tiles,
)

EVENT = Event("test_slide", "landslide", date(2020, 8, 6), (76.0, 10.0, 76.1, 10.1))


def test_stack_has_eleven_channels():
    assert STACK_CHANNELS == 11


def test_split_stack_shapes():
    before, after, cond = split_stack(np.zeros((11, 8, 8)))
    assert before.shape == (4, 8, 8) and after.shape == (4, 8, 8) and cond.shape == (3, 8, 8)


def test_split_stack_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="11"):
        split_stack(np.zeros((9, 8, 8)))


def test_iter_tiles_count():
    assert len(list(iter_tiles(np.zeros((11, 512, 768)), 256, 256, 0.2))) == 6


def test_iter_tiles_overlapping_stride():
    tiles = list(iter_tiles(np.zeros((11, 512, 512)), 256, 128, 0.2))
    assert len(tiles) == 9 and (tiles[-1][0], tiles[-1][1]) == (256, 256)


def test_tiles_with_nodata_are_dropped_or_filled():
    stack = np.ones((11, 256, 512))
    stack[:, :, :256] = np.nan  # left tile fully masked: dropped
    stack[:, 0, 300] = np.nan  # one pixel in the right tile: kept, then filled
    tiles = list(iter_tiles(stack, 256, 256, 0.2))
    assert len(tiles) == 1 and np.isfinite(fill_missing(tiles[0][2])).all()


def test_severity_ignores_cloud_filled_pixels(tmp_path, cfg):
    # Before: all vegetated. After: 38 cloud columns, 154 bare, 64 still vegetated.
    stack = np.zeros((11, 256, 256), dtype=np.float32)
    stack[:4] = np.array([0.05, 0.08, 0.05, 0.40])[:, None, None]
    stack[4:8] = np.array([0.05, 0.08, 0.05, 0.40])[:, None, None]
    stack[4:8, :, 38:192] = np.array([0.15, 0.18, 0.22, 0.25])[:, None, None]
    stack[4:8, :, :38] = np.nan
    write_event_tiles(stack, EVENT, tmp_path, cfg)
    z = np.load(next(tmp_path.glob("*.npz")))
    # Only the 218 cloud-free columns count; filling clouds with the (bare) median would give 0.75.
    assert float(z["severity"]) == pytest.approx(154 / 218)
    assert np.isfinite(z["after"]).all()


def test_write_event_tiles_roundtrip(tmp_path, cfg):
    stack = np.full((11, 256, 256), 0.1, dtype=np.float32)
    stack[3] = 0.4  # before NIR high: vegetated
    stack[7] = 0.1  # after NIR ~ red: bare
    n = write_event_tiles(stack, EVENT, tmp_path, cfg)
    files = list(tmp_path.glob("*.npz"))
    assert n == 1 and [f.name for f in files] == ["test_slide_r0_c0.npz"]
    z = np.load(files[0])
    assert z["before"].shape == (4, 256, 256) and z["cond"].shape == (3, 256, 256)
    assert z["before"].dtype == np.float32
    assert float(z["severity"]) == pytest.approx(1.0) and int(z["hazard"]) == 1
    assert str(z["event"]) == "test_slide"

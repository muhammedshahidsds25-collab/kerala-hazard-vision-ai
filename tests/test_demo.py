"""Tests for the interactive demo helpers (map and Earth Engine parts run only in Colab)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from kerala_hazard_vision_ai.data.dataset import ChannelStats
from kerala_hazard_vision_ai.demo import (
    crossfade,
    generate_sweep,
    is_extrapolation,
    rain_plane_cond,
    validate_area,
)
from kerala_hazard_vision_ai.models import UNetGenerator

STATS = ChannelStats(np.full(7, 0.1, np.float32), np.full(7, 0.05, np.float32))


@pytest.fixture
def tiny_g():
    torch.manual_seed(0)
    return UNetGenerator(9, 4, ngf=8, num_downs=6).eval()


def inputs(height, width):
    rng = np.random.default_rng(0)
    before = rng.uniform(0, 0.3, (4, height, width)).astype(np.float32)
    cond = rng.uniform(0, 30, (3, height, width)).astype(np.float32)
    return before, cond


def test_validate_area_rejects_outside_kerala_and_too_large(cfg):
    with pytest.raises(ValueError, match="Kerala"):
        validate_area((80.0, 13.0, 80.1, 13.1), cfg.demo)
    with pytest.raises(ValueError, match="km"):
        validate_area((76.0, 10.0, 76.5, 10.5), cfg.demo)
    validate_area((76.13, 11.46, 76.15, 11.48), cfg.demo)  # Wayanad, ~4.8 km²: accepted


def test_validate_area_rejects_inverted_bbox(cfg):
    with pytest.raises(ValueError, match="min"):
        validate_area((76.15, 11.48, 76.13, 11.46), cfg.demo)


def test_generate_sweep_frames(tiny_g):
    before, cond = inputs(64, 64)
    frames = generate_sweep(tiny_g, before, cond, 1, [0.1, 0.6], STATS, rgb_max=0.3, tile=64)
    assert len(frames) == 2 and frames[0].dtype == np.uint8 and frames[0].shape == (64, 64, 3)
    assert not np.array_equal(frames[0], frames[1])  # severity changes the output


def test_generate_sweep_non_multiple_size(tiny_g):
    before, cond = inputs(100, 150)
    frames = generate_sweep(tiny_g, before, cond, 1, [0.5], STATS, rgb_max=0.3, tile=64)
    assert frames[0].shape == (100, 150, 3)


def test_generate_sweep_rejects_bad_severity(tiny_g):
    before, cond = inputs(64, 64)
    with pytest.raises(ValueError, match="severity"):
        generate_sweep(tiny_g, before, cond, 1, [1.2], STATS, rgb_max=0.3, tile=64)


def test_is_extrapolation(cfg):
    demo = replace(cfg.demo, train_severity_max=0.8)
    assert is_extrapolation(0.95, demo) and not is_extrapolation(0.5, demo)


def test_crossfade_endpoints_and_length():
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 200, np.uint8)
    frames = crossfade([a, b], steps=5)
    assert len(frames) == 5 and np.array_equal(frames[0], a) and np.array_equal(frames[-1], b)
    assert frames[2].dtype == np.uint8 and frames[2].mean() == pytest.approx(100, abs=1)


def test_rain_plane_cond():
    slope, soil = np.ones((5, 6)), np.full((5, 6), 3.0)
    cond = rain_plane_cond(slope, soil, rain_mm=250.0)
    assert cond.shape == (3, 5, 6) and cond.dtype == np.float32
    assert (cond[1] == 250.0).all() and (cond[0] == 1).all() and (cond[2] == 3).all()

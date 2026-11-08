"""Helpers for the interactive demo: area checks, data fetching and severity sweeps.

The user draws an area in Kerala; the demo fetches its latest dry-season Sentinel-2 image,
slope and soil from Earth Engine, takes the rainfall scenario from a slider, and generates the
after image at several severities, which the notebook crossfades.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import torch

from disastergen.config import Config, DemoConfig
from disastergen.data.dataset import ChannelStats, build_input, to_reflectance
from disastergen.evaluate import rgb_uint8
from disastergen.models import UNetGenerator

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_AT_EQUATOR = 111.320

BBox = tuple[float, float, float, float]


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate area of a (min_lon, min_lat, max_lon, max_lat) box in km²."""
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    width = (max_lon - min_lon) * KM_PER_DEG_LON_AT_EQUATOR * math.cos(mid_lat)
    return width * (max_lat - min_lat) * KM_PER_DEG_LAT


def validate_area(bbox: BBox, cfg: DemoConfig) -> None:
    """Check that a user-drawn area is valid, inside Kerala and small enough.

    Raises:
        ValueError: With a message the demo shows to the user.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Bounding box must be (min_lon, min_lat, max_lon, max_lat), got {bbox}")
    k_min_lon, k_min_lat, k_max_lon, k_max_lat = cfg.kerala_bbox
    inside_lon = k_min_lon <= min_lon and max_lon <= k_max_lon
    inside_lat = k_min_lat <= min_lat and max_lat <= k_max_lat
    if not (inside_lon and inside_lat):
        raise ValueError("The area must be inside Kerala: the model was trained on Kerala only.")
    area = bbox_area_km2(bbox)
    if area > cfg.max_demo_area_km2:
        raise ValueError(
            f"The area is {area:.1f} km²; draw one of at most {cfg.max_demo_area_km2:.0f} km²."
        )


def is_extrapolation(severity: float, cfg: DemoConfig) -> bool:
    """True if the severity is above the highest severity seen in training."""
    return severity > cfg.train_severity_max


def rain_plane_cond(slope: np.ndarray, soil: np.ndarray, rain_mm: float) -> np.ndarray:
    """Conditioning stack (slope, rain, soil) with a constant rainfall scenario in mm."""
    return np.stack([slope, np.full_like(slope, rain_mm), soil]).astype(np.float32)


@torch.no_grad()
def generate_sweep(
    g: UNetGenerator,
    before: np.ndarray,
    cond: np.ndarray,
    hazard: int,
    severities: Sequence[float],
    stats: ChannelStats,
    rgb_max: float,
    tile: int,
    device: torch.device | None = None,
) -> list[np.ndarray]:
    """Generate one (H, W, 3) uint8 RGB frame per severity for an area of any size.

    The area is reflect-padded to a multiple of ``tile``, generated tile by tile (the size the
    model was trained on), then cropped back.

    Raises:
        ValueError: If a severity is outside [0, 1].
    """
    device = device or next(g.parameters()).device
    g = g.eval()
    height, width = before.shape[1:]
    pad = ((0, 0), (0, -height % tile), (0, -width % tile))
    before_p, cond_p = np.pad(before, pad, mode="reflect"), np.pad(cond, pad, mode="reflect")
    rows, cols = before_p.shape[1] // tile, before_p.shape[2] // tile

    frames = []
    for severity in severities:
        x = build_input(before_p, cond_p, severity, hazard, stats)
        chunks = x.unfold(1, tile, tile).unfold(2, tile, tile)  # (9, rows, cols, tile, tile)
        batch = chunks.permute(1, 2, 0, 3, 4).reshape(rows * cols, x.shape[0], tile, tile)
        out = to_reflectance(g(batch.to(device)).cpu(), stats)  # (rows*cols, 4, tile, tile)
        mosaic = out.reshape(rows, cols, 4, tile, tile).permute(2, 0, 3, 1, 4)
        mosaic = mosaic.reshape(4, rows * tile, cols * tile)[:, :height, :width]
        frames.append(rgb_uint8(mosaic, rgb_max).permute(1, 2, 0).numpy())
    return frames


def crossfade(frames: Sequence[np.ndarray], steps: int) -> list[np.ndarray]:
    """``steps`` evenly spaced frames blending linearly through ``frames`` in order."""
    stack = np.stack(frames).astype(np.float32)
    positions = np.linspace(0, len(frames) - 1, steps)
    out = []
    for pos in positions:
        low = int(np.floor(pos))
        high = min(low + 1, len(frames) - 1)
        alpha = pos - low
        out.append(np.round((1 - alpha) * stack[low] + alpha * stack[high]).astype(np.uint8))
    return out


def fetch_area(bbox: BBox, cfg: Config, today: date | None = None) -> tuple[np.ndarray, ...]:
    """Download (before (4, H, W), slope (H, W), soil (H, W)) for an area from Earth Engine.

    Needs ``ee.Initialize`` and the ``gee`` extra. Cloud gaps are filled with the area median.
    """
    import geemap

    from disastergen.data import gee_export as gx

    validate_area(bbox, cfg.demo)
    region = gx.aoi(bbox)
    # Same season definition as the training before-images, so the input looks familiar.
    start, end = gx.dry_season_before(today or date.today(), cfg.data)
    image = (
        gx.s2_composite(region, start, end, cfg.data.max_cloud_pct)
        .addBands(gx.slope_image(region))
        .addBands(gx.soil_image())
        .toFloat()
        .unmask(gx.NODATA)
    )
    array = geemap.ee_to_numpy(image, region=region, scale=cfg.data.scale_m)
    stack = np.moveaxis(np.asarray(array, np.float32), -1, 0)
    stack[stack == gx.NODATA] = np.nan
    medians = np.nanmedian(stack.reshape(stack.shape[0], -1), axis=1)
    stack = np.where(np.isfinite(stack), stack, medians[:, None, None])
    return stack[:4], stack[4], stack[5]

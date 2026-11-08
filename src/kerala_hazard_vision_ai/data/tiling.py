"""Cut a co-registered event stack into fixed-size training tiles.

An event stack is an (11, H, W) array exported from Earth Engine:
4 before bands, 4 after bands (both in ``BANDS`` order), then slope, rain and soil.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from kerala_hazard_vision_ai.config import Config, Event
from kerala_hazard_vision_ai.data.severity import HAZARD_CODES, severity

logger = logging.getLogger(__name__)

N_BANDS = 4
N_COND = 3
STACK_CHANNELS = 2 * N_BANDS + N_COND


def split_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an event stack into (before, after, conditioning) arrays.

    Raises:
        ValueError: If the stack does not have ``STACK_CHANNELS`` channels.
    """
    if stack.shape[0] != STACK_CHANNELS:
        raise ValueError(f"Expected {STACK_CHANNELS} channels, got {stack.shape[0]}")
    return stack[:N_BANDS], stack[N_BANDS : 2 * N_BANDS], stack[2 * N_BANDS :]


def iter_tiles(
    stack: np.ndarray, size: int, stride: int, max_nodata_frac: float
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(row, col, tile)`` for every full tile with few enough missing pixels.

    A pixel counts as missing if any channel is NaN (e.g. cloud-masked). Tiles with more than
    ``max_nodata_frac`` missing pixels are skipped. Yielded tiles may still contain NaN; see
    :func:`fill_missing`.
    """
    _, height, width = stack.shape
    for row in range(0, height - size + 1, stride):
        for col in range(0, width - size + 1, stride):
            tile = stack[:, row : row + size, col : col + size]
            if (~np.isfinite(tile).all(axis=0)).mean() <= max_nodata_frac:
                yield row, col, tile


def fill_missing(tile: np.ndarray) -> np.ndarray:
    """Replace missing pixels with the per-channel median of the valid pixels in the tile."""
    missing = ~np.isfinite(tile).all(axis=0)
    if not missing.any():
        return tile
    medians = np.nanmedian(tile[:, ~missing], axis=1)
    return np.where(np.isfinite(tile), tile, medians[:, None, None])


def write_event_tiles(stack: np.ndarray, event: Event, out_dir: Path, cfg: Config) -> int:
    """Tile one event stack and save each tile as a compressed ``.npz`` file.

    Each file holds ``before``, ``after``, ``cond`` (float32 arrays), ``hazard`` (int code),
    ``severity`` (float in [0, 1]) and ``event`` (name). Files are named
    ``{event}_r{row}_c{col}.npz``.

    Returns:
        Number of tiles written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    data = cfg.data
    for row, col, tile in iter_tiles(stack, data.tile_size, data.stride, data.max_nodata_frac):
        # Label the raw tile: cloud gaps must not count as change once they are filled.
        label = severity(*split_stack(tile)[:2], event.hazard, cfg.severity)
        before, after, cond = split_stack(fill_missing(tile).astype(np.float32))
        np.savez_compressed(
            out_dir / f"{event.name}_r{row}_c{col}.npz",
            before=before,
            after=after,
            cond=cond,
            hazard=HAZARD_CODES[event.hazard],
            severity=label,
            event=event.name,
        )
        count += 1
    logger.info("%s: wrote %d tiles to %s", event.name, count, out_dir)
    return count

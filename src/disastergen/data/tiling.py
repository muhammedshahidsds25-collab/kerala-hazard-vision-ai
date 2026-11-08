"""Cut a co-registered event stack into fixed-size training tiles.

An event stack is an (11, H, W) array exported from Earth Engine:
4 before bands, 4 after bands (both in ``BANDS`` order), then slope, rain and soil.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from disastergen.config import Config, Event
from disastergen.data.severity import HAZARD_CODES, severity

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

    A pixel counts as missing if any channel is NaN (e.g. cloud-masked). Tiles above
    ``max_nodata_frac`` are skipped; the remaining gaps are filled with the per-channel
    median of the tile so no NaN ever reaches the model.
    """
    _, height, width = stack.shape
    for row in range(0, height - size + 1, stride):
        for col in range(0, width - size + 1, stride):
            tile = stack[:, row : row + size, col : col + size]
            missing = ~np.isfinite(tile).all(axis=0)
            if missing.mean() > max_nodata_frac:
                continue
            if missing.any():
                medians = np.nanmedian(tile[:, ~missing], axis=1)
                tile = np.where(np.isfinite(tile), tile, medians[:, None, None])
            yield row, col, tile


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
        before, after, cond = split_stack(tile.astype(np.float32))
        np.savez_compressed(
            out_dir / f"{event.name}_r{row}_c{col}.npz",
            before=before,
            after=after,
            cond=cond,
            hazard=HAZARD_CODES[event.hazard],
            severity=severity(before, after, event.hazard, cfg.severity),
            event=event.name,
        )
        count += 1
    logger.info("%s: wrote %d tiles to %s", event.name, count, out_dir)
    return count

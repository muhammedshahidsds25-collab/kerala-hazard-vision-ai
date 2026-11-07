"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pytest

from disastergen.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def default_config_path() -> Path:
    """Path to ``configs/default.yaml``."""
    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The project's default configuration."""
    return load_config(DEFAULT_CONFIG)


def write_tile(directory, event, row, col, sev, rng, size=64, same=False):
    """Write one synthetic tile in the format produced by ``tiling.write_event_tiles``."""
    before = rng.uniform(0.0, 0.4, (4, size, size)).astype(np.float32)
    after = before.copy() if same else rng.uniform(0.0, 0.4, (4, size, size)).astype(np.float32)
    cond = rng.uniform(0.0, 30.0, (3, size, size)).astype(np.float32)
    path = directory / f"{event}_r{row}_c{col}.npz"
    np.savez(path, before=before, after=after, cond=cond, hazard=1, severity=sev, event=event)
    return path


@pytest.fixture
def tile_paths(tmp_path):
    """30 synthetic 64×64 tiles: events a, b_2020, c with 10 tiles each, 2 severe per event."""
    rng = np.random.default_rng(0)
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    return [
        write_tile(tile_dir, event, 0, i * 128, 0.5 if i < 2 else 0.0, rng)
        for event in ("a", "b_2020", "c")
        for i in range(10)
    ]

"""Shared pytest fixtures."""

from pathlib import Path

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

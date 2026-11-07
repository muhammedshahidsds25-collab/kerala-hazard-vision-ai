"""Typed project configuration loaded from YAML.

All tunable numbers live in ``configs/default.yaml`` and ``configs/events.yaml``; the code reads
them through the frozen dataclasses below so that a typo in a key fails loudly instead of being
silently ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

import yaml

HAZARDS = ("flood", "landslide")

T = TypeVar("T")


@dataclass(frozen=True)
class Event:
    """A historical disaster used as a source of before/after image pairs.

    Attributes:
        name: Unique identifier, used in tile file names (e.g. ``"wayanad_2024"``).
        hazard: Either ``"flood"`` or ``"landslide"``.
        date: Day the event happened.
        bbox: Area of interest as (min_lon, min_lat, max_lon, max_lat) in degrees.
    """

    name: str
    hazard: str
    date: date
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class DataConfig:
    """Earth Engine export and tiling settings."""

    tile_size: int = 256
    stride: int = 256
    max_nodata_frac: float = 0.2
    pre_window_days: int = 120
    post_window_days: int = 150
    post_offset_days: int = 90
    max_cloud_pct: int = 40
    scale_m: int = 10
    rain_days_before: int = 7
    drive_folder: str = "disastergen"
    test_event: str = "pettimudi_2020"


@dataclass(frozen=True)
class SeverityConfig:
    """Thresholds that turn spectral change into a 0-1 severity label."""

    ndvi_drop: float = 0.25
    water_ndwi: float = 0.0
    eps: float = 1e-6


@dataclass(frozen=True)
class ModelConfig:
    """pix2pix generator and discriminator sizes."""

    in_channels: int = 9
    out_channels: int = 4
    ngf: int = 64
    ndf: int = 64
    num_downs: int = 8


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings (pix2pix defaults from Isola et al. 2017)."""

    epochs: int = 100
    batch_size: int = 16
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    l1_weight: float = 100.0
    seed: int = 42
    severe_threshold: float = 0.05
    val_frac: float = 0.1
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class DemoConfig:
    """Interactive demo limits."""

    kerala_bbox: tuple[float, float, float, float] = (74.85, 8.17, 77.42, 12.80)
    max_demo_area_km2: float = 25.0
    sweep: tuple[float, ...] = (0.1, 0.3, 0.6, 1.0)
    train_severity_max: float = 1.0


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    events: tuple[Event, ...] = ()


def _build(cls: type[T], raw: dict[str, Any] | None, section: str) -> T:
    """Instantiate a config dataclass, rejecting unknown keys and turning lists into tuples."""
    raw = raw or {}
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"Unknown key(s) in config section '{section}': {sorted(unknown)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**values)


def _build_event(raw: dict[str, Any]) -> Event:
    """Validate one entry of ``events.yaml``."""
    event = _build(Event, raw, "events")
    if event.hazard not in HAZARDS:
        raise ValueError(f"Event '{event.name}': hazard '{event.hazard}' not in {HAZARDS}")
    if len(event.bbox) != 4:
        raise ValueError(f"Event '{event.name}': bbox needs 4 numbers, got {event.bbox}")
    return event


def load_config(path: str | Path) -> Config:
    """Load the main config file and the ``events.yaml`` file next to it.

    Args:
        path: Path to the main YAML file (normally ``configs/default.yaml``).

    Returns:
        The validated, immutable configuration.

    Raises:
        ValueError: If a section or key is unknown, or an event is invalid.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    sections = {f.name for f in dataclasses.fields(Config)} - {"events"}
    unknown = set(raw) - sections
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}")

    events_raw = yaml.safe_load((path.parent / "events.yaml").read_text()) or {}
    events = tuple(_build_event(e) for e in events_raw.get("events", []))

    return Config(
        data=_build(DataConfig, raw.get("data"), "data"),
        severity=_build(SeverityConfig, raw.get("severity"), "severity"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        train=_build(TrainConfig, raw.get("train"), "train"),
        demo=_build(DemoConfig, raw.get("demo"), "demo"),
        events=events,
    )

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
from typing import Any

import yaml

HAZARDS = ("flood", "landslide")



@dataclass(frozen=True)
class Event:
    """A historical disaster used as a source of before/after image pairs.

    Attributes:
        name: Unique identifier, used in tile file names (e.g. ``"wayanad_2024"``).
        hazard: Either ``"flood"`` or ``"landslide"``.
        date: Day the event happened.
        bbox: Area of interest as (min_lon, min_lat, max_lon, max_lat) in degrees.
        after: Optional (start, end) window for the after image. Floods need imagery from
            the days the water was still standing; without it, the next dry season is used.
    """

    name: str
    hazard: str
    date: date
    bbox: tuple[float, float, float, float]
    after: tuple[date, date] | None = None


@dataclass(frozen=True)
class DataConfig:
    """Earth Engine export and tiling settings."""

    tile_size: int = 256
    stride: int = 256
    max_nodata_frac: float = 0.2
    dry_season_start_month: int = 12
    dry_season_end_month: int = 4
    max_cloud_pct: int = 40
    scale_m: int = 10
    crs: str = "EPSG:32643"
    check_scale_m: int = 100
    min_valid_frac: float = 0.6
    toa_hazards: tuple[str, ...] = ("flood",)
    cloud_score_clear: float = 0.6
    rain_days_before: int = 7
    drive_folder: str = "kerala_hazard_vision_ai"
    test_events: tuple[str, ...] = ("pettimudi_2020", "kuttanad_2018")

    def __post_init__(self) -> None:
        """Reject a dry season that does not cross New Year (the date logic relies on it)."""
        if not self.dry_season_end_month < self.dry_season_start_month:
            raise ValueError(
                "dry season must cross New Year (end month < start month), got "
                f"{self.dry_season_start_month}-{self.dry_season_end_month}"
            )


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
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints"


@dataclass(frozen=True)
class DiffusionConfig:
    """Palette-style conditional diffusion model (Saharia et al., 2022) and its training."""

    block_out_channels: tuple[int, ...] = (64, 128, 256, 256)
    layers_per_block: int = 2
    attention_levels: int = 1
    attention_head_dim: int = 64
    norm_num_groups: int = 32
    train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    sample_steps: int = 50
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    ema_decay: float = 0.999
    mixed_precision: bool = True
    time_budget_hours: float = 3.5
    checkpoint_dir: str = "checkpoints_diffusion"


@dataclass(frozen=True)
class DemoConfig:
    """Interactive demo limits and per-hazard severity sweeps.

    Severity is the share of a tile that changed, which differs a lot by hazard (a landslide
    scar covers a few percent of a tile, a flood most of it), so sweeps are set per hazard.
    """

    kerala_bbox: tuple[float, float, float, float] = (74.85, 8.17, 77.42, 12.80)
    max_demo_area_km2: float = 25.0
    sweep: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "landslide": (0.01, 0.03, 0.06, 0.10),
            "flood": (0.1, 0.3, 0.6, 0.9),
        }
    )
    train_severity_max: dict[str, float] = field(
        default_factory=lambda: {"landslide": 0.11, "flood": 0.93}
    )

    def __post_init__(self) -> None:
        """Require the same number of sweep steps per hazard (evaluation runs them in lockstep)."""
        if len({len(steps) for steps in self.sweep.values()}) > 1:
            raise ValueError(f"Every hazard needs the same number of sweep steps: {self.sweep}")


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation and Stable Diffusion comparison settings."""

    rgb_max: float = 0.3
    prompts: tuple[str, ...] = ("satellite image of the same area after a {hazard}",)
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    sd_strength: float = 0.5
    sd_steps: int = 30
    sd_size: int = 512
    sd_batch_size: int = 8


@dataclass(frozen=True)
class Config:
    """Root configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    events: tuple[Event, ...] = ()


def _build[T](cls: type[T], raw: dict[str, Any] | None, section: str) -> T:
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
    if event.after is not None and not (len(event.after) == 2 and event.after[0] <= event.after[1]):
        raise ValueError(f"Event '{event.name}': after must be [start, end], got {event.after}")
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

    data = _build(DataConfig, raw.get("data"), "data")
    unknown_tests = set(data.test_events) - {event.name for event in events}
    if unknown_tests:
        raise ValueError(f"data.test_events not in events.yaml: {sorted(unknown_tests)}")

    return Config(
        data=data,
        severity=_build(SeverityConfig, raw.get("severity"), "severity"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        train=_build(TrainConfig, raw.get("train"), "train"),
        demo=_build(DemoConfig, raw.get("demo"), "demo"),
        eval=_build(EvalConfig, raw.get("eval"), "eval"),
        diffusion=_build(DiffusionConfig, raw.get("diffusion"), "diffusion"),
        events=events,
    )

"""PyTorch dataset over tile files, plus normalisation, event split and balanced sampling.

Model input has 9 channels: the 4 before bands and 3 conditioning layers (z-scored with
training-set statistics), then severity (raw, 0-1) and hazard code (raw, 0 or 1), each as a
constant plane. The target is the 4 after bands, z-scored with the before-band statistics,
clipped to ±3 standard deviations and divided by 3 so it fits the generator's tanh range.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from kerala_hazard_vision_ai.data.severity import HAZARD_CODES

N_NORMALISED = 7  # 4 before bands + slope, rain, soil
TARGET_CLIP_SIGMA = 3.0
_TILE_NAME = re.compile(r"_r(\d+)_c(\d+)\.npz$")


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel mean and standard deviation of the 7 normalised input channels."""

    mean: np.ndarray
    std: np.ndarray

    def save(self, path: Path) -> None:
        """Write the statistics to a JSON file."""
        Path(path).write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}))

    @classmethod
    def load(cls, path: Path) -> ChannelStats:
        """Read statistics written by :meth:`save`."""
        raw = json.loads(Path(path).read_text())
        return cls(np.asarray(raw["mean"], np.float32), np.asarray(raw["std"], np.float32))


def _tile_event(path: Path) -> str:
    with np.load(path) as tile:
        return str(tile["event"])


def _tile_col(path: Path) -> int:
    match = _TILE_NAME.search(Path(path).name)
    if match is None:
        raise ValueError(f"Tile name '{Path(path).name}' does not end in _r<row>_c<col>.npz")
    return int(match.group(2))


def split_by_event(
    paths: list[Path], test_events: set[str], val_frac: float, tile_size: int
) -> dict[str, list[Path]]:
    """Split tiles into train/val/test so the test event never leaks into training.

    Every tile of a test event goes to ``test``. For each remaining event, whole columns of
    tiles are taken from the east edge until at least ``val_frac`` of that event's tiles are
    in ``val``. Training tiles that would share pixels with the validation block (tiles wider
    than the stride overlap) are dropped; the rest go to ``train``.

    Raises:
        ValueError: If a test event has no tiles (usually a typo in the config).
    """
    by_event: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_event[_tile_event(path)].append(path)
    missing = test_events - set(by_event)
    if missing:
        raise ValueError(f"Test event(s) {sorted(missing)} have no tiles; found {sorted(by_event)}")

    split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for event, event_paths in by_event.items():
        if event in test_events:
            split["test"].extend(event_paths)
            continue
        cols = [_tile_col(p) for p in event_paths]
        n_val_target = int(np.ceil(val_frac * len(event_paths)))
        val_cols: set[int] = set()
        for col in sorted(set(cols), reverse=True):
            if sum(c in val_cols for c in cols) >= n_val_target:
                break
            val_cols.add(col)
        val_start = min(val_cols)
        for path, col in zip(event_paths, cols, strict=True):
            if col in val_cols:
                split["val"].append(path)
            elif col + tile_size <= val_start:  # shares no pixel with validation tiles
                split["train"].append(path)
    return split


def compute_stats(paths: list[Path]) -> ChannelStats:
    """Mean and std of the before bands and conditioning layers over ``paths``."""
    total = np.zeros(N_NORMALISED)
    total_sq = np.zeros(N_NORMALISED)
    count = 0
    for path in paths:
        with np.load(path) as tile:
            channels = np.concatenate([tile["before"], tile["cond"]]).astype(np.float64)
        flat = channels.reshape(N_NORMALISED, -1)
        total += flat.sum(axis=1)
        total_sq += (flat**2).sum(axis=1)
        count += flat.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    std = np.where(std > 0, std, 1.0)  # a constant channel is centred, never divided by zero
    return ChannelStats(mean.astype(np.float32), std.astype(np.float32))


def build_input(
    before: np.ndarray, cond: np.ndarray, severity: float, hazard: int, stats: ChannelStats
) -> torch.Tensor:
    """Assemble the (9, H, W) model input.

    Raises:
        ValueError: If ``severity`` is outside [0, 1] or ``hazard`` is not a known code.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    if hazard not in HAZARD_CODES.values():
        raise ValueError(f"hazard must be one of {sorted(HAZARD_CODES.values())}, got {hazard}")
    channels = np.concatenate([before, cond]).astype(np.float32)
    normalised = (channels - stats.mean[:, None, None]) / stats.std[:, None, None]
    shape = normalised.shape[1:]
    planes = np.stack([np.full(shape, severity), np.full(shape, hazard)]).astype(np.float32)
    return torch.from_numpy(np.concatenate([normalised, planes]))


def unnormalise_before(x: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    """Recover before-image reflectance from model input(s) built by :func:`build_input`.

    Works on (9, H, W) and batched (N, 9, H, W) tensors.
    """
    mean = torch.as_tensor(stats.mean[:4], dtype=x.dtype, device=x.device)
    std = torch.as_tensor(stats.std[:4], dtype=x.dtype, device=x.device)
    return x[..., :4, :, :] * std[:, None, None] + mean[:, None, None]


def scale_target(after: np.ndarray, stats: ChannelStats) -> torch.Tensor:
    """Map after-image reflectance to [-1, 1] (the inverse of :func:`to_reflectance`)."""
    z = (after - stats.mean[:4, None, None]) / stats.std[:4, None, None]
    clipped = np.clip(z, -TARGET_CLIP_SIGMA, TARGET_CLIP_SIGMA) / TARGET_CLIP_SIGMA
    return torch.from_numpy(clipped.astype(np.float32))


def to_reflectance(target: torch.Tensor, stats: ChannelStats) -> torch.Tensor:
    """Map a model output or scaled target in [-1, 1] back to reflectance.

    Works on (4, H, W) and batched (N, 4, H, W) tensors.
    """
    mean = torch.as_tensor(stats.mean[:4], dtype=target.dtype, device=target.device)
    std = torch.as_tensor(stats.std[:4], dtype=target.dtype, device=target.device)
    return target * TARGET_CLIP_SIGMA * std[:, None, None] + mean[:, None, None]


class TileDataset(Dataset):
    """Tiles as ``(input (9, H, W), target (4, H, W))`` float32 tensors.

    With ``augment=True`` each item gets a random 90° rotation and horizontal flip, applied
    identically to input and target.
    """

    def __init__(self, paths: list[Path], stats: ChannelStats, augment: bool) -> None:
        """Index the tile files and cache their severity labels for sampling."""
        self.paths = list(paths)
        self.stats = stats
        self.augment = augment
        self.severities = np.array([_read_scalar(p, "severity") for p in self.paths], np.float32)

    def __len__(self) -> int:
        """Number of tiles."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, normalise and (optionally) augment one tile."""
        with np.load(self.paths[index]) as tile:
            before, after, cond = tile["before"], tile["after"], tile["cond"]
            severity, hazard = float(tile["severity"]), int(tile["hazard"])
        x = build_input(before, cond, severity, hazard, self.stats)
        y = scale_target(after, self.stats)
        if self.augment:
            k = int(torch.randint(4, ()).item())
            x, y = torch.rot90(x, k, dims=(1, 2)), torch.rot90(y, k, dims=(1, 2))
            if torch.rand(()) < 0.5:
                x, y = torch.flip(x, dims=(2,)), torch.flip(y, dims=(2,))
        return x.contiguous(), y.contiguous()


def _read_scalar(path: Path, key: str) -> float:
    with np.load(path) as tile:
        return float(tile[key])


def weighted_sampler(ds: TileDataset, threshold: float) -> WeightedRandomSampler:
    """Sampler that gives severe tiles (severity > threshold) and calm tiles equal total weight.

    Unchanged tiles dominate real disaster scenes; without rebalancing the model learns to
    copy the before image.
    """
    severe = ds.severities > threshold
    n_severe, n_calm = int(severe.sum()), int((~severe).sum())
    if n_severe == 0 or n_calm == 0:
        weights = np.ones(len(ds))
    else:
        weights = np.where(severe, 1.0 / n_severe, 1.0 / n_calm)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(ds), replacement=True)

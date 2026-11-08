"""Evaluation: image metrics, severity consistency, baselines and figures.

All methods (pix2pix, the identity baseline, Stable Diffusion) are compared on the same RGB
rendering (B4, B3, B2 with a fixed reflectance stretch), because Stable Diffusion only produces
RGB. Severity consistency needs the NIR band, so it is reported for pix2pix only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from kerala_hazard_vision_ai.config import Config
from kerala_hazard_vision_ai.data.dataset import TileDataset, to_reflectance, unnormalise_before
from kerala_hazard_vision_ai.data.severity import HAZARD_CODES, severity
from kerala_hazard_vision_ai.models import UNetGenerator

HAZARD_NAMES = {code: name for name, code in HAZARD_CODES.items()}
SEVERITY_CHANNEL, HAZARD_CHANNEL = 7, 8
_RGB_INDEX = torch.tensor([2, 1, 0])  # B4, B3, B2 from BANDS order
FID_BATCH = 32


def rgb_uint8(bands4: torch.Tensor, rgb_max: float) -> torch.Tensor:
    """Render (…, 4, H, W) reflectance as (…, 3, H, W) uint8 RGB with a fixed stretch."""
    rgb = bands4.index_select(-3, _RGB_INDEX.to(bands4.device))
    return (rgb / rgb_max).clamp(0, 1).mul(255).round().to(torch.uint8)


def image_metrics(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> dict[str, float]:
    """Mean SSIM and PSNR (dB) between two uint8 RGB batches."""
    pred, target = pred_rgb.float() / 255, target_rgb.float() / 255
    return {
        "ssim": float(structural_similarity_index_measure(pred, target, data_range=1.0)),
        "psnr": float(peak_signal_noise_ratio(pred, target, data_range=1.0)),
    }


def fid(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> float:
    """Fréchet Inception Distance between two uint8 RGB sets (downloads Inception weights).

    Inception was trained on ImageNet photos, so on satellite tiles FID is a rough realism
    signal, best used to rank methods rather than as an absolute score.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = FrechetInceptionDistance(feature=2048).to(device)
    for start in range(0, len(target_rgb), FID_BATCH):
        metric.update(target_rgb[start : start + FID_BATCH].to(device), real=True)
        metric.update(pred_rgb[start : start + FID_BATCH].to(device), real=False)
    return float(metric.compute())


def evaluate_rgb(
    pred_rgb: torch.Tensor, target_rgb: torch.Tensor, with_fid: bool = True
) -> dict[str, float]:
    """SSIM, PSNR and (optionally) FID for a method's RGB predictions."""
    result = {"n_tiles": len(pred_rgb), **image_metrics(pred_rgb, target_rgb)}
    if with_fid:
        result["fid"] = fid(pred_rgb, target_rgb)
    return result


def severity_consistency(requested: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """How well the severity measured on generated images follows the requested severity.

    Returns Pearson correlation (NaN if either side is constant) and mean absolute error.
    """
    requested, measured = np.asarray(requested, float), np.asarray(measured, float)
    if requested.std() == 0 or measured.std() == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(requested, measured)[0, 1])
    return {"pearson_r": pearson, "mae": float(np.abs(requested - measured).mean())}


def _loader(ds: TileDataset, cfg: Config) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)


@torch.no_grad()
def evaluate_model(
    g: UNetGenerator, ds: TileDataset, cfg: Config, device: torch.device, with_fid: bool = True
) -> dict:
    """Score pix2pix on a dataset and run the severity sweep from ``cfg.demo.sweep``.

    Each tile is generated at its own measured severity for the image metrics, and again at
    every sweep severity to check that the requested severity controls the measured one.
    """
    g = g.to(device).eval()
    preds, reals, requested, measured = [], [], [], []
    for x, y in _loader(ds, cfg):
        x = x.to(device)
        preds.append(rgb_uint8(to_reflectance(g(x).cpu(), ds.stats), cfg.eval.rgb_max))
        reals.append(rgb_uint8(to_reflectance(y, ds.stats), cfg.eval.rgb_max))
        before = unnormalise_before(x.cpu(), ds.stats).numpy()
        hazards = x[:, HAZARD_CHANNEL, 0, 0].round().int().tolist()
        for sev in cfg.demo.sweep:
            swept = x.clone()
            swept[:, SEVERITY_CHANNEL] = sev
            out = to_reflectance(g(swept).cpu(), ds.stats).numpy()
            for b, o, h in zip(before, out, hazards, strict=True):
                requested.append(sev)
                measured.append(severity(b, o, HAZARD_NAMES[h], cfg.severity))
    result = evaluate_rgb(torch.cat(preds), torch.cat(reals), with_fid)
    result["severity_consistency"] = severity_consistency(np.array(requested), np.array(measured))
    result["severity_sweep"] = {"requested": requested, "measured": measured}
    return result


def evaluate_identity(ds: TileDataset, cfg: Config, with_fid: bool = True) -> dict:
    """Baseline that predicts "nothing changes": the after image equals the before image."""
    preds, reals = [], []
    for x, y in _loader(ds, cfg):
        preds.append(rgb_uint8(unnormalise_before(x, ds.stats), cfg.eval.rgb_max))
        reals.append(rgb_uint8(to_reflectance(y, ds.stats), cfg.eval.rgb_max))
    return evaluate_rgb(torch.cat(preds), torch.cat(reals), with_fid)


def save_metrics(results: dict, path: Path = Path("outputs/metrics.json")) -> None:
    """Write results as indented JSON; the report takes every number from this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


def _show(ax, rgb: torch.Tensor, title: str) -> None:
    ax.imshow(rgb.permute(1, 2, 0).numpy())
    ax.set_title(title)
    ax.axis("off")


def plot_triplets(
    before_rgb: torch.Tensor,
    real_rgb: torch.Tensor,
    fake_rgb: torch.Tensor,
    path: Path,
    n: int = 4,
) -> Figure:
    """Grid of before / real after / generated after for the first ``n`` tiles; saved to path."""
    rows = min(n, len(before_rgb))
    fig = Figure(figsize=(9, 3 * rows))
    axes = fig.subplots(rows, 3, squeeze=False)
    for i in range(rows):
        _show(axes[i, 0], before_rgb[i], "Before")
        _show(axes[i, 1], real_rgb[i], "Real after")
        _show(axes[i, 2], fake_rgb[i], "Generated after")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_severity_sweep(
    before_rgb: torch.Tensor,
    frames: Sequence[torch.Tensor],
    severities: Sequence[float],
    path: Path,
) -> Figure:
    """One row: the before image, then the generated image at each severity; saved to path."""
    fig = Figure(figsize=(3 * (len(frames) + 1), 3))
    axes = fig.subplots(1, len(frames) + 1)
    _show(axes[0], before_rgb, "Before")
    for ax, frame, sev in zip(axes[1:], frames, severities, strict=True):
        _show(ax, frame, f"Severity {sev:.1f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig

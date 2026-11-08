"""pix2pix training loop with resumable checkpoints.

Usage (on Colab, with tiles and checkpoints on Google Drive)::

    python -m disastergen.train --config configs/default.yaml \
        --tiles /content/drive/MyDrive/disastergen/tiles \
        --checkpoint-dir /content/drive/MyDrive/disastergen/checkpoints --resume

``last.pt`` is written after every epoch, so a dropped Colab session loses at most one epoch.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from disastergen.config import Config, TrainConfig, load_config
from disastergen.data.dataset import (
    TileDataset,
    compute_stats,
    split_by_event,
    weighted_sampler,
)
from disastergen.models import PatchDiscriminator, UNetGenerator, build_models

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ("epoch", "d_loss", "g_gan", "g_l1", "val_l1")
_bce = nn.BCEWithLogitsLoss()
_l1 = nn.L1Loss()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizers(
    g: nn.Module, d: nn.Module, cfg: TrainConfig
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Adam optimizers for generator and discriminator with the pix2pix settings."""
    betas = (cfg.beta1, cfg.beta2)
    return (
        torch.optim.Adam(g.parameters(), lr=cfg.lr, betas=betas),
        torch.optim.Adam(d.parameters(), lr=cfg.lr, betas=betas),
    )


def train_step(
    g: UNetGenerator,
    d: PatchDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    l1_weight: float,
) -> dict[str, float]:
    """One discriminator update followed by one generator update.

    The discriminator learns to tell (input, real after) from (input, generated after).
    The generator is trained to fool it while staying close to the real image (L1).

    Returns:
        ``d_loss``, ``g_gan`` and ``g_l1`` as plain floats.
    """
    fake = g(x)

    opt_d.zero_grad(set_to_none=True)
    real_logits = d(torch.cat([x, y], dim=1))
    fake_logits = d(torch.cat([x, fake.detach()], dim=1))
    d_loss = 0.5 * (
        _bce(real_logits, torch.ones_like(real_logits))
        + _bce(fake_logits, torch.zeros_like(fake_logits))
    )
    d_loss.backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    fooled_logits = d(torch.cat([x, fake], dim=1))
    g_gan = _bce(fooled_logits, torch.ones_like(fooled_logits))
    g_l1 = _l1(fake, y)
    (g_gan + l1_weight * g_l1).backward()
    opt_g.step()

    return {"d_loss": d_loss.item(), "g_gan": g_gan.item(), "g_l1": g_l1.item()}


def save_checkpoint(
    path: Path,
    epoch: int,
    g: nn.Module,
    d: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> None:
    """Save models and optimizer state after ``epoch`` (written atomically)."""
    tmp = Path(path).with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "g": g.state_dict(),
            "d": d.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)  # a session dying mid-write never corrupts the last good checkpoint


def load_checkpoint(
    path: Path,
    g: nn.Module,
    d: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
) -> int:
    """Restore a checkpoint in place and return the epoch to continue from."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    g.load_state_dict(state["g"])
    d.load_state_dict(state["d"])
    opt_g.load_state_dict(state["opt_g"])
    opt_d.load_state_dict(state["opt_d"])
    return int(state["epoch"]) + 1


@torch.no_grad()
def validation_l1(g: UNetGenerator, loader: DataLoader, device: torch.device) -> float:
    """Mean L1 between generated and real after images over a loader."""
    g.eval()
    total, count = 0.0, 0
    for x, y in loader:
        fake = g(x.to(device))
        total += _l1(fake, y.to(device)).item() * len(x)
        count += len(x)
    g.train()
    return total / count


def _read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _append_history(path: Path, row: dict[str, float]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def fit(cfg: Config, tile_dir: Path, resume: bool) -> Path:
    """Train pix2pix on the tiles in ``tile_dir``; return the path of the best generator.

    Writes to ``cfg.train.checkpoint_dir``: ``last.pt`` (every epoch), ``best_g.pt`` (lowest
    validation L1), ``history.csv``, ``stats.json`` (normalisation) and ``config.json``.

    Raises:
        ValueError: If there are no training or validation tiles.
    """
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best_g.pt"
    history_path = ckpt_dir / "history.csv"

    split = split_by_event(
        sorted(Path(tile_dir).glob("*.npz")),
        {cfg.data.test_event},
        cfg.train.val_frac,
        cfg.data.tile_size,
    )
    if not split["train"] or not split["val"]:
        raise ValueError(f"Need train and val tiles, got {({k: len(v) for k, v in split.items()})}")
    stats = compute_stats(split["train"])
    stats.save(ckpt_dir / "stats.json")
    (ckpt_dir / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=str)
    )

    train_ds = TileDataset(split["train"], stats, augment=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=weighted_sampler(train_ds, cfg.train.severe_threshold),
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        TileDataset(split["val"], stats, augment=False),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    logger.info("Tiles: %s", {k: len(v) for k, v in split.items()})

    g, d = build_models(cfg.model)
    g, d = g.to(device), d.to(device)
    opt_g, opt_d = make_optimizers(g, d, cfg.train)

    start_epoch, best_val = 0, math.inf
    if resume and last_path.exists():
        start_epoch = load_checkpoint(last_path, g, d, opt_g, opt_d)
        best_val = min((float(r["val_l1"]) for r in _read_history(history_path)), default=math.inf)
        logger.info("Resumed at epoch %d (best val L1 %.4f)", start_epoch, best_val)
    elif history_path.exists():
        history_path.unlink()  # fresh run: do not mix with an old history

    for epoch in range(start_epoch, cfg.train.epochs):
        sums = dict.fromkeys(("d_loss", "g_gan", "g_l1"), 0.0)
        for x, y in train_loader:
            losses = train_step(
                g, d, opt_g, opt_d, x.to(device), y.to(device), cfg.train.l1_weight
            )
            for key, value in losses.items():
                sums[key] += value
        row = {"epoch": epoch, **{k: v / len(train_loader) for k, v in sums.items()}}
        row["val_l1"] = validation_l1(g, val_loader, device)
        _append_history(history_path, row)

        if row["val_l1"] < best_val:
            best_val = row["val_l1"]
            torch.save(g.state_dict(), best_path)
        save_checkpoint(last_path, epoch, g, d, opt_g, opt_d)
        logger.info("Epoch %d: %s", epoch, {k: round(v, 4) for k, v in row.items()})

    return best_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the pix2pix disaster generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--tiles", type=Path, required=True, help="Directory of .npz tiles")
    parser.add_argument("--checkpoint-dir", type=Path, help="Overrides train.checkpoint_dir")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.checkpoint_dir:
        cfg = dataclasses.replace(
            cfg, train=dataclasses.replace(cfg.train, checkpoint_dir=str(args.checkpoint_dir))
        )
    best = fit(cfg, args.tiles, resume=args.resume)
    logger.info("Best generator: %s", best)


if __name__ == "__main__":
    main()

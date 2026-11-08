"""Training loop for the conditional diffusion model, sharing data and checkpoint code with pix2pix.

Usage (on Colab)::

    python -m kerala_hazard_vision_ai.train_diffusion --config configs/default.yaml \
        --tiles /content/drive/MyDrive/kerala_hazard_vision_ai/tiles \
        --checkpoint-dir /content/drive/MyDrive/kerala_hazard_vision_ai/checkpoints_diffusion \
        --resume

An exponential moving average (EMA) of the weights is kept and evaluated, as is standard for
diffusion models: it gives smoother, higher-quality samples than the raw weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from kerala_hazard_vision_ai.config import Config, load_config
from kerala_hazard_vision_ai.models.diffusion import ConditionalDiffusion
from kerala_hazard_vision_ai.train import (
    History,
    prepare_data,
    resume_or_start,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ("epoch", "train_loss", "val_loss")
VAL_SEED = 0  # fixed noise and timesteps, so validation losses are comparable across epochs


def build_diffusion(cfg: Config) -> ConditionalDiffusion:
    """The diffusion model for the configured tile size and channels."""
    return ConditionalDiffusion(cfg.diffusion, cfg.model, cfg.data.tile_size)


@torch.no_grad()
def validation_loss(
    model: ConditionalDiffusion, loader: DataLoader, device: torch.device
) -> float:
    """Mean denoising loss over a loader, with seeded noise per batch."""
    total, count = 0.0, 0
    for batch, (x, y) in enumerate(loader):
        loss = model.loss(x.to(device), y.to(device), seed=VAL_SEED + batch)
        total += loss.item() * len(x)
        count += len(x)
    return total / count


def fit_diffusion(cfg: Config, tile_dir: Path, resume: bool) -> Path:
    """Train the diffusion model; return the path of the best EMA weights.

    Writes to ``cfg.diffusion.checkpoint_dir``: ``last.pt`` (every epoch), ``best_ema.pt``
    (lowest validation loss of the EMA model), ``history.csv``, ``stats.json`` and
    ``config.json``.

    Raises:
        ValueError: If there are no training or validation tiles.
    """
    dcfg = cfg.diffusion
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(dcfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best_ema.pt"
    history = History(ckpt_dir / "history.csv", HISTORY_FIELDS)

    train_loader, val_loader, _ = prepare_data(cfg, tile_dir, ckpt_dir, dcfg.batch_size)
    model = build_diffusion(cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(dcfg.ema_decay))
    optimizer = torch.optim.AdamW(model.parameters(), lr=dcfg.lr)
    use_amp = dcfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    objects = {"model": model, "ema": ema, "optimizer": optimizer, "scaler": scaler}
    start_epoch, best_val = resume_or_start(resume, last_path, objects, history, "val_loss")

    for epoch in range(start_epoch, dcfg.epochs):
        total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.loss(x.to(device), y.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update_parameters(model)
            total += loss.item()
        row = {
            "epoch": epoch,
            "train_loss": total / len(train_loader),
            "val_loss": validation_loss(ema.module, val_loader, device),
        }

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(ema.module.state_dict(), best_path)
        save_checkpoint(last_path, epoch, objects)
        history.append(row)
        logger.info("Epoch %d: %s", epoch, {k: round(v, 5) for k, v in row.items()})

    return best_path


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the conditional diffusion model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--tiles", type=Path, required=True, help="Directory of .npz tiles")
    parser.add_argument("--checkpoint-dir", type=Path, help="Overrides diffusion.checkpoint_dir")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.checkpoint_dir:
        diffusion = dataclasses.replace(cfg.diffusion, checkpoint_dir=str(args.checkpoint_dir))
        cfg = dataclasses.replace(cfg, diffusion=diffusion)
    logger.info("Best EMA weights: %s", fit_diffusion(cfg, args.tiles, resume=args.resume))


if __name__ == "__main__":
    main()

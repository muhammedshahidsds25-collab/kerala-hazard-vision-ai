"""pix2pix generator and discriminator."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kerala_hazard_vision_ai.config import ModelConfig
from kerala_hazard_vision_ai.models.discriminator import PatchDiscriminator
from kerala_hazard_vision_ai.models.generator import UNetGenerator

INIT_STD = 0.02  # pix2pix / DCGAN weight initialisation

__all__ = ["PatchDiscriminator", "UNetGenerator", "build_models", "init_weights", "load_generator"]


def init_weights(module: nn.Module, std: float = INIT_STD) -> None:
    """Initialise conv weights ~ N(0, std) and batch-norm scales ~ N(1, std), biases to 0."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)


def build_models(cfg: ModelConfig) -> tuple[UNetGenerator, PatchDiscriminator]:
    """Create and initialise the generator and its matching discriminator."""
    generator = UNetGenerator(cfg.in_channels, cfg.out_channels, cfg.ngf, cfg.num_downs)
    discriminator = PatchDiscriminator(cfg.in_channels + cfg.out_channels, cfg.ndf)
    init_weights(generator)
    init_weights(discriminator)
    return generator, discriminator


def load_generator(path: Path, cfg: ModelConfig, device: torch.device) -> UNetGenerator:
    """Load trained generator weights (``best_g.pt``) for inference, in eval mode."""
    generator = UNetGenerator(cfg.in_channels, cfg.out_channels, cfg.ngf, cfg.num_downs)
    generator.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return generator.to(device).eval()

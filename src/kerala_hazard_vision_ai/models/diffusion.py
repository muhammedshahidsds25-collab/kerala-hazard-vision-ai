"""Palette-style conditional diffusion model (Saharia et al., 2022).

The diffusion counterpart of pix2pix. It receives the same 9-channel condition (before
bands, slope, rain, soil, severity, hazard), concatenated with a noisy version of the after
image, and a U-Net learns to predict the added noise (DDPM, Ho et al., 2020). At inference,
DDIM (Song et al., 2021) turns pure noise into an after image in a few dozen steps. Different
seeds give different plausible outcomes for the same area, which a GAN cannot do.
"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from torch import nn
from torch.nn import functional as F

from kerala_hazard_vision_ai.config import DiffusionConfig, ModelConfig


def _block_types(cfg: DiffusionConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Down/up block names, with self-attention in the ``attention_levels`` deepest levels."""
    n_levels = len(cfg.block_out_channels)
    attention = [level >= n_levels - cfg.attention_levels for level in range(n_levels)]
    down = tuple("AttnDownBlock2D" if a else "DownBlock2D" for a in attention)
    up = tuple("AttnUpBlock2D" if a else "UpBlock2D" for a in reversed(attention))
    return down, up


class ConditionalDiffusion(nn.Module):
    """Noise-prediction U-Net plus its training (DDPM) and sampling (DDIM) schedules.

    Args:
        cfg: Diffusion architecture and schedule settings.
        model_cfg: Shared channel counts (condition and output), the same as pix2pix.
        sample_size: Tile side in pixels.
    """

    def __init__(self, cfg: DiffusionConfig, model_cfg: ModelConfig, sample_size: int) -> None:
        """Build the U-Net and the noise schedules."""
        super().__init__()
        down, up = _block_types(cfg)
        self.out_channels = model_cfg.out_channels
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=model_cfg.out_channels + model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            down_block_types=down,
            up_block_types=up,
            norm_num_groups=cfg.norm_num_groups,
        )
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.train_timesteps,
            beta_schedule=cfg.beta_schedule,
            clip_sample=True,
            clip_sample_range=1.0,  # targets are scaled to [-1, 1]
        )
        self.sample_scheduler = DDIMScheduler.from_config(self.train_scheduler.config)

    def _denoise(self, noisy: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(torch.cat([noisy, x], dim=1), t).sample

    def loss(self, x: torch.Tensor, y: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        """DDPM training loss: MSE between the true and the predicted noise.

        Args:
            x: Condition, (N, 9, H, W).
            y: Real after image scaled to [-1, 1], (N, 4, H, W).
            seed: Fix the noise and timesteps (used for a comparable validation loss).
        """
        generator = None if seed is None else torch.Generator(y.device).manual_seed(seed)
        noise = torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)
        steps = self.train_scheduler.config.num_train_timesteps
        t = torch.randint(0, steps, (len(y),), generator=generator, device=y.device)
        noisy = self.train_scheduler.add_noise(y, noise, t)
        return F.mse_loss(self._denoise(noisy, x, t), noise)

    @torch.no_grad()
    def sample(self, x: torch.Tensor, steps: int, seed: int) -> torch.Tensor:
        """Generate after images in [-1, 1] for conditions ``x`` with ``steps`` DDIM steps.

        The same ``seed`` gives the same starting noise, so outputs differ only through the
        condition; this keeps a severity sweep comparable across severities.
        """
        generator = torch.Generator(x.device).manual_seed(seed)
        shape = (len(x), self.out_channels, *x.shape[-2:])
        sample = torch.randn(shape, generator=generator, device=x.device, dtype=x.dtype)
        self.sample_scheduler.set_timesteps(steps, device=x.device)
        for t in self.sample_scheduler.timesteps:
            noise_pred = self._denoise(sample, x, t)
            sample = self.sample_scheduler.step(noise_pred, t, sample).prev_sample
        return sample.clamp(-1.0, 1.0)

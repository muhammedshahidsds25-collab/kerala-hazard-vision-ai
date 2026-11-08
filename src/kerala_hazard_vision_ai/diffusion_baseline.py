"""Zero-shot comparison: pretrained Stable Diffusion img2img on the same before images.

Stable Diffusion was never trained on this task; it only gets the before image and a text
prompt. Comparing it with pix2pix shows what a large general-purpose diffusion model can and
cannot do without task-specific training data. Requires the ``diffusion`` extra and a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from kerala_hazard_vision_ai.config import EvalConfig

if TYPE_CHECKING:
    from diffusers import StableDiffusionImg2ImgPipeline


def render_prompts(templates: Sequence[str], hazard: str) -> list[str]:
    """Fill ``{hazard}`` into each prompt template."""
    return [template.format(hazard=hazard) for template in templates]


def load_pipeline(cfg: EvalConfig, device: torch.device) -> StableDiffusionImg2ImgPipeline:
    """Load the fp16 img2img pipeline.

    The NSFW safety checker is disabled: on satellite tiles its false positives return black
    images, which would corrupt the metrics.
    """
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.sd_model, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def sd_img2img(
    pipe: StableDiffusionImg2ImgPipeline,
    rgb: torch.Tensor,
    prompt: str,
    cfg: EvalConfig,
    seed: int,
) -> torch.Tensor:
    """Transform a (N, 3, H, W) uint8 batch with one prompt; returns the same shape and dtype.

    Tiles are upscaled to ``cfg.sd_size`` (SD's native resolution), transformed, and resized
    back so the result can be scored against the real after image.
    """
    from PIL import Image

    height, width = rgb.shape[-2:]
    generator = torch.Generator(pipe.device).manual_seed(seed)
    outputs = []
    for start in range(0, len(rgb), cfg.sd_batch_size):
        chunk = rgb[start : start + cfg.sd_batch_size]
        images = [
            Image.fromarray(img.permute(1, 2, 0).numpy()).resize(
                (cfg.sd_size, cfg.sd_size), Image.BICUBIC
            )
            for img in chunk
        ]
        results = pipe(
            prompt=[prompt] * len(images),
            image=images,
            strength=cfg.sd_strength,
            num_inference_steps=cfg.sd_steps,
            generator=generator,
        ).images
        for result in results:
            small = np.asarray(result.resize((width, height), Image.BICUBIC))
            outputs.append(torch.from_numpy(small).permute(2, 0, 1))
    return torch.stack(outputs)

"""70×70 PatchGAN discriminator (Isola et al., 2017).

Instead of one real/fake score per image, it outputs a grid of scores, each judging one
overlapping 70×70 patch. That pushes the generator towards realistic local texture, while
the L1 loss takes care of the overall layout.
"""

from __future__ import annotations

import torch
from torch import nn

from disastergen.models.generator import LEAKY_SLOPE, MAX_CHANNEL_MULT


class PatchDiscriminator(nn.Module):
    """Scores (condition, image) pairs patch by patch; outputs logits (N, 1, h, w).

    Args:
        in_ch: Channels of the concatenated condition and image (9 + 4 = 13).
        ndf: Channels in the first layer.
        n_layers: Number of stride-2 layers; 3 gives a 70×70 receptive field.
    """

    def __init__(self, in_ch: int, ndf: int, n_layers: int = 3) -> None:
        """Build the convolution stack."""
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
        ]
        prev = ndf
        for i in range(1, n_layers + 1):
            ch = ndf * min(2**i, MAX_CHANNEL_MULT)
            stride = 2 if i < n_layers else 1
            layers += [
                nn.Conv2d(prev, ch, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(LEAKY_SLOPE, inplace=True),
            ]
            prev = ch
        layers.append(nn.Conv2d(prev, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one real/fake logit per patch."""
        return self.net(x)

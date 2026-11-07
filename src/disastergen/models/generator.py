"""pix2pix U-Net generator (Isola et al., 2017).

The encoder halves the resolution ``num_downs`` times; the decoder mirrors it and concatenates
each encoder feature map (skip connection), so fine spatial detail from the before image
reaches the output directly. The three innermost decoder blocks use dropout 0.5, which pix2pix
uses as its source of noise instead of a random input vector.
"""

from __future__ import annotations

import torch
from torch import nn

MAX_CHANNEL_MULT = 8
DROPOUT_BLOCKS = 3
DROPOUT_P = 0.5
LEAKY_SLOPE = 0.2


def _down(in_ch: int, out_ch: int, norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(LEAKY_SLOPE, inplace=True))
    return nn.Sequential(*layers)


def _up(in_ch: int, out_ch: int, dropout: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(DROPOUT_P))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """U-Net mapping a (N, in_ch, H, W) input to a (N, out_ch, H, W) image in [-1, 1].

    Args:
        in_ch: Input channels (9: before bands, conditioning, severity, hazard).
        out_ch: Output channels (4: after bands).
        ngf: Channels in the first encoder layer; deeper layers use up to ``8 * ngf``.
        num_downs: Number of downsamplings; H and W must be divisible by ``2**num_downs``.
    """

    def __init__(self, in_ch: int, out_ch: int, ngf: int, num_downs: int) -> None:
        """Build the encoder, decoder and output layer."""
        super().__init__()
        self.num_downs = num_downs
        channels = [ngf * min(2**i, MAX_CHANNEL_MULT) for i in range(num_downs)]

        self.encoder = nn.ModuleList()
        prev = in_ch
        for i, ch in enumerate(channels):
            # No normalisation on the first layer (raw input) or at the 1×1 bottleneck.
            self.encoder.append(_down(prev, ch, norm=0 < i < num_downs - 1))
            prev = ch

        self.decoder = nn.ModuleList()
        for step, i in enumerate(range(num_downs - 2, -1, -1)):
            self.decoder.append(_up(prev, channels[i], dropout=step < DROPOUT_BLOCKS))
            prev = 2 * channels[i]  # upsampled features + skip connection

        self.output = nn.Sequential(
            nn.ConvTranspose2d(prev, out_ch, 4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the after image.

        Raises:
            ValueError: If the spatial size is not divisible by ``2**num_downs``.
        """
        factor = 2**self.num_downs
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(
                f"Input size {tuple(x.shape[-2:])} must be divisible by {factor} "
                f"(2**num_downs with num_downs={self.num_downs})"
            )
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        skips.pop()  # the bottleneck output is the decoder's input, not a skip
        for block in self.decoder:
            x = torch.cat([block(x), skips.pop()], dim=1)
        return self.output(x)

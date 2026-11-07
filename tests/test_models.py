"""Tests for the pix2pix generator and PatchGAN discriminator."""

import pytest
import torch

from disastergen.config import ModelConfig
from disastergen.models import PatchDiscriminator, UNetGenerator, build_models


def test_generator_shape_and_range():
    g = UNetGenerator(9, 4, ngf=8, num_downs=8)
    y = g(torch.randn(1, 9, 256, 256))
    assert y.shape == (1, 4, 256, 256) and y.abs().max() <= 1


def test_small_generator_for_tests():
    g = UNetGenerator(9, 4, ngf=8, num_downs=6)
    assert g(torch.randn(2, 9, 64, 64)).shape == (2, 4, 64, 64)


def test_skip_connections_carry_input_detail():
    # With skips, two inputs that differ only in one pixel give different outputs.
    g = UNetGenerator(9, 4, ngf=8, num_downs=6).eval()
    x = torch.randn(1, 9, 64, 64)
    x2 = x.clone()
    x2[0, 0, 10, 10] += 5.0
    assert not torch.allclose(g(x), g(x2))


def test_discriminator_patch_shape():
    assert PatchDiscriminator(13, ndf=8)(torch.randn(1, 13, 256, 256)).shape == (1, 1, 30, 30)


def test_bad_input_size_raises():
    with pytest.raises(ValueError, match="divisible"):
        UNetGenerator(9, 4, ngf=8, num_downs=8)(torch.randn(1, 9, 100, 100))


def test_build_models_wires_channels():
    g, d = build_models(ModelConfig(ngf=8, ndf=8, num_downs=6))
    x = torch.randn(1, 9, 64, 64)
    fake = g(x)
    assert d(torch.cat([x, fake], dim=1)).shape[1] == 1


def test_init_weights_std():
    g, _ = build_models(ModelConfig(ngf=8, ndf=8, num_downs=6))
    conv = next(m for m in g.modules() if isinstance(m, torch.nn.Conv2d))
    assert 0.01 < conv.weight.std().item() < 0.03

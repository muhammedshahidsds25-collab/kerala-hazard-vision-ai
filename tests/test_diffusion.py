"""Tests for the Palette-style conditional diffusion model."""

import pytest
import torch

from kerala_hazard_vision_ai.config import DiffusionConfig, ModelConfig
from kerala_hazard_vision_ai.models.diffusion import ConditionalDiffusion

TINY = DiffusionConfig(
    block_out_channels=(8, 16), layers_per_block=1, attention_levels=1, norm_num_groups=4
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return ConditionalDiffusion(TINY, ModelConfig(), sample_size=32)


def data(n=2):
    gen = torch.Generator().manual_seed(0)
    return torch.randn(n, 9, 32, 32, generator=gen), torch.rand(n, 4, 32, 32, generator=gen) * 2 - 1


def test_unet_takes_condition_and_noisy_target(model):
    assert model.unet.config.in_channels == 13 and model.unet.config.out_channels == 4


def test_loss_is_finite_scalar(model):
    loss = model.loss(*data())
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_with_fixed_seed_is_reproducible(model):
    x, y = data()
    assert torch.equal(model.loss(x, y, seed=1), model.loss(x, y, seed=1))


def test_sample_shape_and_range(model):
    x, _ = data()
    out = model.sample(x, steps=3, seed=0)
    assert out.shape == (2, 4, 32, 32) and out.abs().max() <= 1


def test_sample_deterministic_for_seed(model):
    x, _ = data()
    assert torch.equal(model.sample(x, steps=3, seed=7), model.sample(x, steps=3, seed=7))
    assert not torch.equal(model.sample(x, steps=3, seed=7), model.sample(x, steps=3, seed=8))


def test_condition_changes_sample(model):
    x, _ = data()
    x2 = x.clone()
    x2[:, 7] = 1.0  # different severity plane
    assert not torch.equal(model.sample(x, steps=3, seed=0), model.sample(x2, steps=3, seed=0))


def test_loss_drops_on_fixed_batch(model):
    x, y = data()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = model.loss(x, y, seed=0).item()
    for step in range(40):
        opt.zero_grad()
        model.loss(x, y, seed=step).backward()
        opt.step()
    assert model.loss(x, y, seed=0).item() < first

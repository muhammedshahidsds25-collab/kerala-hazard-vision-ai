"""Tests for the pix2pix training loop and checkpointing."""

import csv
import math
from dataclasses import replace

import pytest
import torch

from kerala_hazard_vision_ai.config import ModelConfig
from kerala_hazard_vision_ai.models import build_models
from kerala_hazard_vision_ai.train import (
    fit,
    load_checkpoint,
    make_optimizers,
    save_checkpoint,
    set_seed,
    train_step,
)

TINY = ModelConfig(ngf=8, ndf=8, num_downs=6)


@pytest.fixture
def setup(cfg):
    set_seed(0)
    g, d = build_models(TINY)
    opt_g, opt_d = make_optimizers(g, d, cfg.train)
    return g, d, opt_g, opt_d


def batch():
    gen = torch.Generator().manual_seed(0)
    return torch.randn(2, 9, 64, 64, generator=gen), torch.rand(2, 4, 64, 64, generator=gen) * 2 - 1


def test_train_step_finite_losses(setup):
    out = train_step(*setup, *batch(), l1_weight=100.0)
    assert set(out) == {"d_loss", "g_gan", "g_l1"}
    assert all(math.isfinite(v) for v in out.values())


def test_train_step_reduces_l1_on_fixed_batch(setup):
    x, y = batch()
    first = train_step(*setup, x, y, l1_weight=100.0)
    for _ in range(30):
        last = train_step(*setup, x, y, l1_weight=100.0)
    assert last["g_l1"] < first["g_l1"]


def test_checkpoint_roundtrip_resumes_epoch(setup, tmp_path, cfg):
    g, d, opt_g, opt_d = setup
    train_step(g, d, opt_g, opt_d, *batch(), l1_weight=100.0)  # populate optimizer state
    save_checkpoint(tmp_path / "c.pt", 4, g, d, opt_g, opt_d)

    g2, d2 = build_models(TINY)
    opt_g2, opt_d2 = make_optimizers(g2, d2, cfg.train)
    assert load_checkpoint(tmp_path / "c.pt", g2, d2, opt_g2, opt_d2) == 5
    for a, b in zip(g.state_dict().values(), g2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    assert opt_g2.state_dict()["state"].keys() == opt_g.state_dict()["state"].keys()


def test_set_seed_reproducible():
    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    assert torch.equal(a, torch.rand(3))


def tiny_config(cfg, tmp_path, epochs):
    return replace(
        cfg,
        model=TINY,
        data=replace(cfg.data, test_event="a"),
        train=replace(
            cfg.train, epochs=epochs, batch_size=4, num_workers=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )


def read_history(tmp_path):
    with open(tmp_path / "ckpt" / "history.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_fit_runs_and_resumes(cfg, tile_paths, tmp_path):
    tile_dir = tile_paths[0].parent
    best = fit(tiny_config(cfg, tmp_path, epochs=1), tile_dir, resume=False)
    assert best.exists() and (tmp_path / "ckpt" / "stats.json").exists()
    assert [row["epoch"] for row in read_history(tmp_path)] == ["0"]

    fit(tiny_config(cfg, tmp_path, epochs=2), tile_dir, resume=True)
    rows = read_history(tmp_path)
    assert [row["epoch"] for row in rows] == ["0", "1"]
    assert all(math.isfinite(float(row["val_l1"])) for row in rows)

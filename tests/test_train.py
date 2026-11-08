"""Tests for the pix2pix training loop and checkpointing."""

import csv
import math
from dataclasses import replace

import pytest
import torch
from conftest import SYNTHETIC_EVENTS

from kerala_hazard_vision_ai.config import ModelConfig
from kerala_hazard_vision_ai.models import build_models
from kerala_hazard_vision_ai.train import (
    History,
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
    save_checkpoint(tmp_path / "c.pt", 4, {"g": g, "d": d, "opt_g": opt_g, "opt_d": opt_d})

    g2, d2 = build_models(TINY)
    opt_g2, opt_d2 = make_optimizers(g2, d2, cfg.train)
    objects = {"g": g2, "d": d2, "opt_g": opt_g2, "opt_d": opt_d2}
    assert load_checkpoint(tmp_path / "c.pt", objects) == 5
    for a, b in zip(g.state_dict().values(), g2.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    first_param = next(iter(opt_g.state_dict()["state"]))
    assert torch.equal(
        opt_g.state_dict()["state"][first_param]["exp_avg"],
        opt_g2.state_dict()["state"][first_param]["exp_avg"],
    )


def test_history_appends_and_reads(tmp_path):
    history = History(tmp_path / "h.csv", ("epoch", "loss"))
    history.append({"epoch": 0, "loss": 0.5})
    history.append({"epoch": 1, "loss": 0.25})
    assert history.column("loss") == [0.5, 0.25]


def test_set_seed_reproducible():
    set_seed(1)
    a = torch.rand(3)
    set_seed(1)
    assert torch.equal(a, torch.rand(3))


def tiny_config(cfg, tmp_path, epochs):
    return replace(
        cfg,
        model=TINY,
        data=replace(cfg.data, test_events=("a",)),
        events=SYNTHETIC_EVENTS,
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


def test_resume_from_cpu_checkpoint_with_enabled_scaler(tmp_path):
    # A run started without CUDA saves a disabled (empty) scaler; resuming on a GPU must work.
    save_checkpoint(tmp_path / "c.pt", 0, {"scaler": torch.amp.GradScaler("cpu", enabled=False)})
    enabled = torch.amp.GradScaler("cpu", enabled=True)
    assert load_checkpoint(tmp_path / "c.pt", {"scaler": enabled}) == 1

"""Tests for the diffusion training loop."""

import math
from dataclasses import replace

from conftest import SYNTHETIC_EVENTS

from kerala_hazard_vision_ai.config import ModelConfig
from kerala_hazard_vision_ai.train import History
from kerala_hazard_vision_ai.train_diffusion import fit_diffusion


def tiny_config(cfg, tmp_path, epochs):
    return replace(
        cfg,
        model=ModelConfig(),
        data=replace(cfg.data, test_events=("a",)),
        events=SYNTHETIC_EVENTS,
        train=replace(cfg.train, num_workers=0),
        diffusion=replace(
            cfg.diffusion,
            block_out_channels=(8, 16),
            layers_per_block=1,
            attention_levels=1,
            norm_num_groups=4,
            epochs=epochs,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "ckpt"),
        ),
    )


def test_fit_diffusion_runs_and_resumes(cfg, tile_paths, tmp_path):
    tile_dir = tile_paths[0].parent
    best = fit_diffusion(tiny_config(cfg, tmp_path, epochs=1), tile_dir, resume=False)
    ckpt = tmp_path / "ckpt"
    assert best == ckpt / "best_ema.pt" and best.exists() and (ckpt / "stats.json").exists()

    fit_diffusion(tiny_config(cfg, tmp_path, epochs=2), tile_dir, resume=True)
    history = History(ckpt / "history.csv", ())
    assert history.column("epoch") == [0.0, 1.0]
    assert all(math.isfinite(v) for v in history.column("val_loss"))


def test_best_ema_loads_into_fresh_model_and_resume_restores_ema(cfg, tile_paths, tmp_path):
    import torch

    from kerala_hazard_vision_ai.train_diffusion import load_diffusion

    run_cfg = tiny_config(cfg, tmp_path, epochs=1)
    best = fit_diffusion(run_cfg, tile_paths[0].parent, resume=False)
    model = load_diffusion(best, run_cfg, torch.device("cpu"))
    saved = torch.load(best, weights_only=True)
    assert all(torch.equal(v, saved[k]) for k, v in model.state_dict().items())

    last = torch.load(tmp_path / "ckpt" / "last.pt", weights_only=True)
    assert last["ema"]["n_averaged"].item() > 0

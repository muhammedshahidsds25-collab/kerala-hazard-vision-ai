"""Tests for evaluation metrics, baselines and figures (FID itself runs only on Colab)."""

import json

import numpy as np
import pytest
import torch
from conftest import write_tile

from kerala_hazard_vision_ai.config import DiffusionConfig
from kerala_hazard_vision_ai.data.dataset import TileDataset, compute_stats
from kerala_hazard_vision_ai.diffusion_baseline import render_prompts
from kerala_hazard_vision_ai.evaluate import (
    damage_scores,
    diffusion_predict,
    evaluate_damage,
    evaluate_identity,
    evaluate_model,
    gan_predict,
    identity_predict,
    image_metrics,
    plot_comparison,
    plot_severity_sweep,
    rgb_uint8,
    save_metrics,
    severity_consistency,
)
from kerala_hazard_vision_ai.models import UNetGenerator
from kerala_hazard_vision_ai.models.diffusion import ConditionalDiffusion


def test_identical_images_perfect():
    rgb = rgb_uint8(torch.rand(2, 4, 64, 64) * 0.3, rgb_max=0.3)
    m = image_metrics(rgb, rgb)
    assert m["ssim"] == pytest.approx(1.0) and m["psnr"] > 40


def test_different_images_worse():
    a = rgb_uint8(torch.rand(2, 4, 64, 64) * 0.3, rgb_max=0.3)
    b = rgb_uint8(torch.rand(2, 4, 64, 64) * 0.3, rgb_max=0.3)
    assert image_metrics(a, b)["ssim"] < 0.5


def test_rgb_uint8_shape_dtype_and_band_order():
    bands = torch.zeros(1, 4, 8, 8)
    bands[:, 2] = 0.3  # B4 = red at full scale
    rgb = rgb_uint8(bands, rgb_max=0.3)
    assert rgb.dtype == torch.uint8 and rgb.shape == (1, 3, 8, 8)
    assert rgb[0, 0].unique().tolist() == [255] and rgb[0, 2].unique().tolist() == [0]


def test_severity_consistency_perfect():
    result = severity_consistency(np.array([0.1, 0.5, 0.9]), np.array([0.1, 0.5, 0.9]))
    assert result["pearson_r"] == pytest.approx(1.0) and result["mae"] == pytest.approx(0.0)


def test_severity_consistency_constant_output_is_nan():
    result = severity_consistency(np.array([0.1, 0.5, 0.9]), np.zeros(3))
    assert np.isnan(result["pearson_r"]) and result["mae"] == pytest.approx(0.5)


def test_render_prompts():
    prompts = render_prompts(["satellite image after a {hazard}", "{hazard} damage"], "landslide")
    assert prompts == ["satellite image after a landslide", "landslide damage"]


def test_save_metrics_roundtrip(tmp_path):
    save_metrics({"a": 1.0}, tmp_path / "m.json")
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1.0}


@pytest.fixture
def same_ds(tmp_path):
    rng = np.random.default_rng(0)
    paths = [write_tile(tmp_path, "t", 0, i, 0.0, rng, same=True) for i in range(3)]
    return TileDataset(paths, compute_stats(paths), augment=False)


def test_identity_baseline_on_unchanged_tiles_is_perfect(same_ds, cfg):
    result = evaluate_identity(same_ds, cfg, with_fid=False)
    assert result["ssim"] == pytest.approx(1.0) and result["n_tiles"] == 3


def test_evaluate_model_reports_metrics_sweep_and_speed(same_ds, cfg):
    g = UNetGenerator(9, 4, ngf=8, num_downs=6)
    result = evaluate_model(gan_predict(g, torch.device("cpu")), same_ds, cfg, with_fid=False)
    assert {"ssim", "psnr", "n_tiles", "severity_consistency", "sec_per_tile"} <= set(result)
    assert result["sec_per_tile"] > 0
    sweep = cfg.demo.sweep["landslide"]  # the synthetic tiles are landslides
    assert len(result["severity_sweep"]["requested"]) == 3 * len(sweep)
    assert set(result["severity_sweep"]["requested"]) == set(sweep)


def test_diffusion_predict_shape_and_determinism(cfg):
    model = ConditionalDiffusion(
        DiffusionConfig(block_out_channels=(8, 16), layers_per_block=1, attention_levels=1,
                        norm_num_groups=4, attention_head_dim=8),
        cfg.model,
        sample_size=32,
    )
    predict = diffusion_predict(model, torch.device("cpu"), steps=2, seed=0)
    x = torch.randn(2, 9, 32, 32)
    out = predict(x)
    assert out.shape == (2, 4, 32, 32) and torch.equal(out, predict(x))


def test_plots_write_files(tmp_path):
    rgb = rgb_uint8(torch.rand(2, 4, 16, 16) * 0.3, rgb_max=0.3)
    fig = plot_comparison({"Before": rgb, "Real": rgb, "pix2pix": rgb, "Diffusion": rgb},
                          tmp_path / "t.png")
    assert len(fig.axes) == 2 * 4  # 2 tiles x 4 columns
    plot_severity_sweep(rgb[0], [rgb[0], rgb[1]], [0.1, 0.6], tmp_path / "s.png")
    assert (tmp_path / "t.png").stat().st_size > 0 and (tmp_path / "s.png").stat().st_size > 0


def test_damage_scores_counts():
    assert damage_scores(tp=8, fp=2, fn=8) == pytest.approx(
        {"precision": 0.8, "recall": 0.5, "f1": 8 / 13, "iou": 8 / 18}
    )
    nothing = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    assert damage_scores(tp=0, fp=0, fn=5) == nothing


def _landslide_tiles(tmp_path, n_changed=2, n_calm=2):
    """Tiles whose left half loses vegetation (changed) or stays the same (calm)."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n_changed + n_calm):
        vegetation = np.array([0.05, 0.08, 0.05, 0.40], np.float32)[:, None, None]
        # Realistic spread, so the +-3 sigma target scaling keeps the damage visible.
        before = (vegetation + rng.normal(0, 0.05, (4, 64, 64))).clip(0.01, 1).astype(np.float32)
        after = before.copy()
        sev = 0.0
        if i < n_changed:
            after[:, :, :32] = np.array([0.15, 0.18, 0.22, 0.25])[:, None, None]
            sev = 0.5
        cond = rng.uniform(0, 30, (3, 64, 64)).astype(np.float32)
        path = tmp_path / f"t_r0_c{i}.npz"
        np.savez(path, before=before, after=after, cond=cond, hazard=1, severity=sev, event="t")
        paths.append(path)
    return TileDataset(paths, compute_stats(paths), augment=False)


def test_identity_finds_no_damage(tmp_path, cfg):
    ds = _landslide_tiles(tmp_path)
    result = evaluate_damage(identity_predict(ds.stats), ds, cfg)
    assert result["damage"]["recall"] == 0.0 and result["damage"]["f1"] == 0.0
    assert result["changed_tiles"]["n_tiles"] == 2


def test_perfect_prediction_finds_all_damage(tmp_path, cfg):
    ds = _landslide_tiles(tmp_path)
    targets = {i: ds[i][1] for i in range(len(ds))}
    lookup = {round(float(ds[i][0][0].mean()), 6): i for i in range(len(ds))}

    def oracle(x):  # returns the real after image of each tile in the batch
        return torch.stack([targets[lookup[round(float(t[0].mean()), 6)]] for t in x])

    result = evaluate_damage(oracle, ds, cfg)
    assert result["damage"]["f1"] == pytest.approx(1.0)
    assert result["changed_tiles"]["ssim"] == pytest.approx(1.0, abs=1e-3)

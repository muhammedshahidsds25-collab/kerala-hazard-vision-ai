"""Tests for the tile dataset, event split and weighted sampling."""

import numpy as np
import pytest
import torch
from conftest import write_tile

from kerala_hazard_vision_ai.data.dataset import (
    ChannelStats,
    TileDataset,
    build_input,
    compute_stats,
    list_tiles,
    load_shared_stats,
    split_by_event,
    to_reflectance,
    unnormalise_before,
    weighted_sampler,
)

SIZE = 64


@pytest.fixture
def paths(tile_paths):
    return tile_paths


@pytest.fixture
def stats(paths):
    return compute_stats(paths)


def test_item_shapes(paths, stats):
    x, y = TileDataset(paths, stats, augment=False)[0]
    assert x.shape == (9, SIZE, SIZE) and y.shape == (4, SIZE, SIZE)
    assert x.dtype == torch.float32 and y.dtype == torch.float32


def test_severity_and_hazard_channels_constant(paths, stats):
    x, _ = TileDataset(paths, stats, augment=False)[0]
    assert x[7].unique().tolist() == [0.5] and x[8].unique().tolist() == [1.0]


def test_augment_applies_same_transform(tmp_path, stats):
    path = write_tile(tmp_path, "same", 0, 0, 0.0, np.random.default_rng(1), same=True)
    ds = TileDataset([path], stats, augment=True)
    torch.manual_seed(3)
    for _ in range(8):  # several random flips/rotations
        x, y = ds[0]
        assert torch.allclose(unnormalise_before(x, stats), to_reflectance(y, stats), atol=1e-5)


def test_target_in_range(paths, stats):
    _, y = TileDataset(paths, stats, augment=False)[0]
    assert y.min() >= -1 and y.max() <= 1


def test_to_reflectance_inverts_scaling(paths, stats):
    _, y = TileDataset(paths, stats, augment=False)[0]
    after = np.load(paths[0])["after"]
    assert torch.allclose(to_reflectance(y, stats), torch.from_numpy(after), atol=1e-5)


def test_stats_shape_and_roundtrip(paths, stats, tmp_path):
    assert stats.mean.shape == (7,) and stats.std.shape == (7,) and (stats.std > 0).all()
    stats.save(tmp_path / "stats.json")
    loaded = ChannelStats.load(tmp_path / "stats.json")
    assert np.allclose(loaded.mean, stats.mean) and np.allclose(loaded.std, stats.std)


def test_split_is_disjoint_and_complete(paths):
    split = split_by_event(paths, {"a"}, 0.2, tile_size=64)
    assert {np.load(p)["event"].item() for p in split["test"]} == {"a"}
    assert not set(split["train"]) & set(split["val"])
    assert sum(len(v) for v in split.values()) == len(paths)
    # Validation is the easternmost block of each training event.
    assert {p.name for p in split["val"]} == {"b_2020_r0_c1152.npz", "c_r0_c1152.npz",
                                              "b_2020_r0_c1024.npz", "c_r0_c1024.npz"}


def test_split_drops_train_tiles_overlapping_validation(paths):
    # Tiles 256 px wide at 128 px spacing: the column next to the val block overlaps it.
    split = split_by_event(paths, {"a"}, 0.2, tile_size=256)
    assert {p.name for p in split["val"]} >= {"c_r0_c1024.npz", "c_r0_c1152.npz"}
    assert "c_r0_c896.npz" not in {p.name for p in split["train"]}
    assert "c_r0_c768.npz" in {p.name for p in split["train"]}
    assert len(split["train"]) == 2 * 7


def test_split_rejects_unknown_test_event(paths):
    with pytest.raises(ValueError, match="typo"):
        split_by_event(paths, {"typo"}, 0.2, tile_size=64)


def test_build_input_rejects_bad_severity(stats):
    before = np.zeros((4, SIZE, SIZE), dtype=np.float32)
    cond = np.zeros((3, SIZE, SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="severity"):
        build_input(before, cond, 1.5, 1, stats)
    with pytest.raises(ValueError, match="severity"):
        build_input(before, cond, -0.1, 1, stats)


def test_sampler_oversamples_severe(paths, stats):
    ds = TileDataset(paths, stats, augment=False)
    weights = weighted_sampler(ds, threshold=0.05).weights.tolist()
    pairs = list(zip(weights, ds.severities, strict=True))
    severe = [w for w, sev in pairs if sev > 0.05]
    calm = [w for w, sev in pairs if sev <= 0.05]
    assert min(severe) > max(calm)
    # Severe and calm tiles end up with equal total sampling weight.
    assert sum(severe) == pytest.approx(sum(calm))


def test_load_shared_stats_rejects_different_normalisation(stats, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    stats.save(tmp_path / "a" / "stats.json")
    stats.save(tmp_path / "b" / "stats.json")
    shared = load_shared_stats(tmp_path / "a", tmp_path / "b")
    assert np.allclose(shared.mean, stats.mean)

    ChannelStats(stats.mean + 1, stats.std).save(tmp_path / "b" / "stats.json")
    with pytest.raises(ValueError, match="different tiles"):
        load_shared_stats(tmp_path / "a", tmp_path / "b")


def test_list_tiles_keeps_only_configured_events(paths):
    tiles = list_tiles(paths[0].parent, {"a", "c"})
    assert len(tiles) == 20 and tiles == sorted(tiles)
    assert {p.name.split("_r")[0] for p in tiles} == {"a", "c"}

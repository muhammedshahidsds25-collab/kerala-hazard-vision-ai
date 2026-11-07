"""Tests for the tile dataset, event split and weighted sampling."""

import numpy as np
import pytest
import torch
from conftest import write_tile

from disastergen.data.dataset import (
    ChannelStats,
    TileDataset,
    build_input,
    compute_stats,
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
    split = split_by_event(paths, {"a"}, 0.2)
    assert {np.load(p)["event"].item() for p in split["test"]} == {"a"}
    assert not set(split["train"]) & set(split["val"])
    assert sum(len(v) for v in split.values()) == len(paths)
    # Validation is the easternmost block of each training event.
    assert {p.name for p in split["val"]} == {"b_2020_r0_c1152.npz", "c_r0_c1152.npz",
                                              "b_2020_r0_c1024.npz", "c_r0_c1024.npz"}


def test_split_rejects_unknown_test_event(paths):
    with pytest.raises(ValueError, match="typo"):
        split_by_event(paths, {"typo"}, 0.2)


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

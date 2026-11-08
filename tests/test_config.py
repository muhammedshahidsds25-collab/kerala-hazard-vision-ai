"""Tests for the typed configuration loader."""

import shutil

import pytest

from kerala_hazard_vision_ai.config import load_config


def test_default_config_loads(cfg):
    assert cfg.model.in_channels == 9 and cfg.model.out_channels == 4
    assert cfg.train.l1_weight == 100.0 and cfg.data.tile_size == 256
    assert {e.hazard for e in cfg.events} <= {"flood", "landslide"}
    assert cfg.data.test_event in {e.name for e in cfg.events}
    assert cfg.eval.prompts and all("{hazard}" in p for p in cfg.eval.prompts)


def test_events_have_valid_bboxes(cfg):
    for event in cfg.events:
        min_lon, min_lat, max_lon, max_lat = event.bbox
        assert min_lon < max_lon and min_lat < max_lat


def test_unknown_key_raises(tmp_path, default_config_path):
    shutil.copy(default_config_path.parent / "events.yaml", tmp_path / "events.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        load_config(bad)


def test_unknown_hazard_raises(tmp_path, default_config_path):
    shutil.copy(default_config_path, tmp_path / "default.yaml")
    (tmp_path / "events.yaml").write_text(
        "events:\n  - name: x\n    hazard: tsunami\n    date: 2020-01-01\n"
        "    bbox: [76.0, 10.0, 76.1, 10.1]\n"
    )
    with pytest.raises(ValueError, match="tsunami"):
        load_config(tmp_path / "default.yaml")

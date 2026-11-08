"""Tests for the typed configuration loader."""

import shutil

import pytest

from kerala_hazard_vision_ai.config import load_config


def test_default_config_loads(cfg):
    assert cfg.model.in_channels == 9 and cfg.model.out_channels == 4
    assert cfg.train.l1_weight == 100.0 and cfg.data.tile_size == 256
    assert {e.hazard for e in cfg.events} <= {"flood", "landslide"}
    assert set(cfg.data.test_events) <= {e.name for e in cfg.events}
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


def test_one_kerala_test_event_per_hazard(cfg):
    tests = [e for e in cfg.events if e.name in cfg.data.test_events]
    assert sorted(e.hazard for e in tests) == ["flood", "landslide"]
    k_min_lon, k_min_lat, k_max_lon, k_max_lat = cfg.demo.kerala_bbox
    for test in tests:
        min_lon, min_lat, max_lon, max_lat = test.bbox
        assert k_min_lon <= min_lon and max_lon <= k_max_lon
        assert k_min_lat <= min_lat and max_lat <= k_max_lat


def test_flood_events_have_near_event_after_window(cfg):
    floods = [e for e in cfg.events if e.hazard == "flood"]
    assert len(floods) >= 3
    assert all(e.after and e.date <= e.after[0] < e.after[1] for e in floods)


def test_unknown_test_event_in_config_raises(tmp_path, default_config_path):
    text = default_config_path.read_text().replace("pettimudi_2020", "typo_2020")
    (tmp_path / "default.yaml").write_text(text)
    shutil.copy(default_config_path.parent / "events.yaml", tmp_path / "events.yaml")
    with pytest.raises(ValueError, match="typo_2020"):
        load_config(tmp_path / "default.yaml")


def test_extra_western_ghats_events_present(cfg):
    names = {e.name for e in cfg.events}
    assert {"taliye_2021", "ambeghar_2021", "irshalwadi_2023", "shiroor_2024"} <= names

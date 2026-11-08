"""Tests for the real-vs-generated human evaluation survey."""

import csv

import pytest
import torch

from disastergen.human_eval import build_survey, save_rgb_pngs, score_survey


@pytest.fixture
def images(tmp_path):
    rgb = torch.randint(0, 256, (24, 3, 16, 16), dtype=torch.uint8)
    real = save_rgb_pngs(rgb[:12], tmp_path / "src", "real")  # real[i] and fake[i]: same tile
    fake = save_rgb_pngs(rgb[12:], tmp_path / "src", "fake")
    return real, fake


def read_key(path):
    with open(path, newline="") as f:
        return {row["item"]: row["is_generated"] == "1" for row in csv.DictReader(f)}


def write_answers(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rater", "item", "answer"])
        writer.writerows(rows)
    return path


def test_save_rgb_pngs(images):
    real, _ = images
    assert [p.name for p in real[:2]] == ["real_000.png", "real_001.png"]
    assert all(p.stat().st_size > 0 for p in real)


def test_build_survey_balanced_and_shuffled(images, tmp_path):
    key = read_key(build_survey(*images, n=10, out_dir=tmp_path / "survey", seed=0))
    assert sorted(key) == [f"q{i:02d}" for i in range(1, 11)]
    assert sum(key.values()) == 5
    assert all((tmp_path / "survey" / f"{item}.png").exists() for item in key)
    assert list(key.values()) != [False] * 5 + [True] * 5  # not in source order


def test_build_survey_never_shows_both_versions_of_a_tile(images, tmp_path):
    build_survey(*images, n=6, out_dir=tmp_path / "survey", seed=0)
    with open(tmp_path / "survey" / "key.csv", newline="") as f:
        tiles = [row["source"].split("_")[1] for row in csv.DictReader(f)]
    assert len(set(tiles)) == len(tiles)


def test_score_rejects_empty_answers(images, tmp_path):
    key_path = build_survey(*images, n=6, out_dir=tmp_path / "survey", seed=0)
    with pytest.raises(ValueError, match="no answers"):
        score_survey(write_answers(tmp_path / "a.csv", []), key_path)


def test_build_survey_rejects_too_few_images(images, tmp_path):
    with pytest.raises(ValueError, match="images"):
        build_survey(*images, n=14, out_dir=tmp_path / "survey", seed=0)


def test_score_all_fooled(images, tmp_path):
    key_path = build_survey(*images, n=10, out_dir=tmp_path / "survey", seed=0)
    key = read_key(key_path)
    answers = [(r, item, "real") for r in ("r1", "r2") for item in key]
    result = score_survey(write_answers(tmp_path / "a.csv", answers), key_path)
    assert result == {"accuracy": 0.5, "fooled_rate": 1.0, "n_raters": 2}


def test_score_perfect_raters(images, tmp_path):
    key_path = build_survey(*images, n=10, out_dir=tmp_path / "survey", seed=0)
    key = read_key(key_path)
    answers = [("r1", item, "generated" if gen else "real") for item, gen in key.items()]
    result = score_survey(write_answers(tmp_path / "a.csv", answers), key_path)
    assert result["accuracy"] == 1.0 and result["fooled_rate"] == 0.0


def test_score_rejects_unknown_item(images, tmp_path):
    key_path = build_survey(*images, n=10, out_dir=tmp_path / "survey", seed=0)
    with pytest.raises(ValueError, match="q99"):
        score_survey(write_answers(tmp_path / "a.csv", [("r1", "q99", "real")]), key_path)


def test_score_rejects_bad_answer(images, tmp_path):
    key_path = build_survey(*images, n=10, out_dir=tmp_path / "survey", seed=0)
    with pytest.raises(ValueError, match="maybe"):
        score_survey(write_answers(tmp_path / "a.csv", [("r1", "q01", "maybe")]), key_path)

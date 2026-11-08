"""Tests for the real-vs-generated human evaluation survey (two models)."""

import csv

import pytest
import torch

from kerala_hazard_vision_ai.human_eval import build_survey, save_rgb_pngs, score_survey


@pytest.fixture
def images(tmp_path):
    # Index i of every list shows the same tile: real, pix2pix and diffusion versions.
    rgb = torch.randint(0, 256, (36, 3, 16, 16), dtype=torch.uint8)
    real = save_rgb_pngs(rgb[:12], tmp_path / "src", "real")
    fakes = {
        "pix2pix": save_rgb_pngs(rgb[12:24], tmp_path / "src", "pix2pix"),
        "diffusion": save_rgb_pngs(rgb[24:], tmp_path / "src", "diffusion"),
    }
    return real, fakes


def read_key(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_answers(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rater", "item", "answer"])
        writer.writerows(rows)
    return path


def survey(images, tmp_path, n_per_source=3):
    real, fakes = images
    return build_survey(real, fakes, n_per_source, tmp_path / "survey", seed=0)


def test_save_rgb_pngs(images):
    real, _ = images
    assert [p.name for p in real[:2]] == ["real_000.png", "real_001.png"]
    assert all(p.stat().st_size > 0 for p in real)


def test_build_survey_balanced_and_shuffled(images, tmp_path):
    key = read_key(survey(images, tmp_path))
    assert [row["item"] for row in key] == [f"q{i:02d}" for i in range(1, 10)]
    assert sorted(row["label"] for row in key) == ["diffusion"] * 3 + ["pix2pix"] * 3 + ["real"] * 3
    assert all((tmp_path / "survey" / f"{row['item']}.png").exists() for row in key)
    assert [row["label"] for row in key] != ["real"] * 3 + ["pix2pix"] * 3 + ["diffusion"] * 3


def test_build_survey_never_shows_two_versions_of_a_tile(images, tmp_path):
    tiles = [row["source"].rsplit("_", 1)[1] for row in read_key(survey(images, tmp_path, 4))]
    assert len(set(tiles)) == len(tiles) == 12


def test_build_survey_rejects_too_few_tiles(images, tmp_path):
    with pytest.raises(ValueError, match="tiles"):
        survey(images, tmp_path, n_per_source=5)  # needs 15 tiles, only 12


def test_score_all_fooled(images, tmp_path):
    key_path = survey(images, tmp_path)
    answers = [(r, row["item"], "real") for r in ("r1", "r2") for row in read_key(key_path)]
    result = score_survey(write_answers(tmp_path / "a.csv", answers), key_path)
    assert result["fooled_rate"] == {"diffusion": 1.0, "pix2pix": 1.0}
    assert result["accuracy"] == pytest.approx(1 / 3) and result["n_raters"] == 2


def test_score_perfect_raters(images, tmp_path):
    key_path = survey(images, tmp_path)
    answers = [
        ("r1", row["item"], "real" if row["label"] == "real" else "generated")
        for row in read_key(key_path)
    ]
    result = score_survey(write_answers(tmp_path / "a.csv", answers), key_path)
    assert result["accuracy"] == 1.0 and result["fooled_rate"] == {"diffusion": 0.0, "pix2pix": 0.0}


def test_score_rejects_unknown_item(images, tmp_path):
    key_path = survey(images, tmp_path)
    with pytest.raises(ValueError, match="q99"):
        score_survey(write_answers(tmp_path / "a.csv", [("r1", "q99", "real")]), key_path)


def test_score_rejects_bad_answer(images, tmp_path):
    key_path = survey(images, tmp_path)
    with pytest.raises(ValueError, match="maybe"):
        score_survey(write_answers(tmp_path / "a.csv", [("r1", "q01", "maybe")]), key_path)


def test_score_rejects_empty_answers(images, tmp_path):
    key_path = survey(images, tmp_path)
    with pytest.raises(ValueError, match="no answers"):
        score_survey(write_answers(tmp_path / "a.csv", []), key_path)

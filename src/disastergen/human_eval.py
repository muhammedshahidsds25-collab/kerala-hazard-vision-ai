"""Human evaluation: can people tell real after-disaster tiles from generated ones?

1. :func:`build_survey` mixes real and generated images, renames them ``q01.png``... in random
   order and writes the answer key.
2. Raters label every image "real" or "generated" (e.g. in a Google Form); their answers are
   collected in a CSV with columns ``rater,item,answer``.
3. :func:`score_survey` reports rater accuracy and the fooled rate: the share of generated
   images that raters labelled real (higher means more realistic).
"""

from __future__ import annotations

import csv
import random
import shutil
from collections.abc import Sequence
from pathlib import Path

import torch

ANSWERS = ("real", "generated")


def save_rgb_pngs(rgb: torch.Tensor, out_dir: Path, prefix: str) -> list[Path]:
    """Save a (N, 3, H, W) uint8 batch as ``{prefix}_000.png``, ``{prefix}_001.png``..."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, image in enumerate(rgb):
        path = out_dir / f"{prefix}_{i:03d}.png"
        Image.fromarray(image.permute(1, 2, 0).numpy()).save(path)
        paths.append(path)
    return paths


def build_survey(
    real: Sequence[Path], fake: Sequence[Path], n: int, out_dir: Path, seed: int
) -> Path:
    """Copy ``n / 2`` real and ``n / 2`` generated images in shuffled order; return the key path.

    ``real[i]`` and ``fake[i]`` must show the same tile. Each tile appears at most once, so a
    rater never sees the real and the generated version of one place side by side.

    Raises:
        ValueError: If ``n`` is odd or there are fewer than ``n`` tiles.
    """
    if n % 2:
        raise ValueError(f"n must be even to balance real and generated images, got {n}")
    n_tiles = min(len(real), len(fake))
    if n_tiles < n:
        raise ValueError(f"Need {n} distinct tiles for {n} images, got {n_tiles}")

    rng = random.Random(seed)
    tiles = rng.sample(range(n_tiles), n)
    half = n // 2
    items = [(real[i], 0) for i in tiles[:half]] + [(fake[i], 1) for i in tiles[half:]]
    rng.shuffle(items)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path = out_dir / "key.csv"
    with key_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "is_generated", "source"])
        for number, (source, is_generated) in enumerate(items, start=1):
            item = f"q{number:02d}"
            shutil.copy(source, out_dir / f"{item}.png")
            writer.writerow([item, is_generated, Path(source).name])
    return key_path


def score_survey(answers_csv: Path, key_csv: Path) -> dict[str, float]:
    """Score rater answers against the key.

    Returns:
        ``accuracy`` (share of correct labels), ``fooled_rate`` (share of generated images
        labelled real) and ``n_raters``.

    Raises:
        ValueError: If there are no answers, or an answer names an unknown item or is not
            "real"/"generated".
    """
    with Path(key_csv).open(newline="") as f:
        key = {row["item"]: row["is_generated"] == "1" for row in csv.DictReader(f)}
    with Path(answers_csv).open(newline="") as f:
        answers = list(csv.DictReader(f))
    if not answers:
        raise ValueError(f"{answers_csv} has no answers")

    correct = fooled = generated_seen = 0
    for row in answers:
        item, answer = row["item"].strip(), row["answer"].strip().lower()
        if item not in key:
            raise ValueError(f"Unknown survey item '{item}'")
        if answer not in ANSWERS:
            raise ValueError(f"Answer '{answer}' for {item} must be one of {ANSWERS}")
        said_generated = answer == "generated"
        correct += said_generated == key[item]
        if key[item]:
            generated_seen += 1
            fooled += not said_generated
    return {
        "accuracy": correct / len(answers),
        "fooled_rate": fooled / generated_seen if generated_seen else float("nan"),
        "n_raters": len({row["rater"] for row in answers}),
    }

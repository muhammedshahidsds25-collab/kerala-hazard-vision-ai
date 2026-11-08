"""Human evaluation: can people tell real after-disaster tiles from generated ones?

1. :func:`build_survey` mixes real images with images from each model (pix2pix, diffusion),
   renames them ``q01.png``... in random order and writes the answer key.
2. Raters label every image "real" or "generated" (e.g. in a Google Form); their answers are
   collected in a CSV with columns ``rater,item,answer``.
3. :func:`score_survey` reports rater accuracy and, per model, the fooled rate: the share of
   its images that raters labelled real (higher means more realistic).
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

REAL = "real"
ANSWERS = (REAL, "generated")


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
    real: Sequence[Path],
    fakes: Mapping[str, Sequence[Path]],
    n_per_source: int,
    out_dir: Path,
    seed: int,
) -> Path:
    """Mix ``n_per_source`` real images and as many from each model; return the key path.

    Index ``i`` of ``real`` and of every list in ``fakes`` must show the same tile. Each tile
    appears at most once, so a rater never compares two versions of one place.

    Raises:
        ValueError: If there are not enough distinct tiles.
    """
    sources: dict[str, Sequence[Path]] = {REAL: real, **fakes}
    n_tiles = min(len(paths) for paths in sources.values())
    needed = n_per_source * len(sources)
    if n_tiles < needed:
        raise ValueError(f"Need {needed} distinct tiles, got {n_tiles}")

    rng = random.Random(seed)
    tiles = rng.sample(range(n_tiles), needed)
    items = [
        (label, paths[tile])
        for k, (label, paths) in enumerate(sources.items())
        for tile in tiles[k * n_per_source : (k + 1) * n_per_source]
    ]
    rng.shuffle(items)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path = out_dir / "key.csv"
    with key_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "label", "source"])
        for number, (label, source) in enumerate(items, start=1):
            item = f"q{number:02d}"
            shutil.copy(source, out_dir / f"{item}.png")
            writer.writerow([item, label, Path(source).stem])
    return key_path


def score_survey(answers_csv: Path, key_csv: Path) -> dict:
    """Score rater answers against the key.

    Returns:
        ``accuracy`` (share of correct labels), ``fooled_rate`` per model (share of that
        model's images labelled real; higher means more realistic) and ``n_raters``.

    Raises:
        ValueError: If there are no answers, or an answer names an unknown item or is not
            "real"/"generated".
    """
    with Path(key_csv).open(newline="") as f:
        key = {row["item"]: row["label"] for row in csv.DictReader(f)}
    with Path(answers_csv).open(newline="") as f:
        answers = list(csv.DictReader(f))
    if not answers:
        raise ValueError(f"{answers_csv} has no answers")

    correct = 0
    shown: dict[str, int] = defaultdict(int)
    fooled: dict[str, int] = defaultdict(int)
    for row in answers:
        item, answer = row["item"].strip(), row["answer"].strip().lower()
        if item not in key:
            raise ValueError(f"Unknown survey item '{item}'")
        if answer not in ANSWERS:
            raise ValueError(f"Answer '{answer}' for {item} must be one of {ANSWERS}")
        label = key[item]
        correct += (answer == REAL) == (label == REAL)
        if label != REAL:
            shown[label] += 1
            fooled[label] += answer == REAL
    return {
        "accuracy": correct / len(answers),
        "fooled_rate": {model: fooled[model] / shown[model] for model in sorted(shown)},
        "n_raters": len({row["rater"] for row in answers}),
    }

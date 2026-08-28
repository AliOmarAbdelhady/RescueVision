"""File I/O helpers, path resolution, and corrupted file handling."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_image_pairs(
    images_dir: str | Path,
    pattern_pre: str = "*_pre_disaster.png",
    pattern_post: str = "*_post_disaster.png",
) -> list[tuple[Path, Path]]:
    """Find matching pre/post image pairs in a directory.

    Returns list of (pre_path, post_path) tuples.
    Logs warnings for unpaired images.
    """
    images_dir = Path(images_dir)
    pre_images = sorted(images_dir.glob(pattern_pre))
    pairs = []

    for pre_path in pre_images:
        post_path = Path(str(pre_path).replace("_pre_disaster", "_post_disaster"))
        if post_path.exists():
            pairs.append((pre_path, post_path))
        else:
            logger.warning("No matching post image for: %s", pre_path.name)

    return pairs


def find_label_for_image(image_path: str | Path, labels_dir: str | Path) -> Path | None:
    """Find the JSON label file corresponding to an image."""
    image_path = Path(image_path)
    labels_dir = Path(labels_dir)
    label_name = image_path.with_suffix(".json").name
    label_path = labels_dir / label_name

    if label_path.exists():
        return label_path

    logger.warning("Label not found for image: %s", image_path.name)
    return None


def load_json_safe(path: str | Path) -> dict | None:
    """Load JSON file with error handling. Returns None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logger.warning("Failed to load JSON %s: %s", path, e)
        return None


def find_files_with_labels(
    images_dir: str | Path,
    labels_dir: str | Path,
    pattern: str = "*.png",
) -> list[tuple[Path, Path]]:
    """Find images that have corresponding label files."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    results = []

    for img_path in sorted(images_dir.glob(pattern)):
        label_path = labels_dir / img_path.with_suffix(".json").name
        if label_path.exists():
            results.append((img_path, label_path))
        else:
            logger.debug("No label for: %s", img_path.name)

    return results

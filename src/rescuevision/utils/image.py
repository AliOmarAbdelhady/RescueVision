"""Image loading, resizing, and normalization utilities."""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path


def load_image(path: str | Path, mode: str = "RGB") -> np.ndarray:
    """Load image as numpy array (H, W, C) with given mode."""
    img = Image.open(path).convert(mode)
    return np.array(img)


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array as image. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype in (np.float32, np.float64):
        array = (array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask image as uint8 numpy array."""
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def load_npy(path: str | Path) -> np.ndarray:
    """Load a .npy file."""
    return np.load(str(path))


def save_npy(array: np.ndarray, path: str | Path) -> None:
    """Save numpy array as .npy file. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def resize_image(
    image: np.ndarray,
    size: tuple[int, int],
    interpolation: int = Image.BILINEAR,
) -> np.ndarray:
    """Resize image to (height, width)."""
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((size[1], size[0]), interpolation)
    return np.array(pil_img)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize mask with nearest-neighbor interpolation."""
    return resize_image(mask, size, interpolation=Image.NEAREST)


def normalize_image(
    image: np.ndarray,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    """Normalize image from [0,255] uint8 to [0,1] float, then subtract mean/std."""
    image = image.astype(np.float32) / 255.0
    if mean and std:
        image = (image - np.array(mean)) / np.array(std)
    return image

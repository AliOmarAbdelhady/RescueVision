"""Damage map builder: aggregate per-building labels into color-coded map."""

from __future__ import annotations

import numpy as np

from rescuevision.constants import DAMAGE_COLOR_MAP


def build_damage_map(
    building_mask: np.ndarray,
    building_labels: list[dict],
) -> np.ndarray:
    """Build a color-coded damage map from building mask and per-building labels.

    Args:
        building_mask: Binary building mask.
        building_labels: List of dicts with 'label' (instance) and 'damage_class'.

    Returns:
        Damage class mask with values 0-4.
    """
    from skimage.measure import label

    labeled = label(building_mask)
    damage_map = np.zeros_like(building_mask, dtype=np.uint8)

    for bl in building_labels:
        instance_mask = labeled == bl["label"]
        damage_map[instance_mask] = bl["damage_class"]

    return damage_map


def damage_map_to_rgb(damage_map: np.ndarray) -> np.ndarray:
    """Convert damage class map to RGB visualization."""
    h, w = damage_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in DAMAGE_COLOR_MAP.items():
        rgb[damage_map == cls_id] = color

    return rgb


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Create semi-transparent overlay of damage map on image."""
    colored = damage_map_to_rgb(mask)
    overlay = (image * (1 - alpha) + colored * alpha).astype(np.uint8)
    return overlay

"""Post-processing utilities for prediction masks."""

from __future__ import annotations

import numpy as np
from skimage.morphology import area_opening, binary_closing, binary_dilation, disk
from skimage.measure import label, regionprops


def remove_small_objects(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Remove small connected components from binary mask."""
    cleaned = area_opening(mask, area_threshold=min_area)
    return cleaned.astype(np.uint8)


def fill_small_holes(mask: np.ndarray, disk_radius: int = 3) -> np.ndarray:
    """Fill small holes in binary mask using morphological closing."""
    closed = binary_closing(mask, footprint=disk(disk_radius))
    return closed.astype(np.uint8)


def refine_mask(mask: np.ndarray, min_area: int = 100, close_radius: int = 3) -> np.ndarray:
    """Full refinement: remove small objects and fill holes."""
    mask = remove_small_objects(mask, min_area)
    mask = fill_small_holes(mask, close_radius)
    return mask


def extract_building_instances(mask: np.ndarray) -> list[dict]:
    """Extract individual building instances from binary mask.

    Returns list of dicts with:
        - label: instance label
        - bbox: (y1, x1, y2, x2)
        - area: pixel count
        - centroid: (y, x)
    """
    labeled = label(mask)
    instances = []

    for region in regionprops(labeled):
        instances.append({
            "label": region.label,
            "bbox": region.bbox,  # (y1, x1, y2, x2)
            "area": region.area,
            "centroid": region.centroid,
        })

    return instances


def majority_vote_per_instance(
    label_map: np.ndarray,
    instance_mask: np.ndarray,
) -> np.ndarray:
    """Assign majority label to each instance in the instance mask.

    Args:
        label_map: Per-pixel label predictions.
        instance_mask: Instance segmentation mask.

    Returns:
        Instance map with majority labels.
    """
    result = np.zeros_like(label_map)
    labeled = label(instance_mask)

    for region in regionprops(labeled):
        region_pixels = labeled == region.label
        labels_in_region = label_map[region_pixels]
        valid_labels = labels_in_region[labels_in_region > 0]

        if len(valid_labels) > 0:
            majority = np.bincount(valid_labels).argmax()
            result[region_pixels] = majority

    return result

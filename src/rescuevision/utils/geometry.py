"""Polygon-to-mask rasterization and bounding box utilities."""

from __future__ import annotations

import numpy as np
import rasterio.features
from shapely.geometry import Polygon, mapping
from shapely.validation import make_valid


def polygon_to_mask(
    polygons: list[tuple[Polygon, int]],
    height: int,
    width: int,
    default_value: int = 0,
) -> np.ndarray:
    """Rasterize a list of (polygon, value) pairs into a mask.

    Args:
        polygons: List of (Shapely polygon, integer value) tuples.
        height: Output mask height.
        width: Output mask width.
        default_value: Default fill value for pixels outside all polygons.

    Returns:
        2D numpy array of shape (height, width) with dtype uint8.
    """
    shapes = []
    for poly, value in polygons:
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        shapes.append((mapping(poly), value))

    if not shapes:
        return np.full((height, width), default_value, dtype=np.uint8)

    mask = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        fill=default_value,
        dtype=np.uint8,
    )
    return mask


def bbox_from_polygon(
    polygon: Polygon,
    margin: int = 16,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[int, int, int, int]:
    """Compute bounding box from polygon with optional margin and clipping.

    Returns:
        (x1, y1, x2, y2) tuple clipped to image bounds.
    """
    minx, miny, maxx, maxy = polygon.bounds
    x1 = max(0, int(minx) - margin)
    y1 = max(0, int(miny) - margin)
    x2 = int(maxx) + margin
    y2 = int(maxy) + margin

    if image_width is not None:
        x2 = min(x2, image_width)
    if image_height is not None:
        y2 = min(y2, image_height)

    return x1, y1, x2, y2


def polygon_area(polygon: Polygon) -> float:
    """Return the area of a polygon."""
    return polygon.area

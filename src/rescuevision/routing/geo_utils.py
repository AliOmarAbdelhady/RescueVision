"""Geospatial utilities for coordinate conversion and mapping."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def pixel_to_geo(
    pixel_coords: Tuple[int, int],
    transform: list | np.ndarray,
) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates using an affine transform.

    Args:
        pixel_coords: (row, col) pixel coordinates.
        transform: Affine transform [a, b, c, d, e, f] where:
            geo_x = a * col + b * row + c
            geo_y = d * col + e * row + f

    Returns:
        (longitude, latitude) or (easting, northing).
    """
    row, col = pixel_coords
    if len(transform) == 6:
        geo_x = transform[0] * col + transform[1] * row + transform[2]
        geo_y = transform[3] * col + transform[4] * row + transform[5]
    elif len(transform) == 9:
        # Full 3x3 affine matrix
        geo_x = transform[0] * col + transform[1] * row + transform[2]
        geo_y = transform[3] * col + transform[4] * row + transform[5]
    else:
        raise ValueError(f"Expected transform of length 6 or 9, got {len(transform)}")

    return geo_x, geo_y


def geo_to_pixel(
    geo_coords: Tuple[float, float],
    transform: list | np.ndarray,
) -> Tuple[int, int]:
    """Convert geographic coordinates to pixel coordinates.

    Args:
        geo_coords: (x, y) geographic coordinates.
        transform: Inverse affine transform.

    Returns:
        (row, col) pixel coordinates.
    """
    x, y = geo_coords
    a, b, c, d, e, f = transform[:6]

    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Transform is not invertible")

    inv_det = 1.0 / det
    col = inv_det * (e * (x - c) - b * (y - f))
    row = inv_det * (-d * (x - c) + a * (y - f))

    return int(round(row)), int(round(col))


def compute_pixel_scale(transform: list | np.ndarray) -> float:
    """Compute the pixel scale (meters per pixel) from an affine transform.

    Args:
        transform: Affine transform coefficients.

    Returns:
        Scale in ground units per pixel.
    """
    a = transform[0]
    b = transform[1]
    return float(np.sqrt(a**2 + b**2))


def create_simple_transform(
    origin_x: float,
    origin_y: float,
    pixel_size: float,
    image_height: int,
) -> list[float]:
    """Create a simple north-up affine transform.

    Args:
        origin_x: X coordinate of top-left corner.
        origin_y: Y coordinate of top-left corner.
        pixel_size: Pixel size in ground units.
        image_height: Image height in pixels.

    Returns:
        Affine transform coefficients [a, b, c, d, e, f].
    """
    return [
        pixel_size,       # a: pixel width
        0.0,              # b: row rotation
        origin_x,         # c: x origin
        0.0,              # d: column rotation
        -pixel_size,      # e: pixel height (negative for north-up)
        origin_y + pixel_size * image_height,  # f: y origin
    ]


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Calculate the great-circle distance between two points (km).

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        Distance in kilometers.
    """
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

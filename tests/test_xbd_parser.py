"""Unit tests for xBD JSON parser and mask generation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rescuevision.constants import DAMAGE_CLASS_TO_ID, IGNORE_INDEX
from rescuevision.data.xbd_parser import BuildingAnnotation, parse_xbd_json
from rescuevision.utils.geometry import bbox_from_polygon, polygon_to_mask
from shapely.geometry import Polygon


@pytest.fixture
def sample_xbd_json(tmp_path):
    """Create a sample xBD JSON file for testing."""
    data = {
        "metadata": {
            "disaster_type": "earthquake",
            "disaster_name": "test-earthquake",
            "img_fn": "test-earthquake_00000001_pre_disaster.png",
        },
        "features": {
            "xy": [
                {
                    "properties": {
                        "uid": "bldg_001",
                        "subtype": "no-damage",
                    },
                    "wkt": "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))",
                },
                {
                    "properties": {
                        "uid": "bldg_002",
                        "subtype": "major-damage",
                    },
                    "wkt": "POLYGON ((300 300, 400 300, 400 400, 300 400, 300 300))",
                },
            ]
        },
    }
    json_path = tmp_path / "test-earthquake_00000001_pre_disaster.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def test_parse_xbd_json(sample_xbd_json):
    """Test basic JSON parsing."""
    tile = parse_xbd_json(sample_xbd_json)

    assert tile.image_id == "test-earthquake_00000001"
    assert tile.disaster_name == "test-earthquake"
    assert tile.disaster_type == "earthquake"
    assert tile.num_buildings == 2

    assert tile.buildings[0].polygon_id == "bldg_001"
    assert tile.buildings[0].damage_label == "no-damage"
    assert tile.buildings[0].damage_id == 1

    assert tile.buildings[1].polygon_id == "bldg_002"
    assert tile.buildings[1].damage_label == "major-damage"
    assert tile.buildings[1].damage_id == 3


def test_damage_distribution(sample_xbd_json):
    """Test damage distribution computation."""
    tile = parse_xbd_json(sample_xbd_json)
    dist = tile.damage_distribution

    assert dist["no-damage"] == 1
    assert dist["major-damage"] == 1


def test_polygon_to_mask():
    """Test polygon rasterization."""
    poly1 = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])
    poly2 = Polygon([(110, 110), (190, 110), (190, 190), (110, 190)])

    mask = polygon_to_mask([(poly1, 1), (poly2, 2)], height=200, width=200)

    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert mask[50, 50] == 1  # Inside poly1
    assert mask[150, 150] == 2  # Inside poly2
    assert mask[0, 0] == 0  # Background


def test_polygon_to_mask_empty():
    """Test empty polygon list returns zero mask."""
    mask = polygon_to_mask([], height=100, width=100)
    assert mask.shape == (100, 100)
    assert mask.sum() == 0


def test_bbox_from_polygon():
    """Test bounding box computation with margin."""
    poly = Polygon([(50, 50), (150, 50), (150, 150), (50, 150)])
    x1, y1, x2, y2 = bbox_from_polygon(poly, margin=10, image_height=200, image_width=200)

    assert x1 == 40
    assert y1 == 40
    assert x2 == 160
    assert y2 == 160


def test_bbox_clipping():
    """Test bounding box clipping to image bounds."""
    poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    x1, y1, x2, y2 = bbox_from_polygon(poly, margin=10, image_height=25, image_width=25)

    assert x1 == 0  # Clipped to 0
    assert y1 == 0
    assert x2 == 25  # Clipped to width
    assert y2 == 25


def test_parse_post_disaster_json(tmp_path):
    """Test parsing a post-disaster JSON with damage labels."""
    data = {
        "metadata": {
            "disaster_type": "hurricane",
            "disaster_name": "test-hurricane",
        },
        "features": {
            "xy": [
                {
                    "properties": {"uid": "b1", "subtype": "destroyed"},
                    "wkt": "POLYGON ((0 0, 50 0, 50 50, 0 50, 0 0))",
                },
                {
                    "properties": {"uid": "b2", "subtype": "un-classified"},
                    "wkt": "POLYGON ((60 60, 100 60, 100 100, 60 100, 60 60))",
                },
            ]
        },
    }
    json_path = tmp_path / "test-hurricane_00000001_post_disaster.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    tile = parse_xbd_json(json_path)
    assert tile.buildings[0].damage_id == 4  # destroyed
    assert tile.buildings[1].damage_id == IGNORE_INDEX  # un-classified

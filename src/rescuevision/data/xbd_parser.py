"""Parse xBD/xView2 JSON annotation files into structured building annotations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shapely import wkt
from shapely.geometry import Polygon
from shapely.validation import make_valid

from rescuevision.constants import DAMAGE_CLASS_TO_ID

logger = logging.getLogger(__name__)


@dataclass
class BuildingAnnotation:
    """A single building annotation from xBD JSON."""

    image_id: str
    polygon_id: str
    polygon: Polygon
    damage_label: Optional[str] = None
    damage_id: Optional[int] = None
    disaster_name: Optional[str] = None
    disaster_type: Optional[str] = None
    image_path: Optional[Path] = None
    label_path: Optional[Path] = None


@dataclass
class TileAnnotations:
    """All annotations for a single tile (image)."""

    image_id: str
    disaster_name: str = ""
    disaster_type: str = ""
    image_path: Optional[Path] = None
    label_path: Optional[Path] = None
    buildings: list[BuildingAnnotation] = field(default_factory=list)

    @property
    def num_buildings(self) -> int:
        return len(self.buildings)

    @property
    def damage_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for b in self.buildings:
            label = b.damage_label or "un-classified"
            dist[label] = dist.get(label, 0) + 1
        return dist


def parse_xbd_json(
    json_path: str | Path,
    image_path: str | Path | None = None,
) -> TileAnnotations:
    """Parse a single xBD JSON annotation file.

    The xBD JSON format:
    {
        "metadata": {
            "disaster_type": "earthquake",
            "disaster_name": "guatemala-earthquake",
            "img_fn": "guatemala-earthquake_00000000_pre_disaster.png"
        },
        "features": {
            "xy": [
                {
                    "properties": {
                        "uid": "abc123",
                        "subtype": "no-damage"
                    },
                    "wkt": "POLYGON ((x1 y1, x2 y2, ...))"
                }
            ]
        }
    }

    Args:
        json_path: Path to the xBD JSON file.
        image_path: Optional path to the corresponding image file.

    Returns:
        TileAnnotations with parsed building annotations.
    """
    json_path = Path(json_path)
    json_name = json_path.stem  # e.g. "guatemala-earthquake_00000000_pre_disaster"

    # Derive image_id from filename: everything except _pre_disaster / _post_disaster
    if "_pre_disaster" in json_name:
        image_id = json_name.replace("_pre_disaster", "")
    elif "_post_disaster" in json_name:
        image_id = json_name.replace("_post_disaster", "")
    else:
        image_id = json_name

    with open(json_path, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    disaster_name = metadata.get("disaster_name", "")
    disaster_type = metadata.get("disaster_type", "")

    tile = TileAnnotations(
        image_id=image_id,
        disaster_name=disaster_name,
        disaster_type=disaster_type,
        image_path=Path(image_path) if image_path else None,
        label_path=json_path,
    )

    features_xy = data.get("features", {}).get("xy", [])

    for feat in features_xy:
        props = feat.get("properties", {})
        uid = props.get("uid", "")
        subtype = props.get("subtype", "un-classified")
        wkt_str = feat.get("wkt", "")

        if not wkt_str:
            logger.debug("Empty WKT for uid=%s in %s", uid, json_path.name)
            continue

        try:
            polygon = wkt.loads(wkt_str)
        except Exception as e:
            logger.warning("Failed to parse WKT for uid=%s: %s", uid, e)
            continue

        if not polygon.is_valid:
            polygon = make_valid(polygon)

        if polygon.is_empty:
            logger.debug("Empty polygon for uid=%s in %s", uid, json_path.name)
            continue

        # Ensure we have a Polygon (make_valid can return MultiPolygon)
        if polygon.geom_type == "MultiPolygon":
            # Take the largest polygon
            polygon = max(polygon.geoms, key=lambda g: g.area)

        damage_id = DAMAGE_CLASS_TO_ID.get(subtype, 255)

        building = BuildingAnnotation(
            image_id=image_id,
            polygon_id=uid,
            polygon=polygon,
            damage_label=subtype,
            damage_id=damage_id,
            disaster_name=disaster_name,
            disaster_type=disaster_type,
            image_path=Path(image_path) if image_path else None,
            label_path=json_path,
        )
        tile.buildings.append(building)

    return tile


def parse_all_labels(
    labels_dir: str | Path,
    images_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[TileAnnotations]:
    """Parse all JSON files in a labels directory.

    Args:
        labels_dir: Directory containing xBD JSON label files.
        images_dir: Optional corresponding images directory.
        limit: Maximum number of files to parse (for debug mode).

    Returns:
        List of TileAnnotations.
    """
    labels_dir = Path(labels_dir)
    json_files = sorted(labels_dir.glob("*.json"))

    if limit:
        json_files = json_files[:limit]

    if images_dir:
        images_dir = Path(images_dir)

    tiles = []
    for json_path in json_files:
        image_path = None
        if images_dir:
            candidate = images_dir / json_path.with_suffix(".png").name
            if candidate.exists():
                image_path = candidate

        try:
            tile = parse_xbd_json(json_path, image_path)
            tiles.append(tile)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path.name, e)

    logger.info("Parsed %d tile annotations from %s", len(tiles), labels_dir)
    return tiles

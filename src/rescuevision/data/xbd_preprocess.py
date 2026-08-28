"""xBD preprocessing: mask generation, crop creation, and metadata export."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from rescuevision.constants import CHANGED_CLASSES, IGNORE_INDEX
from rescuevision.data.xbd_parser import TileAnnotations, parse_all_labels
from rescuevision.utils.geometry import bbox_from_polygon, polygon_area, polygon_to_mask
from rescuevision.utils.image import load_image, save_image, save_npy
from rescuevision.utils.io import ensure_dir, find_image_pairs, find_label_for_image

logger = logging.getLogger(__name__)


def generate_masks_for_tile(
    pre_tile: TileAnnotations,
    post_tile: TileAnnotations | None,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    """Generate building, damage, and change masks for a single tile.

    Args:
        pre_tile: Parsed pre-disaster annotations (building polygons).
        post_tile: Parsed post-disaster annotations (damage labels). Can be None.
        height: Image height.
        width: Image width.

    Returns:
        Dict with keys: building_binary, damage_multiclass, change_binary.
    """
    # Binary building mask from pre-disaster polygons
    building_polys = [(b.polygon, 1) for b in pre_tile.buildings]
    building_mask = polygon_to_mask(building_polys, height, width)

    # Multiclass damage mask from post-disaster annotations
    damage_mask = np.zeros((height, width), dtype=np.uint8)
    change_mask = np.zeros((height, width), dtype=np.uint8)

    if post_tile and post_tile.buildings:
        damage_polys = []
        change_polys = []

        for b in post_tile.buildings:
            if b.damage_id is None or b.damage_id == IGNORE_INDEX:
                # Use ignore_index for unclassified buildings
                damage_polys.append((b.polygon, IGNORE_INDEX))
            else:
                damage_polys.append((b.polygon, b.damage_id))
                # Change: any damaged building counts as changed
                if b.damage_id in CHANGED_CLASSES:
                    change_polys.append((b.polygon, 1))

        if damage_polys:
            damage_mask = polygon_to_mask(damage_polys, height, width)
        if change_polys:
            change_mask = polygon_to_mask(change_polys, height, width)

    return {
        "building_binary": building_mask,
        "damage_multiclass": damage_mask,
        "change_binary": change_mask,
    }


def generate_crops_for_tile(
    pre_tile: TileAnnotations,
    post_tile: TileAnnotations | None,
    pre_image: np.ndarray,
    post_image: np.ndarray | None,
    margin: int = 16,
) -> list[dict]:
    """Generate per-building crops from pre and post images.

    Args:
        pre_tile: Pre-disaster annotations.
        post_tile: Post-disaster annotations (may be None).
        pre_image: Pre-disaster image array (H, W, C).
        post_image: Post-disaster image array (H, W, C).
        margin: Padding around bounding box.

    Returns:
        List of crop metadata dicts.
    """
    h, w = pre_image.shape[:2]
    crops = []

    # Build a lookup: polygon_id -> damage info from post_tile
    post_lookup = {}
    if post_tile:
        for b in post_tile.buildings:
            post_lookup[b.polygon_id] = b

    for pre_building in pre_tile.buildings:
        x1, y1, x2, y2 = bbox_from_polygon(
            pre_building.polygon, margin=margin, image_height=h, image_width=w
        )

        crop_h = y2 - y1
        crop_w = x2 - x1
        if crop_h < 8 or crop_w < 8:
            continue

        pre_crop = pre_image[y1:y2, x1:x2]
        post_crop = None
        if post_image is not None:
            post_crop = post_image[y1:y2, x1:x2]

        # Get damage label from post annotation if available
        post_building = post_lookup.get(pre_building.polygon_id)
        damage_label = post_building.damage_label if post_building else "un-classified"
        damage_id = post_building.damage_id if post_building else IGNORE_INDEX

        crops.append({
            "polygon_id": pre_building.polygon_id,
            "tile_id": pre_tile.image_id,
            "disaster_name": pre_tile.disaster_name,
            "disaster_type": pre_tile.disaster_type,
            "damage_label": damage_label,
            "damage_id": damage_id,
            "area": polygon_area(pre_building.polygon),
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "pre_crop": pre_crop,
            "post_crop": post_crop,
        })

    return crops


def preprocess_xbd(
    raw_dir: str | Path,
    out_dir: str | Path,
    image_size: int = 1024,
    make_crops: bool = True,
    make_visualizations: bool = True,
    crop_margin: int = 16,
    debug_limit: int | None = None,
) -> None:
    """Full xBD preprocessing pipeline.

    Reads raw xBD data, generates masks, crops, metadata, and splits.

    Args:
        raw_dir: Path to raw xBD data (contains train/hold/test dirs).
        out_dir: Path to write processed output.
        image_size: Target image size (images are NOT resized, this records original size).
        make_crops: Whether to generate per-building crops.
        make_visualizations: Whether to save sanity-check visualizations.
        crop_margin: Pixel margin around building bounding boxes.
        debug_limit: Limit number of tiles for debugging.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # Create output directories
    dirs = {
        "pre_images": ensure_dir(out_dir / "images" / "pre"),
        "post_images": ensure_dir(out_dir / "images" / "post"),
        "building_masks": ensure_dir(out_dir / "masks" / "building_binary"),
        "damage_masks": ensure_dir(out_dir / "masks" / "damage_multiclass"),
        "change_masks": ensure_dir(out_dir / "masks" / "change_binary"),
        "pre_crops": ensure_dir(out_dir / "crops" / "building_pre"),
        "post_crops": ensure_dir(out_dir / "crops" / "building_post"),
        "metadata": ensure_dir(out_dir / "metadata"),
        "figures": ensure_dir(out_dir / "debug_subset") if make_visualizations else None,
    }

    all_tiles_metadata = []
    all_buildings = []
    crop_counter = 0

    # Process each split
    for split_name in ["train", "hold", "test"]:
        split_images_dir = raw_dir / split_name / "images"
        split_labels_dir = raw_dir / split_name / "labels"

        if not split_images_dir.exists():
            logger.warning("Split directory not found: %s", split_images_dir)
            continue

        logger.info("Processing split: %s", split_name)

        # Find image pairs
        image_pairs = find_image_pairs(split_images_dir)
        if debug_limit:
            image_pairs = image_pairs[:debug_limit]

        # Parse labels
        pre_tiles_dict = {}
        post_tiles_dict = {}

        pre_labels = find_label_for_image(
            # Dummy call to get pattern
            split_images_dir / "dummy.png", split_labels_dir
        )

        # Parse pre-disaster labels
        for pre_img, _ in image_pairs:
            label_path = split_labels_dir / pre_img.with_suffix(".json").name
            if label_path.exists():
                try:
                    tile = parse_all_labels(split_labels_dir, split_images_dir, limit=None)
                    # We parse individually to get the right mapping
                except Exception:
                    pass

        # Re-parse properly: iterate per pair
        for pre_img_path, post_img_path in tqdm(image_pairs, desc=f"Processing {split_name}"):
            tile_id = pre_img_path.stem.replace("_pre_disaster", "")

            # Parse labels
            pre_label_path = split_labels_dir / pre_img_path.with_suffix(".json").name
            post_label_path = split_labels_dir / post_img_path.with_suffix(".json").name

            if not pre_label_path.exists():
                logger.debug("Missing pre label: %s", pre_label_path.name)
                continue

            from rescuevision.data.xbd_parser import parse_xbd_json

            try:
                pre_tile = parse_xbd_json(pre_label_path, pre_img_path)
            except Exception as e:
                logger.warning("Error parsing pre label %s: %s", pre_label_path.name, e)
                continue

            post_tile = None
            if post_label_path.exists():
                try:
                    post_tile = parse_xbd_json(post_label_path, post_img_path)
                except Exception as e:
                    logger.warning("Error parsing post label %s: %s", post_label_path.name, e)

            # Load images
            try:
                pre_image = load_image(pre_img_path)
            except Exception as e:
                logger.warning("Error loading pre image %s: %s", pre_img_path.name, e)
                continue

            post_image = None
            try:
                post_image = load_image(post_img_path)
            except Exception as e:
                logger.warning("Error loading post image %s: %s", post_img_path.name, e)

            h, w = pre_image.shape[:2]

            # Generate masks
            masks = generate_masks_for_tile(pre_tile, post_tile, h, w)

            # Save images (copy/symlink to processed dir)
            save_npy(masks["building_binary"], dirs["building_masks"] / f"{tile_id}.npy")
            save_npy(masks["damage_multiclass"], dirs["damage_masks"] / f"{tile_id}.npy")
            save_npy(masks["change_binary"], dirs["change_masks"] / f"{tile_id}.npy")

            # Copy images as numpy for speed
            save_npy(pre_image, dirs["pre_images"] / f"{tile_id}.npy")
            if post_image is not None:
                save_npy(post_image, dirs["post_images"] / f"{tile_id}.npy")

            # Tile metadata
            all_tiles_metadata.append({
                "tile_id": tile_id,
                "split": split_name,
                "disaster_name": pre_tile.disaster_name,
                "disaster_type": pre_tile.disaster_type,
                "num_buildings": pre_tile.num_buildings,
                "image_height": h,
                "image_width": w,
                "damage_distribution": str(pre_tile.damage_distribution),
            })

            # Generate crops
            if make_crops and post_image is not None:
                crops = generate_crops_for_tile(
                    pre_tile, post_tile, pre_image, post_image, margin=crop_margin
                )
                for crop in crops:
                    crop_id = f"crop_{crop_counter:06d}"
                    crop_counter += 1

                    save_npy(crop["pre_crop"], dirs["pre_crops"] / f"{crop_id}.npy")
                    if crop["post_crop"] is not None:
                        save_npy(crop["post_crop"], dirs["post_crops"] / f"{crop_id}.npy")

                    all_buildings.append({
                        "crop_id": crop_id,
                        "tile_id": crop["tile_id"],
                        "polygon_id": crop["polygon_id"],
                        "pre_crop_path": str(dirs["pre_crops"] / f"{crop_id}.npy"),
                        "post_crop_path": str(dirs["post_crops"] / f"{crop_id}.npy"),
                        "label": crop["damage_label"],
                        "label_id": crop["damage_id"],
                        "area": crop["area"],
                        "bbox_x1": crop["bbox_x1"],
                        "bbox_y1": crop["bbox_y1"],
                        "bbox_x2": crop["bbox_x2"],
                        "bbox_y2": crop["bbox_y2"],
                        "disaster_name": crop["disaster_name"],
                        "disaster_type": crop["disaster_type"],
                        "split": split_name,
                    })

            # Sanity visualizations
            if make_visualizations and len(all_tiles_metadata) <= 20:
                _save_visualization(
                    pre_image, post_image, masks, tile_id, dirs["figures"]
                )

    # Save metadata CSVs
    if all_tiles_metadata:
        tiles_df = pd.DataFrame(all_tiles_metadata)
        tiles_df.to_csv(dirs["metadata"] / "tiles.csv", index=False)
        logger.info("Saved %d tile records", len(tiles_df))

    if all_buildings:
        buildings_df = pd.DataFrame(all_buildings)
        buildings_df.to_csv(dirs["metadata"] / "buildings.csv", index=False)
        logger.info("Saved %d building crop records", len(buildings_df))

    # Generate event-aware splits
    from rescuevision.data.split_utils import generate_event_aware_splits

    if all_tiles_metadata:
        generate_event_aware_splits(
            tiles_csv_path=dirs["metadata"] / "tiles.csv",
            output_path=dirs["metadata"] / "splits_event_aware.csv",
        )

    # Class distribution
    if all_buildings:
        _save_class_distribution(all_buildings, dirs["metadata"])

    logger.info("Preprocessing complete. Output: %s", out_dir)


def _save_visualization(
    pre_image: np.ndarray,
    post_image: np.ndarray | None,
    masks: dict[str, np.ndarray],
    tile_id: str,
    figures_dir: Path,
) -> None:
    """Save a side-by-side visualization for sanity checking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].imshow(pre_image)
    axes[0, 0].set_title(f"Pre-disaster: {tile_id}")
    axes[0, 0].axis("off")

    if post_image is not None:
        axes[0, 1].imshow(post_image)
        axes[0, 1].set_title("Post-disaster")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(masks["building_binary"], cmap="gray")
    axes[1, 0].set_title("Building Mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(masks["damage_multiclass"], cmap="tab10", vmin=0, vmax=4)
    axes[1, 1].set_title("Damage Mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{tile_id}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_class_distribution(buildings: list[dict], metadata_dir: Path) -> None:
    """Save class distribution statistics."""
    df = pd.DataFrame(buildings)
    dist = df["label"].value_counts()
    dist.to_csv(metadata_dir / "class_distribution.csv")
    logger.info("Class distribution:\n%s", dist.to_string())

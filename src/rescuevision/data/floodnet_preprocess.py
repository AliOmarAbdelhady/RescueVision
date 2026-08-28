"""FloodNet preprocessing and dataset class for flood/road segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from rescuevision.constants import FLOODNET_CLASSES, NUM_FLOODNET_CLASSES
from rescuevision.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class FloodNetDataset(Dataset):
    """FloodNet semantic segmentation dataset.

    Expects masks with pixel values 0-9 corresponding to FLOODNET_CLASSES.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform=None,
        image_size: int = 512,
        debug_limit: int | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size

        # Find image-mask pairs
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")) + sorted(self.images_dir.glob("*.png")):
            mask_path = self.masks_dir / img_path.with_suffix(".png").name
            if not mask_path.exists():
                mask_path = self.masks_dir / img_path.with_suffix(".jpg").name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if debug_limit:
            self.samples = self.samples[:debug_limit]

        logger.info("FloodNetDataset: %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Validate mask values
        unique_vals = np.unique(mask)
        invalid = set(unique_vals.tolist()) - set(range(NUM_FLOODNET_CLASSES))
        if invalid:
            logger.debug("Invalid mask values in %s: %s", mask_path.name, invalid)
            mask = np.clip(mask, 0, NUM_FLOODNET_CLASSES - 1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        tile_id = img_path.stem
        return {"image": image, "mask": mask, "tile_id": tile_id}


def prepare_floodnet(
    raw_dir: str | Path,
    out_dir: str | Path,
    debug_limit: int | None = None,
) -> None:
    """Prepare FloodNet dataset: validate masks, generate file lists.

    Args:
        raw_dir: Path to raw FloodNet data (train/val/test with images/ and masks/).
        out_dir: Path to write processed output.
        debug_limit: Limit samples for debugging.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir / "metadata")

    for split in ["train", "val", "test"]:
        images_dir = raw_dir / split / "images"
        masks_dir = raw_dir / split / "masks"

        if not images_dir.exists():
            logger.warning("Split not found: %s", images_dir)
            continue

        # Validate and count
        records = []
        img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        if debug_limit:
            img_files = img_files[:debug_limit]

        for img_path in img_files:
            mask_path = masks_dir / img_path.with_suffix(".png").name
            if not mask_path.exists():
                mask_path = masks_dir / img_path.with_suffix(".jpg").name

            if mask_path.exists():
                records.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "split": split,
                    "tile_id": img_path.stem,
                })

        if records:
            df = pd.DataFrame(records)
            df.to_csv(out_dir / "metadata" / f"{split}_files.csv", index=False)
            logger.info("FloodNet %s: %d samples", split, len(records))

    logger.info("FloodNet preparation complete.")

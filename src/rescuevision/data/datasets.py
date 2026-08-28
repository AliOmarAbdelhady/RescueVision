"""PyTorch Dataset classes for xBD segmentation, damage segmentation, change detection, and classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class XBDSegmentationDataset(Dataset):
    """Dataset for building segmentation (binary).

    Loads pre-disaster images and binary building masks.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_csv: str | Path,
        split: str = "train",
        mask_type: str = "building_binary",
        transform: Callable | None = None,
        image_size: int = 512,
        debug_limit: int | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.images_dir = self.processed_dir / "images" / "pre"
        self.masks_dir = self.processed_dir / "masks" / mask_type
        self.transform = transform
        self.image_size = image_size

        # Load split CSV and filter
        splits_df = pd.read_csv(split_csv)
        self.tile_ids = splits_df[splits_df["split"] == split]["tile_id"].tolist()

        if debug_limit:
            self.tile_ids = self.tile_ids[:debug_limit]

        # Filter to existing files
        existing = []
        for tid in self.tile_ids:
            if (self.images_dir / f"{tid}.npy").exists() and (
                self.masks_dir / f"{tid}.npy"
            ).exists():
                existing.append(tid)
        self.tile_ids = existing

        logger.info(
            "XBDSegmentationDataset: split=%s, %d tiles", split, len(self.tile_ids)
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tile_id = self.tile_ids[idx]

        image = np.load(self.images_dir / f"{tile_id}.npy")
        mask = np.load(self.masks_dir / f"{tile_id}.npy")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, "tile_id": tile_id}


class XBDDamageSegmentationDataset(Dataset):
    """Dataset for damage segmentation (multiclass, 6-channel input).

    Loads pre+post images concatenated as 6-channel input and multiclass damage masks.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_csv: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: int = 512,
        debug_limit: int | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.pre_dir = self.processed_dir / "images" / "pre"
        self.post_dir = self.processed_dir / "images" / "post"
        self.masks_dir = self.processed_dir / "masks" / "damage_multiclass"
        self.transform = transform
        self.image_size = image_size

        splits_df = pd.read_csv(split_csv)
        self.tile_ids = splits_df[splits_df["split"] == split]["tile_id"].tolist()

        if debug_limit:
            self.tile_ids = self.tile_ids[:debug_limit]

        existing = []
        for tid in self.tile_ids:
            if (
                (self.pre_dir / f"{tid}.npy").exists()
                and (self.post_dir / f"{tid}.npy").exists()
                and (self.masks_dir / f"{tid}.npy").exists()
            ):
                existing.append(tid)
        self.tile_ids = existing

        logger.info(
            "XBDDamageSegmentationDataset: split=%s, %d tiles", split, len(self.tile_ids)
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tile_id = self.tile_ids[idx]

        pre_image = np.load(self.pre_dir / f"{tile_id}.npy")
        post_image = np.load(self.post_dir / f"{tile_id}.npy")
        mask = np.load(self.masks_dir / f"{tile_id}.npy")

        if self.transform:
            aug_pre = self.transform(image=pre_image, mask=mask)
            pre_image = aug_pre["image"]
            aug_post = self.transform(image=post_image, mask=mask)
            post_image = aug_post["image"]
            mask = aug_pre["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        # Concatenate pre and post along channel dimension -> 6 channels
        if isinstance(pre_image, torch.Tensor) and isinstance(post_image, torch.Tensor):
            image = torch.cat([pre_image, post_image], dim=0)
        else:
            image = np.concatenate([pre_image, post_image], axis=2)

        return {"image": image, "mask": mask, "tile_id": tile_id}


class XBDChangeDetectionDataset(Dataset):
    """Dataset for change detection.

    Returns pre and post images separately with binary change mask.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_csv: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: int = 512,
        debug_limit: int | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.pre_dir = self.processed_dir / "images" / "pre"
        self.post_dir = self.processed_dir / "images" / "post"
        self.masks_dir = self.processed_dir / "masks" / "change_binary"
        self.transform = transform
        self.image_size = image_size

        splits_df = pd.read_csv(split_csv)
        self.tile_ids = splits_df[splits_df["split"] == split]["tile_id"].tolist()

        if debug_limit:
            self.tile_ids = self.tile_ids[:debug_limit]

        existing = []
        for tid in self.tile_ids:
            if (
                (self.pre_dir / f"{tid}.npy").exists()
                and (self.post_dir / f"{tid}.npy").exists()
                and (self.masks_dir / f"{tid}.npy").exists()
            ):
                existing.append(tid)
        self.tile_ids = existing

        logger.info(
            "XBDChangeDetectionDataset: split=%s, %d tiles", split, len(self.tile_ids)
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tile_id = self.tile_ids[idx]

        pre_image = np.load(self.pre_dir / f"{tile_id}.npy")
        post_image = np.load(self.post_dir / f"{tile_id}.npy")
        mask = np.load(self.masks_dir / f"{tile_id}.npy")

        if self.transform:
            # Apply same spatial transform to both images
            aug = self.transform(image=pre_image, mask=mask)
            pre_image = aug["image"]
            mask_out = aug["mask"]

            aug2 = self.transform(image=post_image, mask=mask)
            post_image = aug2["image"]
            mask = mask_out

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {
            "pre_image": pre_image,
            "post_image": post_image,
            "mask": mask,
            "tile_id": tile_id,
        }


class XBDDamageCropDataset(Dataset):
    """Dataset for building-level damage classification from crops.

    Loads pre/post building crops and returns damage label.
    """

    def __init__(
        self,
        crops_csv: str | Path,
        split_csv: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        image_size: int = 224,
        debug_limit: int | None = None,
    ):
        self.transform = transform
        self.image_size = image_size

        # Load crop metadata
        crops_df = pd.read_csv(crops_csv)
        splits_df = pd.read_csv(split_csv)

        # Get tile_ids for this split
        split_tiles = set(splits_df[splits_df["split"] == split]["tile_id"].tolist())

        # Filter crops by split
        self.crops_df = crops_df[crops_df["tile_id"].isin(split_tiles)].reset_index(
            drop=True
        )

        # Filter out un-classified labels
        self.crops_df = self.crops_df[self.crops_df["label_id"] != 255].reset_index(
            drop=True
        )

        if debug_limit:
            self.crops_df = self.crops_df.head(debug_limit)

        # Remap labels: 1->0, 2->1, 3->2, 4->3 for 4-class classification
        self.label_map = {1: 0, 2: 1, 3: 2, 4: 3}

        logger.info(
            "XBDDamageCropDataset: split=%s, %d crops", split, len(self.crops_df)
        )

    def __len__(self) -> int:
        return len(self.crops_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.crops_df.iloc[idx]

        pre_crop = np.load(row["pre_crop_path"])
        post_crop = np.load(row["post_crop_path"])

        label = self.label_map[row["label_id"]]

        if self.transform:
            pre_crop = self.transform(image=pre_crop)["image"]
            post_crop = self.transform(image=post_crop)["image"]

        return {
            "pre_crop": pre_crop,
            "post_crop": post_crop,
            "label": label,
            "crop_id": row["crop_id"],
            "tile_id": row["tile_id"],
        }

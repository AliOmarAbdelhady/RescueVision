#!/usr/bin/env python3
"""CLI script for training damage segmentation model."""

import argparse
import logging
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from rescuevision.constants import NUM_DAMAGE_CLASSES
from rescuevision.data.datasets import XBDDamageSegmentationDataset
from rescuevision.data.transforms import get_segmentation_transforms
from rescuevision.models.losses import WeightedCEDiceLoss, compute_class_weights
from rescuevision.models.unet import create_unet
from rescuevision.training.train_segmentation import train_segmentation
from rescuevision.utils.config import load_config
from rescuevision.utils.seed import set_seed
from rescuevision.utils.weight_init import adapt_first_conv


def main():
    parser = argparse.ArgumentParser(description="Train damage segmentation model")
    parser.add_argument("--config", type=str, default="configs/damage_seg_unet_6ch.yaml")
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)
    if args.debug_limit:
        config.setdefault("dataset", {})["debug_limit"] = args.debug_limit
    if args.max_epochs:
        config.setdefault("training", {})["epochs"] = args.max_epochs

    set_seed(config.get("seed", 42))

    # Compute class weights from training masks if requested
    class_weights = None
    if config.get("loss", {}).get("compute_class_weights", False):
        class_weights = _compute_mask_class_weights(
            config["dataset"]["processed_dir"],
            config["dataset"]["split_csv"],
        )
        logging.info("Class weights: %s", class_weights)

    # Create datasets
    train_dataset = XBDDamageSegmentationDataset(
        processed_dir=config["dataset"]["processed_dir"],
        split_csv=config["dataset"]["split_csv"],
        split="train",
        transform=get_segmentation_transforms(
            config.get("augmentation"), phase="train",
            image_size=config["dataset"].get("image_size", 512),
        ),
        image_size=config["dataset"].get("image_size", 512),
        debug_limit=args.debug_limit,
    )

    val_dataset = XBDDamageSegmentationDataset(
        processed_dir=config["dataset"]["processed_dir"],
        split_csv=config["dataset"]["split_csv"],
        split="val",
        transform=get_segmentation_transforms(
            config.get("augmentation"), phase="val",
            image_size=config["dataset"].get("image_size", 512),
        ),
        image_size=config["dataset"].get("image_size", 512),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["dataset"].get("batch_size", 4),
        shuffle=True, num_workers=config["dataset"].get("num_workers", 2), pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["dataset"].get("batch_size", 4),
        shuffle=False, num_workers=config["dataset"].get("num_workers", 2), pin_memory=True,
    )

    # Create model
    model = create_unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"].get("encoder_weights", "imagenet"),
        in_channels=6,
        classes=config["model"].get("classes", 5),
    )

    # Adapt first conv for 6-channel input
    if config["model"].get("adapt_first_conv", False):
        model = adapt_first_conv(model, old_channels=3, new_channels=6)
        logging.info("Adapted first conv for 6-channel input")

    # Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = WeightedCEDiceLoss(
        class_weights=class_weights,
        ce_weight=config.get("loss", {}).get("ce_weight", 0.5),
        dice_weight=config.get("loss", {}).get("dice_weight", 0.5),
        ignore_index=config.get("loss", {}).get("ignore_index", 255),
    )

    output_dir = config.get("checkpoints_dir", "outputs") / "damage_seg_unet"

    best_metrics = train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        output_dir=str(output_dir),
        is_binary=False,
        num_classes=NUM_DAMAGE_CLASSES,
        ignore_index=255,
    )

    print(f"\nTraining complete. Best metrics: {best_metrics}")


def _compute_mask_class_weights(
    processed_dir: str, split_csv: str, num_classes: int = 5
) -> torch.Tensor:
    """Scan training masks to compute class weights."""
    from pathlib import Path
    import pandas as pd

    processed_dir = Path(processed_dir)
    masks_dir = processed_dir / "masks" / "damage_multiclass"

    splits_df = pd.read_csv(split_csv)
    train_tiles = splits_df[splits_df["split"] == "train"]["tile_id"].tolist()

    all_labels = []
    for tid in train_tiles:
        mask_path = masks_dir / f"{tid}.npy"
        if mask_path.exists():
            mask = np.load(mask_path).flatten()
            valid = mask[mask != 255]
            all_labels.extend(valid.tolist())

    return compute_class_weights(all_labels, num_classes)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI script for training building segmentation model."""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

from rescuevision.data.datasets import XBDSegmentationDataset
from rescuevision.data.transforms import get_segmentation_transforms
from rescuevision.models.losses import BCEDiceLoss
from rescuevision.models.unet import create_unet
from rescuevision.training.train_segmentation import train_segmentation
from rescuevision.utils.config import load_config
from rescuevision.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train building segmentation model")
    parser.add_argument("--config", type=str, default="configs/building_seg_unet.yaml")
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

    # Create datasets
    train_dataset = XBDSegmentationDataset(
        processed_dir=config["dataset"]["processed_dir"],
        split_csv=config["dataset"]["split_csv"],
        split="train",
        mask_type=config["dataset"].get("mask_type", "building_binary"),
        transform=get_segmentation_transforms(
            config.get("augmentation"), phase="train",
            image_size=config["dataset"].get("image_size", 512),
        ),
        image_size=config["dataset"].get("image_size", 512),
        debug_limit=args.debug_limit,
    )

    val_dataset = XBDSegmentationDataset(
        processed_dir=config["dataset"]["processed_dir"],
        split_csv=config["dataset"]["split_csv"],
        split="val",
        mask_type=config["dataset"].get("mask_type", "building_binary"),
        transform=get_segmentation_transforms(
            config.get("augmentation"), phase="val",
            image_size=config["dataset"].get("image_size", 512),
        ),
        image_size=config["dataset"].get("image_size", 512),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataset"].get("batch_size", 8),
        shuffle=True,
        num_workers=config["dataset"].get("num_workers", 2),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["dataset"].get("batch_size", 8),
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 2),
        pin_memory=True,
    )

    # Create model
    model = create_unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"].get("encoder_weights", "imagenet"),
        in_channels=config["model"].get("in_channels", 3),
        classes=config["model"].get("classes", 1),
    )

    # Loss
    loss_fn = BCEDiceLoss(
        bce_weight=config.get("loss", {}).get("bce_weight", 0.5),
        dice_weight=config.get("loss", {}).get("dice_weight", 0.5),
    )

    # Output dir
    output_dir = config.get("checkpoints_dir", "outputs") / "building_seg_unet"

    best_metrics = train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        output_dir=str(output_dir),
        is_binary=True,
        num_classes=1,
    )

    print(f"\nTraining complete. Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()

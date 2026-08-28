#!/usr/bin/env python3
"""CLI script for training change detection model."""

import argparse
import logging
import sys

from torch.utils.data import DataLoader

from rescuevision.data.datasets import XBDChangeDetectionDataset
from rescuevision.data.transforms import get_segmentation_transforms
from rescuevision.models.losses import BCEDiceLoss
from rescuevision.models.siamese_unet import SiameseUNet
from rescuevision.training.train_change_detection import train_change_detection
from rescuevision.utils.config import load_config
from rescuevision.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument("--config", type=str, default="configs/change_siamese_unet.yaml")
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

    train_dataset = XBDChangeDetectionDataset(
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

    val_dataset = XBDChangeDetectionDataset(
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

    model = SiameseUNet(
        encoder_name=config["model"].get("encoder", "resnet34"),
        encoder_weights=config["model"].get("encoder_weights", "imagenet"),
        fusion_mode=config["model"].get("fusion_mode", "concat_absdiff"),
        classes=config["model"].get("classes", 1),
    )

    loss_fn = BCEDiceLoss(
        bce_weight=config.get("loss", {}).get("bce_weight", 0.5),
        dice_weight=config.get("loss", {}).get("dice_weight", 0.5),
    )

    output_dir = config.get("checkpoints_dir", "outputs") / "change_siamese_unet"

    best_metrics = train_change_detection(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        output_dir=str(output_dir),
    )

    print(f"\nTraining complete. Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()

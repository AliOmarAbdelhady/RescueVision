#!/usr/bin/env python3
"""CLI script for training FloodNet segmentation model."""

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from rescuevision.constants import NUM_FLOODNET_CLASSES
from rescuevision.data.floodnet_preprocess import FloodNetDataset
from rescuevision.data.transforms import get_segmentation_transforms
from rescuevision.models.losses import WeightedCEDiceLoss
from rescuevision.training.train_segmentation import train_segmentation
from rescuevision.utils.config import load_config
from rescuevision.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train FloodNet segmentation")
    parser.add_argument("--config", default="configs/floodnet_segformer.yaml")
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

    raw_dir = config["dataset"].get("raw_dir", "data/raw/floodnet")

    # SegFormer model (wrapped to behave like SMP model)
    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["backbone"],
        num_labels=config["model"]["num_labels"],
        ignore_mismatched_sizes=config["model"].get("ignore_mismatched_sizes", True),
    )

    # Wrap SegFormer to output raw logits like SMP models
    class SegFormerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            # SegFormer outputs at 1/4 resolution, upsample to input size
            logits = out.logits
            logits = torch.nn.functional.interpolate(
                logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            return logits

    model = SegFormerWrapper(model)

    # Datasets
    train_dataset = FloodNetDataset(
        images_dir=f"{raw_dir}/train/images",
        masks_dir=f"{raw_dir}/train/masks",
        transform=get_segmentation_transforms(config.get("augmentation"), "train", 512),
        image_size=512,
        debug_limit=args.debug_limit,
    )

    val_dataset = FloodNetDataset(
        images_dir=f"{raw_dir}/val/images",
        masks_dir=f"{raw_dir}/val/masks",
        transform=get_segmentation_transforms(config.get("augmentation"), "val", 512),
        image_size=512,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["dataset"].get("batch_size", 4),
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["dataset"].get("batch_size", 4),
                            shuffle=False, num_workers=2, pin_memory=True)

    loss_fn = WeightedCEDiceLoss(
        ce_weight=config.get("loss", {}).get("ce_weight", 0.5),
        dice_weight=config.get("loss", {}).get("dice_weight", 0.5),
        ignore_index=255,
    )

    output_dir = config.get("checkpoints_dir", "outputs") / "floodnet_segformer"

    best_metrics = train_segmentation(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, config=config, output_dir=str(output_dir),
        is_binary=False, num_classes=NUM_FLOODNET_CLASSES, ignore_index=255,
    )

    print(f"\nTraining complete. Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()

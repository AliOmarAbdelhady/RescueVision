#!/usr/bin/env python3
"""CLI script for training building-level damage classifier."""

import argparse
import logging
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rescuevision.data.datasets import XBDDamageCropDataset
from rescuevision.data.transforms import get_classification_transforms
from rescuevision.models.losses import FocalLoss, compute_class_weights
from rescuevision.training.train_damage_classifier import train_damage_classifier
from rescuevision.utils.config import load_config
from rescuevision.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train damage classifier")
    parser.add_argument("--config", type=str, default="configs/dinov2_damage_classifier.yaml")
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

    # Compute class weights
    class_weights = None
    if config.get("loss", {}).get("compute_class_weights", False):
        crops_df = pd.read_csv(config["dataset"]["crops_csv"])
        labels = crops_df[crops_df["label_id"] != 255]["label_id"].tolist()
        # Remap to 0-3
        label_map = {1: 0, 2: 1, 3: 2, 4: 3}
        labels = [label_map[l] for l in labels]
        class_weights = compute_class_weights(labels, num_classes=4)
        logging.info("Class weights: %s", class_weights)

    # Datasets
    train_dataset = XBDDamageCropDataset(
        crops_csv=config["dataset"]["crops_csv"],
        split_csv=config["dataset"]["split_csv"],
        split="train",
        transform=get_classification_transforms(
            config.get("augmentation"), phase="train",
            image_size=config["dataset"].get("image_size", 224),
        ),
        image_size=config["dataset"].get("image_size", 224),
        debug_limit=args.debug_limit,
    )

    val_dataset = XBDDamageCropDataset(
        crops_csv=config["dataset"]["crops_csv"],
        split_csv=config["dataset"]["split_csv"],
        split="val",
        transform=get_classification_transforms(
            config.get("augmentation"), phase="val",
            image_size=config["dataset"].get("image_size", 224),
        ),
        image_size=config["dataset"].get("image_size", 224),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["dataset"].get("batch_size", 32),
        shuffle=True, num_workers=config["dataset"].get("num_workers", 2), pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["dataset"].get("batch_size", 32),
        shuffle=False, num_workers=config["dataset"].get("num_workers", 2), pin_memory=True,
    )

    # Create model based on config
    model_name = config.get("model", {}).get("name", "dinov2_classifier")
    if model_name == "dinov2_classifier":
        from rescuevision.models.dinov2_classifier import DINOv2Classifier

        model = DINOv2Classifier(
            backbone=config["model"].get("backbone", "facebook/dinov2-base"),
            num_classes=config["model"].get("num_classes", 4),
            freeze_backbone=config["model"].get("freeze_backbone", True),
            mlp_hidden_dim=config["model"].get("mlp_hidden_dim", 512),
            dropout=config["model"].get("dropout", 0.3),
        )
    elif model_name == "efficientnet_classifier":
        from rescuevision.models.efficientnet_classifier import EfficientNetClassifier

        model = EfficientNetClassifier(
            backbone=config["model"].get("backbone", "efficientnet_b3"),
            pretrained=config["model"].get("pretrained", True),
            num_classes=config["model"].get("num_classes", 4),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Loss
    loss_type = config.get("loss", {}).get("type", "focal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loss_type == "focal":
        gamma = config.get("loss", {}).get("gamma", 2.0)
        alpha = class_weights.to(device) if class_weights is not None else None
        loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_type == "weighted_ce":
        weight = class_weights.to(device) if class_weights is not None else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    output_dir = config.get("checkpoints_dir", "outputs") / model_name

    best_metrics = train_damage_classifier(
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

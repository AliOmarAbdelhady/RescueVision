"""U-Net model wrapper using segmentation_models_pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def create_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str | None = None,
) -> nn.Module:
    """Create a U-Net model with specified encoder.

    Args:
        encoder_name: Encoder backbone name (e.g., 'resnet34', 'efficientnet-b3').
        encoder_weights: Pretrained weights ('imagenet' or None).
        in_channels: Number of input channels.
        classes: Number of output classes.
        activation: Output activation (None for logits).

    Returns:
        U-Net model.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if in_channels == 3 else None,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def create_fpn(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Create an FPN model."""
    return smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if in_channels == 3 else None,
        in_channels=in_channels,
        classes=classes,
    )


def create_deeplabv3plus(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    classes: int = 10,
) -> nn.Module:
    """Create a DeepLabV3+ model."""
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if in_channels == 3 else None,
        in_channels=in_channels,
        classes=classes,
    )

"""Albumentations-based augmentation pipelines for segmentation and classification."""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from rescuevision.constants import IMAGENET_MEAN, IMAGENET_STD


def get_segmentation_transforms(
    config: dict[str, Any] | None = None,
    phase: str = "train",
    image_size: int = 512,
) -> A.Compose:
    """Build segmentation augmentation pipeline.

    Args:
        config: Augmentation config dict with probability values.
        phase: 'train' or 'val'/'test'.
        image_size: Target image size.

    Returns:
        Albumentations Compose transform.
    """
    if config is None:
        config = {}

    normalize = config.get("normalize", True)

    if phase == "train":
        transforms = [
            A.Resize(image_size, image_size),
        ]

        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))
        if config.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=config["vertical_flip"]))
        if config.get("rotate90", 0) > 0:
            transforms.append(A.RandomRotate90(p=config["rotate90"]))
        if config.get("brightness_contrast", 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(p=config["brightness_contrast"])
            )

        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    else:
        transforms = [
            A.Resize(image_size, image_size),
        ]
        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_classification_transforms(
    config: dict[str, Any] | None = None,
    phase: str = "train",
    image_size: int = 224,
) -> A.Compose:
    """Build classification augmentation pipeline for crop images.

    Args:
        config: Augmentation config dict.
        phase: 'train' or 'val'/'test'.
        image_size: Target image size.

    Returns:
        Albumentations Compose transform.
    """
    if config is None:
        config = {}

    normalize = config.get("normalize", True)

    if phase == "train":
        transforms = [
            A.Resize(image_size, image_size),
        ]

        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))
        if config.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=config["vertical_flip"]))
        if config.get("rotate", 0) > 0:
            transforms.append(A.Rotate(limit=config["rotate"], p=0.5))
        if config.get("brightness_contrast", 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(p=config["brightness_contrast"])
            )

        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    else:
        transforms = [
            A.Resize(image_size, image_size),
        ]
        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_dual_segmentation_transforms(
    config: dict[str, Any] | None = None,
    phase: str = "train",
    image_size: int = 512,
) -> A.Compose:
    """Build augmentation for 6-channel segmentation (pre+post concatenated).

    Applies the same spatial transforms to both pre and post images.
    """
    if config is None:
        config = {}

    normalize = config.get("normalize", True)

    if phase == "train":
        transforms = [
            A.Resize(image_size, image_size),
        ]

        if config.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=config["horizontal_flip"]))
        if config.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=config["vertical_flip"]))
        if config.get("rotate90", 0) > 0:
            transforms.append(A.RandomRotate90(p=config["rotate90"]))
        if config.get("brightness_contrast", 0) > 0:
            transforms.append(
                A.RandomBrightnessContrast(p=config["brightness_contrast"])
            )

        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    else:
        transforms = [
            A.Resize(image_size, image_size),
        ]
        if normalize:
            transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transforms.append(ToTensorV2())

    return A.Compose(transforms)

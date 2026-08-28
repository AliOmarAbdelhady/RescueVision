"""Loss functions for segmentation and classification tasks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary or multiclass segmentation."""

    def __init__(self, mode: str = "binary", smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.mode = mode
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == "binary":
            probs = torch.sigmoid(logits)
            if probs.ndim == 4:
                probs = probs.squeeze(1)
            if targets.ndim == 4:
                targets = targets.squeeze(1)

            mask = targets != self.ignore_index
            probs = probs[mask]
            targets = targets[mask].float()

            intersection = (probs * targets).sum()
            return 1 - (2.0 * intersection + self.smooth) / (
                probs.sum() + targets.sum() + self.smooth
            )
        else:
            # Multiclass: softmax over classes
            num_classes = logits.shape[1]
            probs = F.softmax(logits, dim=1)

            # One-hot encode targets
            valid = targets != self.ignore_index
            targets_clamped = targets.clone()
            targets_clamped[~valid] = 0
            targets_one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
            valid_mask = valid.unsqueeze(1).expand_as(targets_one_hot).float()

            intersection = (probs * targets_one_hot * valid_mask).sum(dim=(2, 3))
            union = (probs * valid_mask).sum(dim=(2, 3)) + (targets_one_hot * valid_mask).sum(dim=(2, 3))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits.squeeze(1), targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class WeightedCEDiceLoss(nn.Module):
    """Weighted CE + Dice loss for multiclass segmentation."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal loss for classification with class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(
    targets: list[int] | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse frequency class weights.

    Args:
        targets: List or tensor of class labels.
        num_classes: Total number of classes.

    Returns:
        Normalized weight tensor of shape (num_classes,).
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    counts = [0] * num_classes
    for t in targets:
        if 0 <= t < num_classes:
            counts[t] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * c))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
    return weights_tensor

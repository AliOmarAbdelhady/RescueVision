"""Segmentation metrics: IoU, Dice, Precision, Recall, F1."""

from __future__ import annotations

import numpy as np
import torch


def compute_iou(pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> float:
    """Compute IoU for binary or single-class prediction."""
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, target: np.ndarray, ignore_index: int = 255) -> float:
    """Compute Dice coefficient."""
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / total)


def compute_precision_recall_f1(
    pred: np.ndarray, target: np.ndarray, ignore_index: int = 255
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1."""
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_multiclass_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float]:
    """Compute per-class and mean metrics for multiclass segmentation.

    Args:
        preds: Predicted class labels (H, W).
        targets: Ground truth class labels (H, W).
        num_classes: Number of valid classes.
        ignore_index: Label to ignore.

    Returns:
        Dict with per-class IoU, mean IoU, macro F1.
    """
    valid = targets != ignore_index
    preds_flat = preds[valid]
    targets_flat = targets[valid]

    class_ious = []
    class_f1s = []

    for cls in range(num_classes):
        pred_cls = preds_flat == cls
        target_cls = targets_flat == cls

        iou = compute_iou(
            pred_cls.astype(np.uint8), target_cls.astype(np.uint8)
        )
        _, _, f1 = compute_precision_recall_f1(
            pred_cls.astype(np.uint8), target_cls.astype(np.uint8)
        )

        class_ious.append(iou)
        class_f1s.append(f1)

    return {
        "miou": float(np.mean(class_ious)),
        "macro_f1": float(np.mean(class_f1s)),
        "class_ious": class_ious,
        "class_f1s": class_f1s,
    }


@torch.no_grad()
def compute_segmentation_metrics_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    is_binary: bool = False,
) -> dict[str, float]:
    """Compute segmentation metrics from a batch of predictions.

    Args:
        logits: Model output (B, C, H, W) or (B, 1, H, W).
        targets: Ground truth (B, H, W).
        num_classes: Number of output classes.
        ignore_index: Label to ignore.
        is_binary: Whether this is binary segmentation (1 class).

    Returns:
        Dict with computed metrics.
    """
    if is_binary:
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).cpu().numpy().astype(np.uint8)
    else:
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

    targets_np = targets.cpu().numpy().astype(np.uint8)

    # Flatten all batches
    preds_flat = preds.flatten()
    targets_flat = targets_np.flatten()

    if is_binary:
        iou = compute_iou(preds_flat, targets_flat)
        dice = compute_dice(preds_flat, targets_flat)
        precision, recall, f1 = compute_precision_recall_f1(preds_flat, targets_flat)
        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        return compute_multiclass_metrics(preds_flat, targets_flat, num_classes, ignore_index)

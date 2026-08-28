"""Confusion matrix computation and visualization for segmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute confusion matrix from flat arrays of predictions and targets.

    Args:
        preds: 1D array of predicted class labels.
        targets: 1D array of ground truth class labels.
        num_classes: Number of classes.
        ignore_index: Index to ignore.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = True,
) -> None:
    """Plot and optionally save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: Names for each class.
        title: Plot title.
        save_path: Where to save the figure.
        normalize: Whether to normalize by row (recall).
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
        fmt = ".2%"
        cm_display = cm_display
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

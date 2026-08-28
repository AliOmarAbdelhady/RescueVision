"""Visualization utilities for inference outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rescuevision.constants import DAMAGE_COLOR_MAP


def create_comparison_figure(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    building_mask: np.ndarray | None = None,
    damage_map: np.ndarray | None = None,
    change_mask: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Create side-by-side comparison figure."""
    panels = [("Pre-disaster", pre_image), ("Post-disaster", post_image)]

    if building_mask is not None:
        panels.append(("Buildings", building_mask))
    if damage_map is not None:
        colored = _colorize(damage_map)
        panels.append(("Damage Map", colored))
    if change_mask is not None:
        panels.append(("Changes", change_mask))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray" if "Building" in title else "hot")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Apply damage color map to mask."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in DAMAGE_COLOR_MAP.items():
        rgb[mask == cls_id] = color
    return rgb

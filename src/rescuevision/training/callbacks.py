"""Training callbacks: early stopping and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback based on validation metric."""

    def __init__(self, patience: int = 8, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Check if training should stop.

        Returns True if a new best score was found.
        """
        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered. Best score: %.4f", self.best_score
                )
            return False


class ModelCheckpoint:
    """Save best model checkpoint."""

    def __init__(self, output_dir: str | Path, metric_name: str = "iou", mode: str = "max"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        config: dict,
        is_best: bool = True,
    ) -> str:
        """Save checkpoint. Returns path to saved file."""
        improved = (
            metric > self.best_score if self.mode == "max" else metric < self.best_score
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            f"best_{self.metric_name}": metric,
            "config": config,
        }

        # Save last checkpoint
        last_path = self.output_dir / "last.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if improved:
            self.best_score = metric
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                "New best %s: %.4f at epoch %d", self.metric_name, metric, epoch
            )
            return str(best_path)

        return str(last_path)

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> dict:
        """Load a checkpoint."""
        return torch.load(path, map_location=device, weights_only=False)

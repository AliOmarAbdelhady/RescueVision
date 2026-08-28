"""Training loop for building-level damage classifier."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rescuevision.evaluation.metrics_classification import (
    compute_classification_metrics,
    plot_classification_confusion_matrix,
)
from rescuevision.training.callbacks import EarlyStopping, ModelCheckpoint
from rescuevision.training.logging_utils import CSVLogger
from rescuevision.training.optim import create_optimizer, create_scheduler
from rescuevision.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train_damage_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: dict,
    output_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Train building-level damage classifier.

    Supports two-stage training for DINOv2:
    - Stage 1: Frozen backbone, train MLP head
    - Stage 2: Unfreeze last transformer blocks
    """
    set_seed(config.get("seed", 42))

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    epochs = config.get("training", {}).get("epochs", 25)
    lr = config.get("training", {}).get("lr", 1e-3)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-4)
    scheduler_name = config.get("training", {}).get("scheduler", "cosine")
    mixed_precision = config.get("training", {}).get("mixed_precision", True)
    patience = config.get("training", {}).get("early_stopping_patience", 7)
    best_metric_name = config.get("training", {}).get("save_best_metric", "macro_f1")

    # Check if two-stage training (DINOv2)
    unfreeze_after = config.get("model", {}).get("unfreeze_after_epoch", None)
    unfreeze_blocks = config.get("model", {}).get("unfreeze_blocks", 4)
    stage2_lr = config.get("training", {}).get("stage2_lr", 1e-5)

    optimizer = create_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, name=scheduler_name, epochs=epochs)

    early_stopper = EarlyStopping(patience=patience, mode="max")
    checkpoint = ModelCheckpoint(ckpt_dir, metric_name=best_metric_name, mode="max")
    csv_logger = CSVLogger(log_dir)

    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    if class_names is None:
        class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]

    logger.info("Starting classifier training: %d epochs, device=%s", epochs, device)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Stage 2: unfreeze backbone blocks
        if unfreeze_after and epoch == unfreeze_after + 1:
            if hasattr(model, "unfreeze_last_blocks"):
                model.unfreeze_last_blocks(num_blocks=unfreeze_blocks)
                logger.info("Unfroze last %d backbone blocks", unfreeze_blocks)

                # Reset optimizer with lower LR for fine-tuning
                optimizer = create_optimizer(
                    model.parameters(), lr=stage2_lr, weight_decay=weight_decay,
                )
                scheduler = create_scheduler(optimizer, name=scheduler_name, epochs=epochs - epoch + 1)
                logger.info("Reset optimizer with lr=%.1e for stage 2", stage2_lr)

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            pre_crops = batch["pre_crop"].to(device, non_blocking=True)
            post_crops = batch["post_crop"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=mixed_precision):
                logits = model(pre_crops, post_crops)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item() * pre_crops.shape[0]
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.shape[0]

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        val_metrics = _evaluate_classifier(
            model, val_loader, device, loss_fn, mixed_precision, class_names
        )
        val_metrics["train_loss"] = avg_train_loss
        val_metrics["train_accuracy"] = train_acc

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get(best_metric_name, 0))
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start
        log_row = {
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            **{k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }
        csv_logger.log(log_row, epoch)

        metric_val = val_metrics.get(best_metric_name, 0)
        is_best = early_stopper.step(metric_val)
        checkpoint.save(model, optimizer, epoch, metric_val, config, is_best=is_best)

        logger.info(
            "Epoch %d/%d | loss=%.4f | acc=%.4f | %s=%.4f | time=%.1fs",
            epoch, epochs, avg_train_loss, val_metrics.get("accuracy", 0),
            best_metric_name, metric_val, epoch_time,
        )

        if early_stopper.should_stop:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Final confusion matrix
    _save_final_confusion(model, val_loader, device, class_names, output_dir, mixed_precision)

    logger.info("Training complete. Best %s: %.4f", best_metric_name, early_stopper.best_score)
    return {best_metric_name: early_stopper.best_score}


@torch.no_grad()
def _evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    mixed_precision: bool = True,
    class_names: list[str] | None = None,
) -> dict:
    """Evaluate damage classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        pre = batch["pre_crop"].to(device, non_blocking=True)
        post = batch["post_crop"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=mixed_precision):
            logits = model(pre, post)
            if loss_fn:
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * pre.shape[0]

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_samples += pre.shape[0]

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    if total_samples > 0 and loss_fn:
        metrics["loss"] = total_loss / total_samples

    return metrics


def _save_final_confusion(
    model, dataloader, device, class_names, output_dir, mixed_precision,
):
    """Save final confusion matrix plot."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            pre = batch["pre_crop"].to(device)
            post = batch["post_crop"].to(device)
            with torch.amp.autocast("cuda", enabled=mixed_precision):
                logits = model(pre, post)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    plot_classification_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

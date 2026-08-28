"""Training loop for change detection models."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rescuevision.evaluation.metrics_segmentation import compute_segmentation_metrics_batch
from rescuevision.training.callbacks import EarlyStopping, ModelCheckpoint
from rescuevision.training.logging_utils import CSVLogger
from rescuevision.training.optim import create_optimizer, create_scheduler
from rescuevision.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train_change_detection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    config: dict,
    output_dir: str | Path,
) -> dict[str, float]:
    """Train a change detection model.

    The model expects (pre_image, post_image) and produces binary change logits.
    """
    set_seed(config.get("seed", 42))

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    epochs = config.get("training", {}).get("epochs", 50)
    lr = config.get("training", {}).get("lr", 1e-4)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-5)
    scheduler_name = config.get("training", {}).get("scheduler", "cosine")
    mixed_precision = config.get("training", {}).get("mixed_precision", True)
    patience = config.get("training", {}).get("early_stopping_patience", 8)
    best_metric_name = config.get("training", {}).get("save_best_metric", "iou")
    grad_accum = config.get("training", {}).get("gradient_accumulation_steps", 1)

    optimizer = create_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, name=scheduler_name, epochs=epochs)

    early_stopper = EarlyStopping(patience=patience, mode="max")
    checkpoint = ModelCheckpoint(ckpt_dir, metric_name=best_metric_name, mode="max")
    csv_logger = CSVLogger(log_dir)

    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    logger.info("Starting change detection training: %d epochs, device=%s", epochs, device)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            pre = batch["pre_image"].to(device, non_blocking=True)
            post = batch["post_image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).float()

            with torch.amp.autocast("cuda", enabled=mixed_precision):
                logits = model(pre, post)
                if logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                loss = loss_fn(logits, masks) / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum * pre.shape[0]
            train_samples += pre.shape[0]

        avg_train_loss = train_loss / max(train_samples, 1)

        # Validation
        val_metrics = _evaluate_change(model, val_loader, device, loss_fn, mixed_precision)
        val_metrics["train_loss"] = avg_train_loss

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get(best_metric_name, 0))
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start
        log_row = {
            "train_loss": avg_train_loss,
            "val_loss": val_metrics.get("loss", 0),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            **{k: v for k, v in val_metrics.items() if k not in ("class_ious", "class_f1s")},
        }
        csv_logger.log(log_row, epoch)

        metric_val = val_metrics.get(best_metric_name, 0)
        is_best = early_stopper.step(metric_val)
        checkpoint.save(model, optimizer, epoch, metric_val, config, is_best=is_best)

        logger.info(
            "Epoch %d/%d | loss=%.4f | %s=%.4f | time=%.1fs",
            epoch, epochs, avg_train_loss, best_metric_name, metric_val, epoch_time,
        )

        if early_stopper.should_stop:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Training complete. Best %s: %.4f", best_metric_name, early_stopper.best_score)
    return {best_metric_name: early_stopper.best_score}


@torch.no_grad()
def _evaluate_change(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    mixed_precision: bool = True,
) -> dict[str, float]:
    """Evaluate change detection model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_metrics = {}

    for batch in dataloader:
        pre = batch["pre_image"].to(device, non_blocking=True)
        post = batch["post_image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=mixed_precision):
            logits = model(pre, post)

            if loss_fn:
                if logits.shape[1] == 1:
                    loss = loss_fn(logits.squeeze(1), masks.float())
                else:
                    loss = loss_fn(logits, masks)
                total_loss += loss.item() * pre.shape[0]

        metrics = compute_segmentation_metrics_batch(
            logits, masks, num_classes=1, is_binary=True,
        )
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                all_metrics[k] = all_metrics.get(k, 0) + v * pre.shape[0]

        total_samples += pre.shape[0]

    result = {k: v / max(total_samples, 1) for k, v in all_metrics.items()}
    if total_samples > 0 and loss_fn:
        result["loss"] = total_loss / total_samples
    return result

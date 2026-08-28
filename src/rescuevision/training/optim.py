"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch.optim as optim


def create_optimizer(
    params,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    name: str = "adamw",
) -> optim.Optimizer:
    """Create an optimizer."""
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    epochs: int = 50,
    min_lr: float = 1e-6,
) -> optim.lr_scheduler.LRScheduler | None:
    """Create a learning rate scheduler."""
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")

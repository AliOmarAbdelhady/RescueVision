"""First-conv weight adaptation for 6-channel input on pretrained encoders."""

from __future__ import annotations

import torch.nn as nn


def adapt_first_conv(model: nn.Module, old_channels: int = 3, new_channels: int = 6) -> nn.Module:
    """Adapt encoder first conv from old_channels to new_channels.

    Strategy: replicate ImageNet weights and normalize so output statistics
    remain similar to the 3-channel case.

    Args:
        model: Full segmentation model with an encoder attribute.
        old_channels: Original number of input channels (usually 3).
        new_channels: Target number of input channels (e.g., 6 for pre+post concat).

    Returns:
        Modified model.
    """
    encoder = model.encoder

    # Find the first Conv2d with old_channels input
    first_conv = None
    first_conv_name = None

    for name, module in encoder.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == old_channels:
            first_conv = module
            first_conv_name = name
            break

    if first_conv is None:
        raise ValueError(
            f"Could not find Conv2d with in_channels={old_channels} in encoder"
        )

    old_weight = first_conv.weight.data  # (out_ch, old_ch, kH, kW)
    repeat_factor = new_channels // old_channels

    # Replicate weights: each group of old_channels gets the same weights
    new_weight = old_weight.repeat(1, repeat_factor, 1, 1)
    # Normalize to preserve output magnitude
    new_weight = new_weight / repeat_factor

    # Create new conv layer
    new_conv = nn.Conv2d(
        new_channels,
        first_conv.out_channels,
        first_conv.kernel_size,
        first_conv.stride,
        first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    # Replace in encoder
    _set_module_by_name(encoder, first_conv_name, new_conv)

    return model


def _set_module_by_name(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a submodule by its dotted name."""
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

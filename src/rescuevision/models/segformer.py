"""SegFormer wrapper for semantic segmentation using Hugging Face Transformers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SegFormerWrapper(nn.Module):
    """SegFormer model wrapper for building/damage segmentation.

    Wraps Hugging Face SegformerForSemanticSegmentation to handle
    input/output size mismatches and provide a consistent interface.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels: int = 5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_labels = num_labels

        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig

            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_labels

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )
            logger.info("Loaded SegFormer: %s (%d labels)", model_name, num_labels)

        except ImportError:
            raise ImportError(
                "transformers library required for SegFormer. "
                "Install with: pip install transformers"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W). C=3 for RGB, C=6 for pre+post.

        Returns:
            Logits tensor (B, num_labels, H, W) at full input resolution.
        """
        input_h, input_w = x.shape[2:]

        # Handle 6-channel input by averaging first conv or using first 3 channels
        if x.shape[1] == 6:
            x = x[:, :3, :, :]  # Use post-disaster channels only

        output = self.model(x)
        logits = output.logits

        # Resize to match input dimensions
        if logits.shape[2] != input_h or logits.shape[3] != input_w:
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode="bilinear", align_corners=False,
            )

        return logits


class SegFormer6Channel(nn.Module):
    """SegFormer variant that accepts 6-channel pre+post input.

    Uses two separate SegFormer encoders with shared weights and
    a learned fusion layer.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels: int = 5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_labels = num_labels

        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_labels

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Get the feature dimension from the model
        hidden_size = self.model.config.hidden_sizes[-1]

        # Fusion layer: combine pre and post features
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, 1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        # Replace segmentation head to handle fused features
        self.decode_head = self.model.decode_head

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Forward pass with separate pre and post inputs.

        Args:
            pre: Pre-disaster image (B, 3, H, W).
            post: Post-disaster image (B, 3, H, W).

        Returns:
            Logits tensor (B, num_labels, H, W).
        """
        input_h, input_w = pre.shape[2:]

        # Extract features from both images
        pre_features = self.model(pre, output_hidden_states=True)
        post_features = self.model(post, output_hidden_states=True)

        # Get last hidden states
        pre_hidden = pre_features.hidden_states[-1]
        post_hidden = post_features.hidden_states[-1]

        # Ensure same spatial size
        if pre_hidden.shape[2:] != post_hidden.shape[2:]:
            post_hidden = F.interpolate(
                post_hidden, size=pre_hidden.shape[2:],
                mode="bilinear", align_corners=False,
            )

        # Fuse features
        fused = self.fusion(torch.cat([pre_hidden, post_hidden], dim=1))

        # Decode through the head
        logits = self.decode_head(fused)

        # Resize to input resolution
        if logits.shape[2] != input_h or logits.shape[3] != input_w:
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode="bilinear", align_corners=False,
            )

        return logits


def create_segformer(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels: int = 5,
    in_channels: int = 3,
    ignore_index: int = 255,
) -> nn.Module:
    """Factory function to create SegFormer model.

    Args:
        model_name: Hugging Face model identifier.
        num_labels: Number of output classes.
        in_channels: Input channels (3 or 6).
        ignore_index: Ignore index for loss computation.

    Returns:
        SegFormer model.
    """
    if in_channels == 6:
        return SegFormer6Channel(
            model_name=model_name,
            num_labels=num_labels,
            ignore_index=ignore_index,
        )
    return SegFormerWrapper(
        model_name=model_name,
        num_labels=num_labels,
        ignore_index=ignore_index,
    )

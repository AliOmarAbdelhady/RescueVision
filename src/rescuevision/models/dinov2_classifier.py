"""DINOv2-based building-level damage classifier.

Uses frozen DINOv2 backbone to extract CLS token embeddings,
then fuses pre/post embeddings and classifies with MLP head.
Supports two-stage training: frozen backbone then partial unfreezing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Classifier(nn.Module):
    """DINOv2 + MLP classifier for building damage assessment.

    Architecture:
        pre_crop -> DINOv2 -> CLS embedding
        post_crop -> DINOv2 -> CLS embedding
        features = concat(emb_pre, emb_post, abs(emb_post - emb_pre))
        features -> MLP -> 4-class prediction
    """

    def __init__(
        self,
        backbone: str = "facebook/dinov2-base",
        num_classes: int = 4,
        freeze_backbone: bool = True,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone

        # Load DINOv2
        self.backbone = AutoModel.from_pretrained(backbone)
        self.embed_dim = self.backbone.config.hidden_size  # 768 for base

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 3, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden_dim // 2, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embedding from image."""
        outputs = self.backbone(x)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def forward(self, pre_crop: torch.Tensor, post_crop: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pre_crop: (B, 3, H, W) pre-disaster building crop.
            post_crop: (B, 3, H, W) post-disaster building crop.

        Returns:
            (B, num_classes) logits.
        """
        emb_pre = self.extract_features(pre_crop)
        emb_post = self.extract_features(post_crop)

        features = torch.cat([emb_pre, emb_post, torch.abs(emb_post - emb_pre)], dim=1)

        return self.classifier(features)

    def unfreeze_last_blocks(self, num_blocks: int = 4) -> None:
        """Unfreeze the last N transformer blocks for fine-tuning.

        Args:
            num_blocks: Number of blocks to unfreeze from the end.
        """
        # DINOv2 backbone has encoder.layer modules
        layers = list(self.backbone.encoder.layer)
        total = len(layers)

        for i, layer in enumerate(layers):
            if i >= total - num_blocks:
                for param in layer.parameters():
                    param.requires_grad = True

        # Also unfreeze layernorm
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

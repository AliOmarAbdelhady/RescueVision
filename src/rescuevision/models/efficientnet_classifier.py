"""EfficientNet-based building-level damage classifier.

Two-stream architecture: pre and post crops are encoded separately,
embeddings are fused, and an MLP head predicts damage class.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class EfficientNetClassifier(nn.Module):
    """Two-stream EfficientNet classifier for building damage.

    Architecture:
        pre_crop -> EfficientNet -> emb_pre
        post_crop -> EfficientNet -> emb_post
        features = concat(emb_pre, emb_post, abs(emb_post - emb_pre))
        features -> FC -> 4-class prediction
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        num_classes: int = 4,
    ):
        super().__init__()
        self.backbone_name = backbone

        # Create backbone
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        embed_dim = self.encoder.num_features

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre_crop: torch.Tensor, post_crop: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pre_crop: (B, 3, H, W) pre-disaster building crop.
            post_crop: (B, 3, H, W) post-disaster building crop.

        Returns:
            (B, num_classes) logits.
        """
        emb_pre = self.encoder(pre_crop)
        emb_post = self.encoder(post_crop)

        # Late fusion: concat + absolute difference
        features = torch.cat([emb_pre, emb_post, torch.abs(emb_post - emb_pre)], dim=1)

        return self.classifier(features)

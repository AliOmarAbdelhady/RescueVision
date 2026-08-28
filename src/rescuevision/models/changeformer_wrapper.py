"""ChangeFormer wrapper for change detection.

Implements a transformer-based change detection model that compares
pre and post disaster images to detect changed regions.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ChangeFormerEncoder(nn.Module):
    """Feature extraction backbone using a segmentation encoder.

    Extracts multi-scale features from input images using a shared
    segmentation model backbone (e.g., ResNet).
    """

    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.encoder = base.encoder

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)


class TransformerFusionBlock(nn.Module):
    """Transformer-based fusion block for change detection features."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class ChangeFormer(nn.Module):
    """ChangeFormer: transformer-based change detection model.

    Architecture:
    1. Shared encoder extracts multi-scale features from pre and post images
    2. Feature difference is computed at each scale
    3. Transformer blocks process the differences
    4. Decoder upsamples and combines to produce change mask
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        classes: int = 1,
        transformer_heads: int = 4,
    ):
        super().__init__()
        self.classes = classes

        # Shared encoder
        self.encoder = ChangeFormerEncoder(encoder_name, pretrained=pretrained)

        # Determine encoder output channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            features = self.encoder(dummy)
            encoder_channels = [f.shape[1] for f in features]

        # Transformer fusion blocks for each scale
        self.transformer_blocks = nn.ModuleList()
        for ch in encoder_channels:
            self.transformer_blocks.append(
                TransformerFusionBlock(dim=ch, num_heads=transformer_heads)
            )

        # Difference projection layers
        self.diff_projections = nn.ModuleList()
        for ch in encoder_channels:
            self.diff_projections.append(nn.Sequential(
                nn.Conv2d(ch * 3, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))

        # Decoder
        decoder_channels = [256, 128, 64, 32, 16]
        in_ch = encoder_channels[-1]
        self.decoder_blocks = nn.ModuleList()
        for out_ch in decoder_channels:
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # Skip convolutions
        self.skip_convs = nn.ModuleList()
        for ch in encoder_channels[:-1]:
            self.skip_convs.append(nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ))

        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], classes, 1)

    def forward(
        self, pre_image: torch.Tensor, post_image: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pre_image: (B, 3, H, W) pre-disaster image.
            post_image: (B, 3, H, W) post-disaster image.

        Returns:
            (B, classes, H, W) change logits.
        """
        # Extract features
        features_pre = self.encoder(pre_image)
        features_post = self.encoder(post_image)

        # Compute and process differences at each scale
        processed_features = []
        for i, (f_pre, f_post) in enumerate(zip(features_pre, features_post)):
            B, C, H, W = f_pre.shape

            # Feature difference: concat(pre, post, abs(pre-post))
            diff = torch.cat([f_pre, f_post, torch.abs(f_pre - f_post)], dim=1)
            diff_proj = self.diff_projections[i](diff)

            # Apply transformer
            flat = diff_proj.flatten(2).transpose(1, 2)  # (B, H*W, C)
            flat = self.transformer_blocks[i](flat)
            flat = flat.transpose(1, 2).view(B, C, H, W)

            processed_features.append(flat)

        # Decoder with skip connections
        x = processed_features[-1]
        for i, (dec_block, skip_conv) in enumerate(
            zip(self.decoder_blocks, self.skip_convs)
        ):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skip_conv(processed_features[-(i + 2)])

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = x + skip
            x = dec_block(x)

        # Handle remaining decoder blocks
        for dec_block in self.decoder_blocks[len(self.skip_convs):]:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = dec_block(x)

        # Final resize and head
        x = F.interpolate(x, size=pre_image.shape[2:], mode="bilinear", align_corners=False)
        return self.seg_head(x)


def create_changeformer(
    encoder_name: str = "resnet34",
    pretrained: bool = True,
    classes: int = 1,
) -> ChangeFormer:
    """Factory function to create ChangeFormer model."""
    return ChangeFormer(
        encoder_name=encoder_name,
        pretrained=pretrained,
        classes=classes,
    )

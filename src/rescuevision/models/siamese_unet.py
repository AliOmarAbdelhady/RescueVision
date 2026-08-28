"""Siamese U-Net for change detection.

Architecture: shared encoder processes pre/post images, features are fused,
then a decoder produces binary change mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    """Siamese U-Net with shared encoder and feature fusion decoder.

    The encoder processes pre and post images with shared weights.
    Features are fused using concatenation and absolute difference,
    then decoded to produce a binary change mask.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        fusion_mode: str = "concat_absdiff",
        classes: int = 1,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.classes = classes

        # Build encoder using SMP
        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
        )
        self.encoder = base_unet.encoder

        # Get encoder channel sizes
        encoder_channels = self._get_encoder_channels()
        bottleneck_ch = encoder_channels[-1]

        # Determine fusion channel multiplier
        if fusion_mode == "concat":
            fusion_mult = 2
        elif fusion_mode == "absdiff":
            fusion_mult = 1
        elif fusion_mode == "concat_absdiff":
            fusion_mult = 3
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        # Build custom decoder head
        # Decoder processes fused features at each scale
        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = nn.ModuleList()

        # Bottleneck fusion -> first decoder block
        in_ch = bottleneck_ch * fusion_mult
        for dec_ch in decoder_channels:
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = dec_ch

        # Skip connection convolutions
        self.skip_convs = nn.ModuleList()
        for enc_ch in encoder_channels[:-1]:
            self.skip_convs.append(nn.Sequential(
                nn.Conv2d(enc_ch * fusion_mult, enc_ch, 1, bias=False),
                nn.BatchNorm2d(enc_ch),
                nn.ReLU(inplace=True),
            ))

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, 1)

    def _get_encoder_channels(self) -> list[int]:
        """Extract output channel sizes from encoder stages."""
        channels = []
        dummy = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            features = self.encoder(dummy)
        for f in features:
            channels.append(f.shape[1])
        return channels

    def _fuse(self, feat_pre: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        """Fuse pre and post features."""
        if self.fusion_mode == "concat":
            return torch.cat([feat_pre, feat_post], dim=1)
        elif self.fusion_mode == "absdiff":
            return torch.abs(feat_post - feat_pre)
        elif self.fusion_mode == "concat_absdiff":
            return torch.cat([feat_pre, feat_post, torch.abs(feat_post - feat_pre)], dim=1)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_mode}")

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pre_image: (B, 3, H, W) pre-disaster image.
            post_image: (B, 3, H, W) post-disaster image.

        Returns:
            (B, 1, H, W) change logits.
        """
        # Encode both images with shared encoder
        features_pre = self.encoder(pre_image)
        features_post = self.encoder(post_image)

        # Fuse bottleneck features
        fused_bottleneck = self._fuse(features_pre[-1], features_post[-1])

        # Decoder with skip connections
        x = fused_bottleneck
        for i, (dec_block, skip_conv) in enumerate(zip(self.decoder, self.skip_convs)):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Fuse and process skip connection
            skip_fused = self._fuse(
                features_pre[-(i + 2)], features_post[-(i + 2)]
            )
            skip = skip_conv(skip_fused)

            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = x + skip
            x = dec_block(x)

        # Final upsampling to original size
        x = F.interpolate(x, size=pre_image.shape[2:], mode="bilinear", align_corners=False)
        return self.segmentation_head(x)

"""End-to-end prediction pipeline for disaster damage assessment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rescuevision.constants import DAMAGE_COLOR_MAP, IGNORE_INDEX
from rescuevision.utils.image import load_image, save_image, save_npy
from rescuevision.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class DisasterPredictor:
    """Orchestrates all models for end-to-end disaster prediction.

    Pipeline:
    1. Load pre/post images
    2. Run building segmentation
    3. Run change detection
    4. Extract individual buildings (connected components)
    5. Run damage classifier on each building crop
    6. Build per-building damage map
    7. Compute summary statistics
    8. Generate visualizations
    9. Export JSON summary
    """

    def __init__(self, config: dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = config.get("image_size", 1024)

        self.building_model = None
        self.damage_seg_model = None
        self.change_model = None
        self.damage_cls_model = None

        self._load_models()

    def _load_models(self) -> None:
        """Load all trained models from checkpoints."""
        models_cfg = self.config.get("models", {})

        for name, cfg in models_cfg.items():
            checkpoint_path = cfg.get("checkpoint")
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.warning("Checkpoint not found for %s: %s", name, checkpoint_path)
                continue

            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model_cfg = checkpoint.get("config", {})

            if name == "building_seg":
                from rescuevision.models.unet import create_unet
                model = create_unet(
                    encoder_name=model_cfg.get("model", {}).get("encoder", "resnet34"),
                    in_channels=3,
                    classes=1,
                )
            elif name == "damage_seg":
                from rescuevision.models.unet import create_unet
                from rescuevision.utils.weight_init import adapt_first_conv
                model = create_unet(
                    encoder_name=model_cfg.get("model", {}).get("encoder", "efficientnet-b3"),
                    in_channels=6,
                    classes=5,
                )
                model = adapt_first_conv(model, 3, 6)
            elif name == "change_detection":
                from rescuevision.models.siamese_unet import SiameseUNet
                model = SiameseUNet(
                    encoder_name=model_cfg.get("model", {}).get("encoder", "resnet34"),
                    fusion_mode=model_cfg.get("model", {}).get("fusion_mode", "concat_absdiff"),
                )
            elif name == "damage_classifier":
                model_name = model_cfg.get("model", {}).get("name", "dinov2_classifier")
                if model_name == "dinov2_classifier":
                    from rescuevision.models.dinov2_classifier import DINOv2Classifier
                    model = DINOv2Classifier(
                        backbone=model_cfg.get("model", {}).get("backbone", "facebook/dinov2-base"),
                        num_classes=4,
                    )
                else:
                    from rescuevision.models.efficientnet_classifier import EfficientNetClassifier
                    model = EfficientNetClassifier(
                        backbone=model_cfg.get("model", {}).get("backbone", "efficientnet_b3"),
                        num_classes=4,
                    )
            else:
                logger.warning("Unknown model: %s", name)
                continue

            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device).eval()

            if name == "building_seg":
                self.building_model = model
            elif name == "damage_seg":
                self.damage_seg_model = model
            elif name == "change_detection":
                self.change_model = model
            elif name == "damage_classifier":
                self.damage_cls_model = model

            logger.info("Loaded %s from %s", name, checkpoint_path)

    def predict(
        self,
        pre_image_path: str | Path,
        post_image_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """Run full prediction pipeline.

        Args:
            pre_image_path: Path to pre-disaster image.
            post_image_path: Path to post-disaster image.
            output_dir: Directory to save outputs.

        Returns:
            Dict with summary, artifacts, and metadata.
        """
        output_dir = ensure_dir(output_dir)

        # Load images
        pre_image = load_image(pre_image_path)
        post_image = load_image(post_image_path)
        h, w = pre_image.shape[:2]

        logger.info("Predicting on images of size %dx%d", w, h)

        # Prepare tensors
        pre_tensor = self._to_tensor(pre_image)
        post_tensor = self._to_tensor(post_image)

        results = {}

        # 1. Building segmentation
        if self.building_model:
            building_mask = self._predict_segmentation(self.building_model, pre_tensor)
            results["building_mask"] = building_mask
            save_npy(building_mask, output_dir / "building_mask.npy")
            save_image(building_mask * 255, output_dir / "building_mask.png")

        # 2. Change detection
        if self.change_model:
            change_mask = self._predict_change(post_tensor, pre_tensor)
            results["change_mask"] = change_mask
            save_npy(change_mask, output_dir / "change_mask.npy")
            save_image(change_mask * 255, output_dir / "change_mask.png")

        # 3. Damage segmentation (6-channel)
        if self.damage_seg_model:
            damage_mask = self._predict_damage_segmentation(pre_tensor, post_tensor)
            results["damage_mask"] = damage_mask
            save_npy(damage_mask, output_dir / "damage_mask.npy")
            damage_rgb = self._colorize_damage(damage_mask)
            save_image(damage_rgb, output_dir / "damage_map.png")

        # 4. Building-level classification
        if self.damage_cls_model and self.building_model is not None:
            building_damage_map = self._predict_building_damage(
                pre_image, post_image, building_mask
            )
            results["building_damage_map"] = building_damage_map
            damage_rgb = self._colorize_damage(building_damage_map)
            save_image(damage_rgb, output_dir / "overlay_damage.png")

        # 5. Summary statistics
        summary = self._compute_summary(results.get("building_damage_map", results.get("damage_mask")))
        results["summary"] = summary

        # Save summary JSON
        with open(output_dir / "emergency_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # 6. Visualization
        self._create_visualizations(pre_image, post_image, results, output_dir)

        logger.info("Prediction complete. Outputs saved to %s", output_dir)
        return results

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor."""
        from rescuevision.constants import IMAGENET_MEAN, IMAGENET_STD

        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Resize to model input size
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return tensor.to(self.device)

    @torch.no_grad()
    def _predict_segmentation(self, model: nn.Module, image: torch.Tensor) -> np.ndarray:
        """Run binary segmentation."""
        logits = model(image)
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        else:
            pred = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return pred

    @torch.no_grad()
    def _predict_change(self, model: nn.Module, pre: torch.Tensor, post: torch.Tensor) -> np.ndarray:
        """Run change detection."""
        logits = model(pre, post)
        if logits.shape[1] == 1:
            pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        else:
            pred = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return pred

    @torch.no_grad()
    def _predict_damage_segmentation(self, pre: torch.Tensor, post: torch.Tensor) -> np.ndarray:
        """Run multiclass damage segmentation."""
        combined = torch.cat([pre, post], dim=1)
        logits = self.damage_seg_model(combined)
        pred = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        return pred

    @torch.no_grad()
    def _predict_building_damage(
        self, pre_image: np.ndarray, post_image: np.ndarray, building_mask: np.ndarray
    ) -> np.ndarray:
        """Classify each building in the mask."""
        from skimage.measure import label, regionprops

        labeled = label(building_mask)
        regions = regionprops(labeled)

        damage_map = np.zeros_like(building_mask, dtype=np.uint8)

        label_map = {0: 1, 1: 2, 2: 3, 3: 4}  # model output -> damage class ID

        for region in regions:
            y1, x1, y2, x2 = region.bbox
            margin = 8
            y1 = max(0, y1 - margin)
            x1 = max(0, x1 - margin)
            y2 = min(pre_image.shape[0], y2 + margin)
            x2 = min(pre_image.shape[1], x2 + margin)

            pre_crop = self._crop_to_tensor(pre_image[y1:y2, x1:x2])
            post_crop = self._crop_to_tensor(post_image[y1:y2, x1:x2])

            logits = self.damage_cls_model(pre_crop, post_crop)
            pred_class = logits.argmax(1).item()
            damage_id = label_map[pred_class]

            # Assign to building pixels in this region
            mask_region = labeled[y1:y2, x1:x2] == region.label
            damage_map[y1:y2, x1:x2][mask_region] = damage_id

        return damage_map

    def _crop_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """Convert a crop to a normalized tensor for classification."""
        from rescuevision.constants import IMAGENET_MEAN, IMAGENET_STD

        tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Resize to 224x224 for classifier
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(224, 224),
            mode="bilinear", align_corners=False,
        )
        return tensor.to(self.device)

    def _colorize_damage(self, mask: np.ndarray) -> np.ndarray:
        """Convert damage mask to RGB color map."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in DAMAGE_COLOR_MAP.items():
            rgb[mask == cls_id] = color
        return rgb

    def _compute_summary(self, damage_map: np.ndarray | None) -> dict:
        """Compute summary statistics from damage map."""
        if damage_map is None:
            return {"error": "No damage map available"}

        total = 0
        counts = {1: 0, 2: 0, 3: 0, 4: 0}

        # Use connected components for building-level counts
        if self.building_model is not None:
            from skimage.measure import label, regionprops
            building_mask = (damage_map > 0).astype(np.uint8)
            labeled = label(building_mask)
            for region in regionprops(labeled):
                region_mask = labeled == region.label
                region_labels = damage_map[region_mask]
                majority_class = np.bincount(region_labels[region_labels > 0]).argmax()
                if majority_class in counts:
                    counts[majority_class] += 1
                total += 1
        else:
            # Pixel-level counts as fallback
            for cls_id in [1, 2, 3, 4]:
                counts[cls_id] = int((damage_map == cls_id).sum())
            total = sum(counts.values())

        destroyed_ratio = counts[4] / max(total, 1)
        major_ratio = counts[3] / max(total, 1)
        score = 0.6 * destroyed_ratio + 0.3 * major_ratio

        if score >= 0.45:
            urgency = "CRITICAL"
        elif score >= 0.25:
            urgency = "HIGH"
        elif score >= 0.10:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        return {
            "total_buildings": total,
            "no_damage": counts[1],
            "minor_damage": counts[2],
            "major_damage": counts[3],
            "destroyed": counts[4],
            "urgency_score": round(score, 3),
            "urgency_level": urgency,
        }

    def _create_visualizations(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        results: dict,
        output_dir: Path,
    ) -> None:
        """Create and save visualization overlays."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_panels = 2  # pre + post
        if "building_mask" in results:
            n_panels += 1
        if "damage_mask" in results or "building_damage_map" in results:
            n_panels += 1
        if "change_mask" in results:
            n_panels += 1

        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        idx = 0
        axes[idx].imshow(pre_image)
        axes[idx].set_title("Pre-disaster")
        axes[idx].axis("off")
        idx += 1

        axes[idx].imshow(post_image)
        axes[idx].set_title("Post-disaster")
        axes[idx].axis("off")
        idx += 1

        if "building_mask" in results:
            axes[idx].imshow(results["building_mask"], cmap="gray")
            axes[idx].set_title("Building Mask")
            axes[idx].axis("off")
            idx += 1

        damage = results.get("building_damage_map", results.get("damage_mask"))
        if damage is not None:
            axes[idx].imshow(self._colorize_damage(damage))
            axes[idx].set_title("Damage Map")
            axes[idx].axis("off")
            idx += 1

        if "change_mask" in results:
            axes[idx].imshow(results["change_mask"], cmap="hot")
            axes[idx].set_title("Change Detection")
            axes[idx].axis("off")

        plt.tight_layout()
        fig.savefig(output_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

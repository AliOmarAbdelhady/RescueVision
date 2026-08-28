# Model Cards

## Building Segmentation

### U-Net Baseline (ResNet34)
- **Task:** Binary building localization
- **Input:** 3-channel RGB image (pre or post disaster)
- **Output:** Binary mask (1=building, 0=background)
- **Encoder:** ResNet34 pretrained on ImageNet
- **Loss:** BCEWithLogitsLoss + DiceLoss
- **Config:** `configs/building_seg_unet.yaml`

### U-Net Advanced (EfficientNet-B3)
- **Task:** Binary building localization
- **Input:** 3-channel RGB image
- **Output:** Binary mask
- **Encoder:** EfficientNet-B3
- **Loss:** BCEWithLogitsLoss + DiceLoss

### SegFormer-B2
- **Task:** Binary building segmentation
- **Input:** 3-channel RGB image
- **Output:** 2-class segmentation (background, building)
- **Base:** `nvidia/segformer-b2-finetuned-ade-512-512`
- **Config:** `configs/segformer_building.yaml`

---

## Damage Segmentation

### U-Net (EfficientNet-B3, 6-channel)
- **Task:** Multiclass damage segmentation
- **Input:** 6-channel (pre RGB + post RGB)
- **Output:** 5 classes (background, no-damage, minor, major, destroyed)
- **Loss:** Weighted CrossEntropy + DiceLoss
- **Config:** `configs/damage_seg_unet_6ch.yaml`

### SegFormer-B3
- **Task:** Multiclass damage segmentation
- **Input:** 3-channel (post-disaster) or 6-channel (pre+post)
- **Output:** 5 classes
- **Base:** `nvidia/segformer-b3-finetuned-ade-512-512`
- **Config:** `configs/segformer_damage.yaml`

---

## Change Detection

### Siamese U-Net (ResNet34)
- **Task:** Binary change detection
- **Input:** Pre + post disaster image pairs
- **Output:** Binary change mask
- **Fusion:** Concatenation + absolute difference
- **Loss:** BCEWithLogitsLoss + DiceLoss
- **Config:** `configs/change_siamese_unet.yaml`

### ChangeFormer
- **Task:** Binary change detection
- **Input:** Pre + post disaster image pairs
- **Output:** Binary change mask
- **Architecture:** Shared encoder + Transformer fusion + Decoder
- **Config:** `configs/changeformer_xbd.yaml`

---

## Damage Classification

### EfficientNet-B3 Classifier
- **Task:** Building-level damage classification
- **Input:** Pre + post building crop pairs
- **Output:** 4 classes (no-damage, minor, major, destroyed)
- **Loss:** Weighted CrossEntropy
- **Config:** `configs/efficientnet_damage_classifier.yaml`

### DINOv2 + MLP
- **Task:** Building-level damage classification
- **Input:** Pre + post building crop pairs
- **Output:** 4 classes
- **Backbone:** `facebook/dinov2-base` (frozen)
- **Fusion:** concat(pre_emb, post_emb, abs(pre_emb - post_emb)) -> MLP
- **Config:** `configs/dinov2_damage_classifier.yaml`

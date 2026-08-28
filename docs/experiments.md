# Experiment Log

| ID | Model | Dataset | Task | Metric | Goal | Status |
|----|-------|---------|------|--------|------|--------|
| E1 | U-Net ResNet34 | xBD | Building segmentation | IoU | Baseline building seg | Pending |
| E2 | U-Net EfficientNet-B3 | xBD | Damage segmentation | macro F1 / mIoU | Baseline damage seg | Pending |
| E3 | SegFormer-B2 | xBD | Building segmentation | IoU | Transformer improvement | Pending |
| E4 | SegFormer-B3 | xBD | Damage segmentation | mIoU | Transformer damage seg | Pending |
| E5 | Siamese U-Net | xBD | Change detection | IoU | Baseline change detection | Pending |
| E6 | ChangeFormer | xBD | Change detection | IoU / F1 | Advanced change detection | Pending |
| E7 | EfficientNet-B3 | xBD crops | Damage classification | macro F1 | Baseline crop classifier | Pending |
| E8 | DINOv2 + MLP | xBD crops | Damage classification | macro F1 | Foundation model classifier | Pending |
| E9 | SegFormer | FloodNet | Flood segmentation | mIoU | Flooded road detection | Pending |
| E10 | DINOv2 + FAISS | xBD | Similar case retrieval | Top-k qualitative | Retrieval system | Pending |
| E11 | Template report | xBD outputs | Report generation | Human review | Template baseline | Pending |

## How to Run Experiments

```bash
# E1: Building segmentation baseline
python scripts/train_building_seg.py --config configs/building_seg_unet.yaml

# E2: Damage segmentation baseline
python scripts/train_damage_seg.py --config configs/damage_seg_unet_6ch.yaml

# E3: SegFormer building segmentation
python scripts/train_building_seg.py --config configs/segformer_building.yaml

# E5: Change detection baseline
python scripts/train_change_detection.py --config configs/change_siamese_unet.yaml

# E7: Crop classifier baseline
python scripts/train_damage_classifier.py --config configs/efficientnet_damage_classifier.yaml

# E8: DINOv2 classifier
python scripts/train_damage_classifier.py --config configs/dinov2_damage_classifier.yaml
```

## Notes

- Use `--debug-limit 50` for quick validation runs
- Use `--max-epochs 1` for smoke tests
- All experiments log to TensorBoard under `outputs/logs/`
- Checkpoints saved to `outputs/checkpoints/`

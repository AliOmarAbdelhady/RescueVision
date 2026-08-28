# RescueVision: Multimodal Deep Learning System for Disaster Damage Assessment

A complete AI pipeline for automated post-disaster damage assessment using satellite and UAV imagery.

## Features

- **Building Segmentation**: U-Net with ResNet34/EfficientNet encoders for building localization
- **Damage Segmentation**: 6-channel pre/post input for pixel-level damage classification (5 classes)
- **Change Detection**: Siamese U-Net for pre/post disaster change mapping
- **Building-Level Classification**: DINOv2 + MLP and EfficientNet classifiers for per-building damage assessment
- **Flood/Road Segmentation**: SegFormer for FloodNet 10-class semantic segmentation
- **Inference Orchestrator**: End-to-end prediction pipeline
- **Report Generation**: Template-based emergency assessment reports with PDF export

## Supported Datasets

| Dataset | Role | Source |
|---------|------|--------|
| xBD/xView2 | Main dataset — building localization, damage classification, change detection | [Kaggle](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test) |
| FloodNet | Flood/road UAV imagery segmentation | [Kaggle](https://www.kaggle.com/datasets/aletbm/aerial-imagery-dataset-floodnet-challenge) |

## Quick Start

### Installation

```bash
cd RescueVision
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Dataset Setup

1. Download xBD from Kaggle and place in `data/raw/xbd/` with structure:
   ```
   data/raw/xbd/
   ├── train/images/  train/labels/
   ├── hold/images/   hold/labels/
   └── test/images/   test/labels/
   ```

2. (Optional) Download FloodNet and place in `data/raw/floodnet/`

### Preprocessing

```bash
python scripts/prepare_xbd.py \
  --raw-dir data/raw/xbd \
  --out-dir data/processed/xbd \
  --make-crops --make-visualizations
```

### Training (Local)

```bash
python scripts/train_building_seg.py --config configs/building_seg_unet.yaml
python scripts/train_damage_seg.py --config configs/damage_seg_unet_6ch.yaml
python scripts/train_change_detection.py --config configs/change_siamese_unet.yaml
python scripts/train_damage_classifier.py --config configs/dinov2_damage_classifier.yaml
```

### Training (Kaggle)

Use the notebooks in `notebooks/` on Kaggle with GPU enabled. Each notebook auto-detects the Kaggle environment and adjusts paths.

### Inference

```bash
python scripts/run_inference.py \
  --config configs/inference.yaml \
  --pre-image path/to/pre.png \
  --post-image path/to/post.png \
  --out-dir outputs/predictions/demo
```

### Report Export

```bash
python scripts/export_report.py \
  --summary outputs/predictions/demo/emergency_summary.json \
  --damage-map outputs/predictions/demo/damage_map.png \
  --out-dir outputs/reports
```

## Project Structure

```
RescueVision/
├── configs/           # YAML experiment configurations
├── data/              # Raw and processed datasets
├── notebooks/         # Kaggle-ready training notebooks
├── scripts/           # CLI entry points
├── src/rescuevision/  # Core Python package
│   ├── data/          # Dataset parsing, preprocessing, transforms
│   ├── models/        # Model architectures (U-Net, Siamese, DINOv2, etc.)
│   ├── training/      # Training loops, optimizers, callbacks
│   ├── evaluation/    # Metrics for segmentation and classification
│   ├── inference/     # End-to-end prediction pipeline
│   ├── reporting/     # Report generation and PDF export
│   └── utils/         # Config, seeding, geometry, I/O helpers
├── tests/             # Unit tests
└── outputs/           # Checkpoints, logs, predictions, reports
```

## Damage Classes

| ID | Label | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | No Damage | Green |
| 2 | Minor Damage | Yellow |
| 3 | Major Damage | Orange |
| 4 | Destroyed | Red |
| 255 | Unclassified (ignore) | Gray |

## Limitations and Safety Disclaimer

This system is a **research prototype** for decision support only. It must NOT be used as the sole basis for emergency response decisions. All outputs require review by qualified emergency management and remote-sensing professionals. Models may fail on unseen locations, disaster types, or imagery with cloud cover, smoke, or registration errors.

## License

MIT

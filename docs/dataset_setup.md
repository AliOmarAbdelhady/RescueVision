# Dataset Setup Guide

## xBD / xView2 (Required)

The xBD dataset is the primary dataset for RescueVision.

### Download

1. Register at [xView2](https://xview2.org/)
2. Download the challenge dataset
3. Extract to `data/raw/xbd/`

### Expected Structure

```
data/raw/xbd/
├── train/
│   ├── images/
│   │   ├── disaster_name_id_pre_disaster.png
│   │   └── disaster_name_id_post_disaster.png
│   └── labels/
│       ├── disaster_name_id_pre_disaster.json
│       └── disaster_name_id_post_disaster.json
├── hold/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Preprocessing

```bash
python scripts/prepare_xbd.py \
  --raw-dir data/raw/xbd \
  --out-dir data/processed/xbd \
  --make-crops \
  --make-visualizations \
  --debug-limit 100
```

## FloodNet (Optional)

UAV flood imagery for flood/road segmentation.

### Download

1. Visit [FloodNet benchmark page](https://roc-hci.github.io/NADBenchmarks/FloodNet.html)
2. Download the dataset
3. Extract to `data/raw/floodnet/`

### Expected Structure

```
data/raw/floodnet/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Preprocessing

```bash
python scripts/prepare_floodnet.py \
  --raw-dir data/raw/floodnet \
  --out-dir data/processed/floodnet
```

## SpaceNet (Optional)

Building/road extraction pretraining.

1. Visit [SpaceNet](https://spacenet.ai/datasets/)
2. Download SpaceNet 2 (buildings) or SpaceNet 3 (roads)
3. Extract to `data/raw/spacenet/`

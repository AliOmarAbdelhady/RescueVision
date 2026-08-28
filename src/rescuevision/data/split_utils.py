"""Event-aware train/val/test splitting for xBD dataset.

Groups tiles by disaster event to prevent data leakage across splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def generate_event_aware_splits(
    tiles_csv_path: str | Path,
    output_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by_disaster_type: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate event-aware splits grouped by disaster_name.

    Ensures all tiles from the same disaster event go into the same split.
    Optionally stratifies by disaster_type to balance disaster categories.

    Args:
        tiles_csv_path: Path to tiles.csv from preprocessing.
        output_path: Where to save the splits CSV.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        stratify_by_disaster_type: Whether to stratify by disaster type.
        seed: Random seed.

    Returns:
        DataFrame with tile_id, split, disaster_name columns.
    """
    tiles_df = pd.read_csv(tiles_csv_path)

    # Get unique disaster events with their types
    events = (
        tiles_df.groupby("disaster_name")
        .agg({
            "disaster_type": "first",
            "tile_id": list,
        })
        .reset_index()
    )

    logger.info("Found %d unique disaster events", len(events))

    # Determine test split ratio
    test_size = test_ratio
    val_size_relative = val_ratio / (train_ratio + val_ratio)

    disaster_names = events["disaster_name"].values
    stratify = events["disaster_type"].values if stratify_by_disaster_type else None

    # Split into train+val and test
    try:
        trainval_names, test_names = train_test_split(
            disaster_names,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        logger.warning("Too few events for stratified split, falling back to random")
        trainval_names, test_names = train_test_split(
            disaster_names, test_size=test_size, random_state=seed
        )

    # Split train+val into train and val
    trainval_events = events[events["disaster_name"].isin(trainval_names)]
    stratify_tv = (
        trainval_events["disaster_type"].values if stratify_by_disaster_type else None
    )

    try:
        train_names, val_names = train_test_split(
            trainval_names,
            test_size=val_size_relative,
            random_state=seed,
            stratify=stratify_tv,
        )
    except ValueError:
        train_names, val_names = train_test_split(
            trainval_names, test_size=val_size_relative, random_state=seed
        )

    # Assign splits
    split_map = {}
    for name in train_names:
        split_map[name] = "train"
    for name in val_names:
        split_map[name] = "val"
    for name in test_names:
        split_map[name] = "test"

    # Build output DataFrame
    rows = []
    for _, row in tiles_df.iterrows():
        disaster_name = row["disaster_name"]
        split = split_map.get(disaster_name, "train")  # default to train if missing
        rows.append({
            "tile_id": row["tile_id"],
            "split": split,
            "disaster_name": disaster_name,
            "disaster_type": row.get("disaster_type", ""),
        })

    splits_df = pd.DataFrame(rows)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False)

    # Log stats
    split_counts = splits_df["split"].value_counts()
    logger.info("Event-aware splits:\n%s", split_counts.to_string())
    for split_name in ["train", "val", "test"]:
        events_in_split = splits_df[splits_df["split"] == split_name]["disaster_name"].nunique()
        tiles_in_split = (splits_df["split"] == split_name).sum()
        logger.info("  %s: %d events, %d tiles", split_name, events_in_split, tiles_in_split)

    return splits_df

"""Environment-aware YAML configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def is_kaggle() -> bool:
    """Detect if running on Kaggle."""
    return os.path.exists("/kaggle/input")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Override data/output paths based on environment (Kaggle vs local).

    On Kaggle:
    - data_root -> /kaggle/input/<dataset>
    - outputs -> /kaggle/working/
    """
    if is_kaggle():
        kaggle_data = os.environ.get(
            "KAGGLE_DATA_ROOT", "/kaggle/input/xview2-challenge-dataset-train-and-test"
        )
        config["data_root"] = kaggle_data
        config["raw_xbd_dir"] = str(Path(kaggle_data))
        config["processed_xbd_dir"] = "/kaggle/working/processed_xbd"
        config["outputs_dir"] = "/kaggle/working"
        config["checkpoints_dir"] = "/kaggle/working/checkpoints"
        config["logs_dir"] = "/kaggle/working/logs"
        config["figures_dir"] = "/kaggle/working/figures"

        if "dataset" in config:
            ds = config["dataset"]
            ds["processed_dir"] = "/kaggle/working/processed_xbd"
            if "split_csv" in ds:
                ds["split_csv"] = str(
                    Path("/kaggle/working/processed_xbd") / "metadata" / "splits_event_aware.csv"
                )
            if "crops_csv" in ds:
                ds["crops_csv"] = str(
                    Path("/kaggle/working/processed_xbd") / "metadata" / "buildings.csv"
                )
    return config


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config with environment overrides and optional CLI overrides.

    Also loads paths.yaml from the same directory and merges it in.
    """
    config_path = Path(config_path)
    config = load_yaml(config_path)

    # Load and merge paths.yaml if it exists
    paths_file = config_path.parent / "paths.yaml"
    if paths_file.exists():
        paths = load_yaml(paths_file)
        for k, v in paths.items():
            config.setdefault(k, v)

    config = resolve_paths(config)

    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

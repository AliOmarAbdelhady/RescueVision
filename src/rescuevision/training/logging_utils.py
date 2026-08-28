"""Logging utilities: CSV logger for training metrics."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CSVLogger:
    """Log training metrics to a CSV file."""

    def __init__(self, log_dir: str | Path, filename: str = "training_log.csv"):
        self.log_path = Path(log_dir) / filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def log(self, metrics: dict[str, float], epoch: int) -> None:
        """Log a row of metrics."""
        row = {"epoch": epoch, **metrics}

        if not self._header_written:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            self._header_written = True
        else:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

    def get_best(self, metric_name: str, mode: str = "max") -> tuple[float, int]:
        """Get best metric value and epoch from log."""
        best_val = float("-inf") if mode == "max" else float("inf")
        best_epoch = 0

        with open(self.log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = float(row[metric_name])
                if (mode == "max" and val > best_val) or (mode == "min" and val < best_val):
                    best_val = val
                    best_epoch = int(row["epoch"])

        return best_val, best_epoch

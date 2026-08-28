"""Batch inference for processing multiple image pairs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from rescuevision.inference.predictor import DisasterPredictor

logger = logging.getLogger(__name__)


def run_batch_inference(
    predictor: DisasterPredictor,
    pairs_csv: str | Path,
    output_root: str | Path,
) -> list[dict]:
    """Run inference on multiple image pairs from a CSV file.

    CSV must have columns: pre_image, post_image, (optional) case_id
    """
    pairs_csv = Path(pairs_csv)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_csv)
    results = []

    for idx, row in df.iterrows():
        case_id = row.get("case_id", f"case_{idx:04d}")
        pre_path = row["pre_image"]
        post_path = row["post_image"]

        case_dir = output_root / case_id

        logger.info("Processing case %s: %s", case_id, Path(pre_path).name)

        try:
            result = predictor.predict(pre_path, post_path, case_dir)
            results.append({
                "case_id": case_id,
                "status": "success",
                "summary": result.get("summary", {}),
            })
        except Exception as e:
            logger.error("Error processing case %s: %s", case_id, e)
            results.append({
                "case_id": case_id,
                "status": "error",
                "error": str(e),
            })

    # Save batch summary
    with open(output_root / "batch_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Batch inference complete: %d cases processed", len(results))
    return results

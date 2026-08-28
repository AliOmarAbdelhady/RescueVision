"""Build FAISS index from disaster image embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from rescuevision.retrieval.embedding import DINOv2Embedder

logger = logging.getLogger(__name__)


def build_faiss_index(
    image_dir: str | Path,
    output_dir: str | Path,
    metadata_csv: str | Path | None = None,
    pattern: str = "*_post_disaster.png",
    backbone: str = "facebook/dinov2-base",
    device: str = "cuda",
    batch_size: int = 16,
) -> Path:
    """Build a FAISS index from disaster images.

    Args:
        image_dir: Directory containing images to index.
        output_dir: Directory to save the index and metadata.
        metadata_csv: Optional CSV with image metadata.
        pattern: Glob pattern for finding images.
        backbone: DINOv2 model name.
        device: Device for embedding extraction.
        batch_size: Batch size for embedding extraction.

    Returns:
        Path to the FAISS index file.
    """
    import faiss

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = sorted(image_dir.rglob(pattern))
    if not images:
        logger.warning("No images found matching %s in %s", pattern, image_dir)
        # Fallback: try any PNG
        images = sorted(image_dir.rglob("*.png"))

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    logger.info("Found %d images to index", len(images))

    # Extract embeddings
    embedder = DINOv2Embedder(backbone=backbone, device=device)
    embeddings = embedder.embed_batch([str(p) for p in images], batch_size=batch_size)

    dim = embeddings.shape[1]
    logger.info("Embedding dimension: %d", dim)

    # Build FAISS index (inner product since embeddings are L2-normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    index_path = output_dir / "disaster_cases.index"
    faiss.write_index(index, str(index_path))

    # Save metadata
    metadata = []
    for i, img_path in enumerate(images):
        entry = {
            "index": i,
            "tile_id": img_path.stem,
            "image_path": str(img_path),
            "disaster_type": "",
            "damage_distribution": {},
        }

        # Try to extract disaster info from path
        parts = img_path.stem.split("_")
        if len(parts) >= 2:
            entry["disaster_type"] = parts[0]

        metadata.append(entry)

    metadata_path = output_dir / "disaster_cases_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Optionally enrich from CSV
    if metadata_csv and Path(metadata_csv).exists():
        import pandas as pd
        df = pd.read_csv(metadata_csv)
        for entry in metadata:
            tile_id = entry["tile_id"].replace("_post_disaster", "")
            match = df[df["tile_id"] == tile_id]
            if not match.empty:
                row = match.iloc[0]
                if "disaster_type" in row:
                    entry["disaster_type"] = str(row["disaster_type"])
                if "disaster_name" in row:
                    entry["disaster_name"] = str(row["disaster_name"])

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("FAISS index saved to %s (%d entries)", index_path, len(metadata))
    return index_path

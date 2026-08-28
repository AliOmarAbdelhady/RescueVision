"""Search FAISS index for similar disaster cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def search_index(
    index_path: str | Path,
    metadata_path: str | Path,
    query_image_path: str | Path,
    top_k: int = 5,
    backbone: str = "facebook/dinov2-base",
    device: str = "cuda",
) -> list[dict]:
    """Search for similar disaster cases using FAISS.

    Args:
        index_path: Path to FAISS index file.
        metadata_path: Path to metadata JSON file.
        query_image_path: Path to query image.
        top_k: Number of results to return.
        backbone: DINOv2 model name.
        device: Device for embedding extraction.

    Returns:
        List of dicts with tile_id, similarity, and metadata.
    """
    import faiss

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load index and metadata
    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Extract query embedding
    from rescuevision.retrieval.embedding import DINOv2Embedder
    embedder = DINOv2Embedder(backbone=backbone, device=device)
    query_embedding = embedder.embed_image(query_image_path).reshape(1, -1)

    # Search
    k = min(top_k, len(metadata))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx].copy()
        entry["similarity"] = float(score)
        results.append(entry)

    return results


def search_by_embedding(
    index_path: str | Path,
    metadata_path: str | Path,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Search using a pre-computed embedding vector.

    Args:
        index_path: Path to FAISS index.
        metadata_path: Path to metadata JSON.
        query_embedding: numpy array of shape (dim,).
        top_k: Number of results.

    Returns:
        List of result dicts.
    """
    import faiss

    index = faiss.read_index(str(Path(index_path)))
    with open(Path(metadata_path)) as f:
        metadata = json.load(f)

    query = query_embedding.reshape(1, -1).astype(np.float32)
    k = min(top_k, len(metadata))
    scores, indices = index.search(query, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx].copy()
        entry["similarity"] = float(score)
        results.append(entry)

    return results

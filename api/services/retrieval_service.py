"""Retrieval service: find similar disaster cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_index_path: Path | None = None
_metadata_path: Path | None = None


def init_service(index_dir: str) -> None:
    global _index_path, _metadata_path
    idx_dir = Path(index_dir)
    _index_path = idx_dir / "disaster_cases.index"
    _metadata_path = idx_dir / "disaster_cases_metadata.json"


def search_similar(query_image_path: str, top_k: int = 5) -> list[dict]:
    """Search for similar disaster cases."""
    if _index_path is None or not _index_path.exists():
        logger.warning("FAISS index not found at %s", _index_path)
        return []

    try:
        from rescuevision.retrieval.search import search_index
        results = search_index(
            index_path=str(_index_path),
            metadata_path=str(_metadata_path),
            query_image_path=query_image_path,
            top_k=top_k,
        )
        return results
    except Exception as e:
        logger.error("Retrieval search failed: %s", e)
        return []

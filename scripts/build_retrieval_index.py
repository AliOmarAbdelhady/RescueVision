#!/usr/bin/env python3
"""CLI script for building the FAISS retrieval index."""

import argparse
import logging
import sys

from rescuevision.retrieval.build_index import build_faiss_index
from rescuevision.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index for disaster images")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--image-dir", default=None, help="Override image directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--backbone", default=None, help="Override DINOv2 backbone")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--debug-limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config) if args.config else {}

    image_dir = args.image_dir or config.get("image_dir", "data/processed/xbd/images/post")
    output_dir = args.output_dir or config.get("output_dir", "outputs/faiss")
    backbone = args.backbone or config.get("backbone", "facebook/dinov2-base")

    if args.debug_limit:
        logging.info("Debug mode: limiting to %d images", args.debug_limit)

    index_path = build_faiss_index(
        image_dir=image_dir,
        output_dir=output_dir,
        metadata_csv=config.get("metadata_csv"),
        pattern=config.get("pattern", "*_post_disaster.png"),
        backbone=backbone,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\nFAISS index saved to: {index_path}")


if __name__ == "__main__":
    main()

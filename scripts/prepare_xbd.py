#!/usr/bin/env python3
"""CLI script for xBD dataset preprocessing."""

import argparse
import logging
import sys

from rescuevision.data.xbd_preprocess import preprocess_xbd


def main():
    parser = argparse.ArgumentParser(description="Preprocess xBD dataset")
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw/xbd",
        help="Path to raw xBD data directory",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data/processed/xbd",
        help="Path to write processed output",
    )
    parser.add_argument(
        "--image-size", type=int, default=1024,
        help="Recorded image size (no resizing applied)",
    )
    parser.add_argument(
        "--make-crops", action="store_true",
        help="Generate per-building crops",
    )
    parser.add_argument(
        "--make-visualizations", action="store_true",
        help="Save sanity-check visualizations",
    )
    parser.add_argument(
        "--crop-margin", type=int, default=16,
        help="Pixel margin around building bounding boxes",
    )
    parser.add_argument(
        "--debug-limit", type=int, default=None,
        help="Limit number of tiles for debugging",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    preprocess_xbd(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        image_size=args.image_size,
        make_crops=args.make_crops,
        make_visualizations=args.make_visualizations,
        crop_margin=args.crop_margin,
        debug_limit=args.debug_limit,
    )


if __name__ == "__main__":
    main()

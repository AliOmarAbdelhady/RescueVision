#!/usr/bin/env python3
"""CLI script for FloodNet preprocessing."""

import argparse
import logging
import sys

from rescuevision.data.floodnet_preprocess import prepare_floodnet


def main():
    parser = argparse.ArgumentParser(description="Prepare FloodNet dataset")
    parser.add_argument("--raw-dir", default="data/raw/floodnet")
    parser.add_argument("--out-dir", default="data/processed/floodnet")
    parser.add_argument("--debug-limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare_floodnet(args.raw_dir, args.out_dir, args.debug_limit)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI script for exporting emergency assessment reports."""

import argparse
import logging
import sys

from rescuevision.reporting.pdf_export import export_report_pdf_from_json


def main():
    parser = argparse.ArgumentParser(description="Export damage assessment report")
    parser.add_argument(
        "--summary", required=True,
        help="Path to emergency_summary.json from inference output",
    )
    parser.add_argument(
        "--out-dir", default="outputs/reports",
        help="Directory to save report outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_path = export_report_pdf_from_json(args.summary, args.out_dir)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()

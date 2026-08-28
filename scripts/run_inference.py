#!/usr/bin/env python3
"""CLI script for running end-to-end inference with optional report and retrieval."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rescuevision.inference.predictor import DisasterPredictor
from rescuevision.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run disaster damage inference")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--pre-image", required=True, help="Pre-disaster image path")
    parser.add_argument("--post-image", required=True, help="Post-disaster image path")
    parser.add_argument("--out-dir", default="outputs/predictions/demo")
    parser.add_argument("--enable-report", action="store_true", default=True)
    parser.add_argument("--enable-retrieval", action="store_true", default=False)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = load_config(args.config)

    predictor = DisasterPredictor(config)

    results = predictor.predict(
        pre_image_path=args.pre_image,
        post_image_path=args.post_image,
        output_dir=args.out_dir,
    )

    out_dir = Path(args.out_dir)
    summary = results.get("summary", {})

    # Generate report
    enable_report = args.enable_report and not args.no_report
    if enable_report:
        from rescuevision.reporting.template_report import generate_report
        from rescuevision.reporting.pdf_export import export_report_pdf

        artifacts = {
            "building_mask": str(out_dir / "building_mask.png"),
            "damage_map": str(out_dir / "damage_map.png"),
        }
        report, md_text = generate_report(
            summary,
            case_id=out_dir.name,
            pre_image=args.pre_image,
            post_image=args.post_image,
            artifacts=artifacts,
        )
        (out_dir / "report.md").write_text(md_text)
        print(f"Report (markdown) saved to: {out_dir / 'report.md'}")

        try:
            pdf_path = export_report_pdf(summary, out_dir / "report", case_id=out_dir.name)
            print(f"Report (PDF) saved to: {pdf_path}")
        except Exception as e:
            logging.warning("PDF export failed: %s", e)

    # Run retrieval
    if args.enable_retrieval:
        faiss_dir = Path(config.get("faiss_dir", "outputs/faiss"))
        index_path = faiss_dir / "disaster_cases.index"
        if index_path.exists():
            from rescuevision.retrieval.search import search_index
            similar = search_index(
                str(index_path),
                str(faiss_dir / "disaster_cases_metadata.json"),
                args.post_image,
                top_k=5,
                device=config.get("device", "cuda"),
            )
            with open(out_dir / "similar_cases.json", "w") as f:
                json.dump(similar, f, indent=2)
            print(f"Similar cases saved to: {out_dir / 'similar_cases.json'}")
        else:
            logging.warning("FAISS index not found at %s. Run build_retrieval_index.py first.", index_path)

    print("\n=== Prediction Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

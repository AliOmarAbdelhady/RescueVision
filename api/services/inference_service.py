"""Inference service: wraps DisasterPredictor for API use."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from rescuevision.utils.config import load_config
from rescuevision.utils.io import ensure_dir

logger = logging.getLogger(__name__)

_outputs_root: Path | None = None
_predictor: Any | None = None


def init_service(config_path: str, outputs_root: str) -> None:
    """Initialize the inference service with config and output directory."""
    global _outputs_root, _predictor
    _outputs_root = Path(outputs_root)
    _outputs_root.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    device = config.get("device", "cuda")

    try:
        from rescuevision.inference.predictor import DisasterPredictor
        _predictor = DisasterPredictor(config, device=device)
        logger.info("Inference service initialized successfully")
    except Exception as e:
        logger.warning("Could not load all models: %s", e)
        _predictor = None


def get_predictor() -> Any | None:
    return _predictor


def get_models_status() -> dict[str, bool]:
    """Return loading status of each model."""
    if _predictor is None:
        return {
            "building_seg": False,
            "damage_seg": False,
            "change_detection": False,
            "damage_classifier": False,
        }
    return {
        "building_seg": _predictor.building_model is not None,
        "damage_seg": _predictor.damage_seg_model is not None,
        "change_detection": _predictor.change_model is not None,
        "damage_classifier": _predictor.damage_cls_model is not None,
    }


def run_prediction(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    case_name: str | None = None,
    enable_report: bool = True,
) -> dict[str, Any]:
    """Run prediction pipeline and return results dict."""
    case_id = case_name or str(uuid.uuid4())[:8]
    case_dir = ensure_dir(_outputs_root / case_id)

    # Copy uploaded images to case directory
    pre_dest = case_dir / "pre_disaster.png"
    post_dest = case_dir / "post_disaster.png"
    shutil.copy2(pre_image_path, pre_dest)
    shutil.copy2(post_image_path, post_dest)

    # Run inference
    if _predictor is None:
        raise RuntimeError("Predictor not initialized")

    results = _predictor.predict(str(pre_dest), str(post_dest), str(case_dir))

    summary = results.get("summary", {})
    artifacts = {
        "building_mask": f"/outputs/{case_id}/building_mask.png",
        "change_mask": f"/outputs/{case_id}/change_mask.png",
        "damage_map": f"/outputs/{case_id}/damage_map.png",
        "overlay_damage": f"/outputs/{case_id}/overlay_damage.png",
        "overview": f"/outputs/{case_id}/overview.png",
    }

    # Generate report if requested
    if enable_report:
        from rescuevision.reporting.template_report import generate_report
        from rescuevision.reporting.pdf_export import export_report_pdf

        report, md_text = generate_report(
            summary,
            case_id=case_id,
            pre_image=str(pre_dest),
            post_image=str(post_dest),
            artifacts=artifacts,
        )

        md_path = case_dir / "report.md"
        md_path.write_text(md_text)
        artifacts["report_md"] = f"/outputs/{case_id}/report.md"

        try:
            pdf_path = export_report_pdf(summary, case_dir / "report", case_id=case_id)
            suffix = pdf_path.suffix
            artifacts["report_pdf"] = f"/outputs/{case_id}/report{suffix}"
        except Exception as e:
            logger.warning("PDF export failed: %s", e)

    # Save case metadata
    case_meta = {
        "case_id": case_id,
        "summary": summary,
        "artifacts": artifacts,
    }
    with open(case_dir / "case_meta.json", "w") as f:
        json.dump(case_meta, f, indent=2)

    return case_meta

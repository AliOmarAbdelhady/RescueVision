"""Report service: retrieve and generate reports for cases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_outputs_root: Path | None = None


def init_service(outputs_root: str) -> None:
    global _outputs_root
    _outputs_root = Path(outputs_root)


def get_report(case_id: str) -> dict | None:
    """Retrieve report data for a case."""
    case_dir = _outputs_root / case_id
    if not case_dir.exists():
        return None

    result = {"case_id": case_id}

    md_path = case_dir / "report.md"
    if md_path.exists():
        result["report_markdown"] = md_path.read_text()

    html_path = case_dir / "report.html"
    if html_path.exists():
        result["report_html"] = html_path.read_text()

    pdf_path = case_dir / "report.pdf"
    if pdf_path.exists():
        result["report_pdf"] = f"/outputs/{case_id}/report.pdf"

    meta_path = case_dir / "case_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            result["summary"] = meta.get("summary", {})

    return result if len(result) > 1 else None


def generate_report_for_case(case_id: str) -> dict | None:
    """Generate a report from existing prediction results."""
    case_dir = _outputs_root / case_id
    meta_path = case_dir / "case_meta.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    summary = meta.get("summary", {})

    from rescuevision.reporting.template_report import generate_report
    from rescuevision.reporting.pdf_export import export_report_pdf

    _, md_text = generate_report(summary, case_id=case_id)
    (case_dir / "report.md").write_text(md_text)

    try:
        export_report_pdf(summary, case_dir / "report", case_id=case_id)
    except Exception as e:
        logger.warning("PDF export failed: %s", e)

    return get_report(case_id)

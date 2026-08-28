"""RescueVision FastAPI backend."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_root = str(Path(__file__).resolve().parent.parent / "src")
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from api.schemas import (
    ArtifactUrls,
    CaseDetailResponse,
    DamageSummaryResponse,
    HealthResponse,
    PredictionResponse,
    ReportResponse,
    UrgencyResponse,
)
from api.services import inference_service, report_service, retrieval_service

app = FastAPI(
    title="RescueVision API",
    description="Multimodal Deep Learning System for Disaster Damage Assessment",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.environ.get("RESCUEVISION_CONFIG", "configs/inference.yaml")
OUTPUTS_DIR = os.environ.get("RESCUEVISION_OUTPUTS", "outputs")
FAISS_DIR = os.environ.get("RESCUEVISION_FAISS", "outputs/faiss")


@app.on_event("startup")
async def startup() -> None:
    inference_service.init_service(CONFIG_PATH, OUTPUTS_DIR)
    report_service.init_service(OUTPUTS_DIR)
    retrieval_service.init_service(FAISS_DIR)

    outputs_path = Path(OUTPUTS_DIR)
    if outputs_path.exists():
        app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=inference_service.get_models_status(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    pre_image: UploadFile = File(..., description="Pre-disaster satellite image"),
    post_image: UploadFile = File(..., description="Post-disaster satellite image"),
    case_name: str | None = Form(None),
    enable_report: bool = Form(True),
    enable_retrieval: bool = Form(False),
) -> PredictionResponse:
    with tempfile.TemporaryDirectory() as tmpdir:
        pre_path = Path(tmpdir) / "pre_disaster.png"
        post_path = Path(tmpdir) / "post_disaster.png"

        pre_path.write_bytes(await pre_image.read())
        post_path.write_bytes(await post_image.read())

        try:
            case_meta = inference_service.run_prediction(
                pre_image_path=str(pre_path),
                post_image_path=str(post_path),
                case_name=case_name,
                enable_report=enable_report,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    summary = case_meta.get("summary", {})
    artifacts = case_meta.get("artifacts", {})

    return PredictionResponse(
        case_id=case_meta["case_id"],
        summary=DamageSummaryResponse(
            total_buildings=summary.get("total_buildings", 0),
            no_damage=summary.get("no_damage", 0),
            minor_damage=summary.get("minor_damage", 0),
            major_damage=summary.get("major_damage", 0),
            destroyed=summary.get("destroyed", 0),
        ),
        urgency=UrgencyResponse(
            score=summary.get("urgency_score", 0.0),
            level=summary.get("urgency_level", "LOW"),
        ),
        artifact_urls=ArtifactUrls(**{k: v for k, v in artifacts.items() if k in ArtifactUrls.model_fields}),
        warnings=[
            "This is a research prototype. Results require expert validation.",
            "Model confidence may be reduced in areas with cloud cover or smoke.",
        ],
    )


@app.get("/cases/{case_id}", response_model=CaseDetailResponse)
async def get_case(case_id: str) -> CaseDetailResponse:
    case_dir = Path(OUTPUTS_DIR) / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    meta_path = case_dir / "case_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"case_id": case_id, "summary": {}, "artifacts": {}}

    report_md_path = case_dir / "report.md"
    report_md = report_md_path.read_text() if report_md_path.exists() else None

    return CaseDetailResponse(
        case_id=case_id,
        summary=meta.get("summary", {}),
        artifacts=meta.get("artifacts", {}),
        report_markdown=report_md,
    )


@app.get("/reports/{case_id}", response_model=ReportResponse)
async def get_report(case_id: str) -> ReportResponse:
    report_data = report_service.get_report(case_id)
    if report_data is None:
        report_data = report_service.generate_report_for_case(case_id)
    if report_data is None:
        raise HTTPException(status_code=404, detail=f"No report found for case {case_id}")

    return ReportResponse(
        case_id=case_id,
        report_markdown=report_data.get("report_markdown", ""),
        report_html=report_data.get("report_html"),
        report_pdf=report_data.get("report_pdf"),
    )


@app.get("/reports/{case_id}/download")
async def download_report(case_id: str):
    case_dir = Path(OUTPUTS_DIR) / case_id
    pdf_path = case_dir / "report.pdf"
    if pdf_path.exists():
        return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"report_{case_id}.pdf")

    html_path = case_dir / "report.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html", filename=f"report_{case_id}.html")

    raise HTTPException(status_code=404, detail="No downloadable report found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

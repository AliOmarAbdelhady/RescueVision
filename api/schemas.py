"""Pydantic schemas for the RescueVision API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    case_name: str | None = None
    enable_retrieval: bool = True
    enable_report: bool = True
    enable_routing: bool = False


class DamageSummaryResponse(BaseModel):
    total_buildings: int
    no_damage: int
    minor_damage: int
    major_damage: int
    destroyed: int


class UrgencyResponse(BaseModel):
    score: float
    level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ArtifactUrls(BaseModel):
    building_mask: str = ""
    change_mask: str = ""
    damage_map: str = ""
    overlay_damage: str = ""
    overview: str = ""
    report_md: str = ""
    report_pdf: str = ""


class PredictionResponse(BaseModel):
    case_id: str
    summary: DamageSummaryResponse
    urgency: UrgencyResponse
    artifact_urls: ArtifactUrls
    warnings: list[str] = Field(default_factory=list)


class CaseDetailResponse(BaseModel):
    case_id: str
    summary: dict
    artifacts: dict[str, str]
    report_markdown: str | None = None


class ReportResponse(BaseModel):
    case_id: str
    report_markdown: str
    report_html: str | None = None
    report_pdf: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    models_loaded: dict[str, bool] = Field(default_factory=dict)

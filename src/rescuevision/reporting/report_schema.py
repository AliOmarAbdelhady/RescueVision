"""Pydantic data models for emergency assessment reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rescuevision.constants import SAFETY_DISCLAIMER


class DamageSummary(BaseModel):
    total_buildings: int = Field(ge=0)
    no_damage: int = Field(ge=0)
    minor_damage: int = Field(ge=0)
    major_damage: int = Field(ge=0)
    destroyed: int = Field(ge=0)


class UrgencyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class FloodSummary(BaseModel):
    flooded_roads: int = 0
    flooded_buildings: int = 0
    blocked_routes: list[str] = Field(default_factory=list)


class RouteSummary(BaseModel):
    safest_route: str = ""
    blocked_routes: list[str] = Field(default_factory=list)


class PredictionReport(BaseModel):
    """Complete emergency assessment report."""
    case_id: str
    pre_image: str = ""
    post_image: str = ""
    damage_summary: DamageSummary
    urgency: UrgencyScore
    flood_summary: FloodSummary | None = None
    route_summary: RouteSummary | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = SAFETY_DISCLAIMER


def compute_urgency(damage_summary: DamageSummary) -> UrgencyScore:
    """Compute urgency score from damage summary.

    Formula: 0.6 * destroyed_ratio + 0.3 * major_ratio
    """
    total = max(damage_summary.total_buildings, 1)
    destroyed_ratio = damage_summary.destroyed / total
    major_ratio = damage_summary.major_damage / total

    score = 0.6 * destroyed_ratio + 0.3 * major_ratio

    if score >= 0.45:
        level = "CRITICAL"
    elif score >= 0.25:
        level = "HIGH"
    elif score >= 0.10:
        level = "MEDIUM"
    else:
        level = "LOW"

    return UrgencyScore(score=round(score, 3), level=level)

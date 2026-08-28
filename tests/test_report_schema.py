"""Tests for report schema and generation."""

import pytest

from rescuevision.reporting.report_schema import (
    DamageSummary,
    UrgencyScore,
    compute_urgency,
)
from rescuevision.reporting.template_report import generate_report


class TestDamageSummary:
    def test_valid_summary(self):
        ds = DamageSummary(
            total_buildings=100,
            no_damage=50,
            minor_damage=20,
            major_damage=20,
            destroyed=10,
        )
        assert ds.total_buildings == 100

    def test_negative_rejected(self):
        with pytest.raises(Exception):
            DamageSummary(total_buildings=-1, no_damage=0, minor_damage=0, major_damage=0, destroyed=0)


class TestUrgencyComputation:
    def test_critical_urgency(self):
        ds = DamageSummary(total_buildings=100, no_damage=0, minor_damage=0, major_damage=0, destroyed=100)
        urgency = compute_urgency(ds)
        assert urgency.level == "CRITICAL"
        assert urgency.score >= 0.45

    def test_low_urgency(self):
        ds = DamageSummary(total_buildings=100, no_damage=90, minor_damage=5, major_damage=3, destroyed=2)
        urgency = compute_urgency(ds)
        assert urgency.level == "LOW"
        assert urgency.score < 0.10

    def test_medium_urgency(self):
        ds = DamageSummary(total_buildings=100, no_damage=50, minor_damage=15, major_damage=20, destroyed=15)
        urgency = compute_urgency(ds)
        assert urgency.level in ("MEDIUM", "HIGH")

    def test_zero_buildings(self):
        ds = DamageSummary(total_buildings=0, no_damage=0, minor_damage=0, major_damage=0, destroyed=0)
        urgency = compute_urgency(ds)
        assert urgency.level == "LOW"
        assert urgency.score == 0.0


class TestTemplateReport:
    def test_generate_report(self):
        summary = {
            "total_buildings": 50,
            "no_damage": 25,
            "minor_damage": 10,
            "major_damage": 10,
            "destroyed": 5,
        }
        report, md_text = generate_report(summary, case_id="test_001")
        assert "test_001" in md_text
        assert "50" in md_text
        assert "Structural Damage" in md_text

    def test_report_with_artifacts(self):
        summary = {
            "total_buildings": 10,
            "no_damage": 5,
            "minor_damage": 3,
            "major_damage": 1,
            "destroyed": 1,
        }
        report, md_text = generate_report(
            summary,
            case_id="test_002",
            artifacts={"damage_map": "outputs/damage_map.png"},
        )
        assert "damage_map.png" in md_text

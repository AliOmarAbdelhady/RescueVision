"""Smoke tests for the RescueVision API."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src is on path
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_predict_missing_files():
    response = client.post("/predict")
    assert response.status_code == 422  # Missing required fields


def test_case_not_found():
    response = client.get("/cases/nonexistent_case_id")
    assert response.status_code == 404


def test_report_not_found():
    response = client.get("/reports/nonexistent_case_id")
    assert response.status_code == 404

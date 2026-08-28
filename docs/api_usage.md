# API Usage

## Starting the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker compose up api
```

## Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "models_loaded": {
    "building_seg": true,
    "damage_seg": true,
    "change_detection": true,
    "damage_classifier": true
  }
}
```

### Run Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "pre_image=@pre_disaster.png" \
  -F "post_image=@post_disaster.png" \
  -F "enable_report=true"
```

Response:
```json
{
  "case_id": "abc12345",
  "summary": {
    "total_buildings": 184,
    "no_damage": 92,
    "minor_damage": 23,
    "major_damage": 42,
    "destroyed": 27
  },
  "urgency": {
    "score": 0.342,
    "level": "HIGH"
  },
  "artifact_urls": {
    "building_mask": "/outputs/abc12345/building_mask.png",
    "damage_map": "/outputs/abc12345/damage_map.png",
    "report_md": "/outputs/abc12345/report.md"
  },
  "warnings": [
    "This is a research prototype. Results require expert validation."
  ]
}
```

### Get Case Details

```bash
curl http://localhost:8000/cases/{case_id}
```

### Get Report

```bash
curl http://localhost:8000/reports/{case_id}
```

### Download Report PDF

```bash
curl -O http://localhost:8000/reports/{case_id}/download
```

## Interactive Docs

When running, visit `http://localhost:8000/docs` for Swagger UI.

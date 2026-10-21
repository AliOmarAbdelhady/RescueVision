#!/usr/bin/env bash
# One-time local setup: Python venv + deps + .env
# Usage:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required (brew install python@3.11)"'

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r mapping/requirements.txt
[ -d src ] && pip install -e . || echo "note: no src/ package yet, skipping editable install"

[ -f .env ] || cp .env.example .env
command -v gdal_translate >/dev/null || echo "note: GDAL CLI missing (needed for mapping thumbnails): brew install gdal"
echo "done — activate with: source .venv/bin/activate"

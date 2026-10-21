#!/usr/bin/env bash
# One-time local setup: Python venv + deps + .env
# Usage:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick the first Python >= 3.10 (macOS's built-in python3 is 3.9)
PY=""
for c in ${PYTHON:-} python3.12 python3.11 python3.10 \
         /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3,10))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python 3.10+ found. Install one, then re-run:"
  echo "    brew install python@3.11"
  exit 1
fi
echo "using $PY ($("$PY" --version))"

# Recreate .venv if it was built with an old Python
if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3,10))'; then
  echo "removing old .venv (Python < 3.10)"; rm -rf .venv
fi

[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r mapping/requirements.txt
[ -d src ] && pip install -e . || echo "note: no src/ package yet, skipping editable install"

[ -f .env ] || cp .env.example .env
command -v gdal_translate >/dev/null || echo "note: GDAL CLI missing (needed for mapping thumbnails): brew install gdal"
echo "done — activate with: source .venv/bin/activate"

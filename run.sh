#!/usr/bin/env bash
set -euo pipefail

# activa venv si existe
if [ -d ".venv" ]; then
  # macOS usa bash por defecto al ejecutar ./run.sh (no uses sh run.sh)
  source ".venv/bin/activate"
fi

export TIMEOUT_SEC=${TIMEOUT_SEC:-10}
export BLADERF_CLI_FORCED=${BLADERF_CLI_FORCED:-}   # si lo necesitas

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload

#!/usr/bin/env sh
set -eu  # (sin pipefail)

# activa venv si existe
if [ -d ".venv" ]; then
  # en sh se usa '.' en vez de 'source'
  . ".venv/bin/activate"
fi

export TIMEOUT_SEC=${TIMEOUT_SEC:-10}
export BLADERF_CLI_FORCED=${BLADERF_CLI_FORCED:-}

exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload

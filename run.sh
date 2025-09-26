#!/usr/bin/env bash
set -euo pipefail
# activa tu venv si aplica
# source /home/juan-dev/bladeRF/host/python/.venv/bin/activate

# si estás en Flatpak asegúrate que BLADERF_CLI apunte al host (basename aceptable)
# export BLADERF_CLI=/usr/local/bin/bladeRF-cli

# arranca uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

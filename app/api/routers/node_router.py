# app/pipeline_router.py
from __future__ import annotations

import sys
import io
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, Body
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

import asyncio
import time
import threading
import math
import numpy as np
import serial

from app.util.node_positition import capture_avg_fix

router = APIRouter(prefix="/node", tags=["node"])

from app.util.node_positition import (
    capture_avg_fix,
    DEFAULT_N_SAMPLES as GPS_DEFAULT_N_SAMPLES,
    DEFAULT_PORT as GPS_PORT
)


@router.get(
    "/gps/capture",
    summary="Captura posición GNSS usando los valores por defecto",
)
async def capture_gps_position(
n_samples: int = Query(
        GPS_DEFAULT_N_SAMPLES,
        ge=1,
        le=50,
        description="Número de muestras a promediar",
    ),

port: str = Query(
    GPS_PORT,
    description="puerto a especificar para cada nodo"
)
):
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: capture_avg_fix(n_samples=n_samples, port=port),
        )
        return result

    except TimeoutError as e:

        raise HTTPException(status_code=504, detail=str(e))

    except serial.SerialException as e:

        raise HTTPException(status_code=500, detail=f"Error de puerto serie: {e}")

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
    DEFAULT_PORT as GPS_PORT,
    detect_gps_ports
)


@router.get("/gps/capture",summary="Captura posición GNSS usando los valores por defecto")
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


@router.get("/gps/all/capture/{n_samples}",)
async def capture_all_gps_positions(
    n_samples: int = Path(
        ge=1,
        le=50,
        description="Número de muestras a promediar por cada dispositivo",
    )
):
    loop = asyncio.get_running_loop()

    try:
        gps_ports = await loop.run_in_executor(
            None,
            detect_gps_ports,  # usa baud por defecto
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error detectando dispositivos GNSS: {e}",
        )

    if not gps_ports:
        raise HTTPException(
            status_code=404,
            detail="No se encontró ningún receptor GNSS conectado.",
        )

    # 2) Lanzar captura en paralelo para cada puerto
    tasks = [
        loop.run_in_executor(
            None,
            # 👇 importante usar p=port para evitar el típico bug de closures
            (lambda p=port: capture_avg_fix(port=p, n_samples=n_samples)),
        )
        for port in gps_ports
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3) Armar respuesta: una entrada por dispositivo
    devices_response = []
    for port, result in zip(gps_ports, results):
        if isinstance(result, TimeoutError):
            devices_response.append(
                {
                    "port": port,
                    "ok": False,
                    "error_type": "timeout",
                    "error": str(result),
                }
            )
        elif isinstance(result, serial.SerialException):
            devices_response.append(
                {
                    "port": port,
                    "ok": False,
                    "error_type": "serial",
                    "error": str(result),
                }
            )
        elif isinstance(result, Exception):
            devices_response.append(
                {
                    "port": port,
                    "ok": False,
                    "error_type": "unknown",
                    "error": str(result),
                }
            )
        else:
            # `result` es el dict que devuelve capture_avg_fix
            devices_response.append(
                {
                    "port": port,
                    "ok": True,
                    "data": result,
                }
            )

    return {
        "count": len(gps_ports),
        "devices": devices_response,
    }

@router.get("/gps/avg/capture/{n_samples}")
async def capture_global_avg_gps_position(
    n_samples: int = Path(
        ge=1,
        le=50,
        description="Número de muestras a promediar por cada dispositivo",
    )
):
    """
    ✅ NUEVO: Devuelve UNA sola posición (lat/lon) promedio global
    - Promedia el avg de cada GPS OK
    - Pondera por n_samples (más muestras = más peso)
    """
    loop = asyncio.get_running_loop()

    try:
        gps_ports = await loop.run_in_executor(None, detect_gps_ports)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detectando dispositivos GNSS: {e}")

    if not gps_ports:
        raise HTTPException(status_code=404, detail="No se encontró ningún receptor GNSS conectado.")

    tasks = [
        loop.run_in_executor(None, (lambda p=port: capture_avg_fix(port=p, n_samples=n_samples)))
        for port in gps_ports
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ok_items = []
    failed = []

    for port, result in zip(gps_ports, results):
        if isinstance(result, Exception):
            failed.append({"port": port, "error": str(result)})
            continue

        # result: {"avg":{"lat":..,"lon":..}, "n_samples": N}
        avg = (result or {}).get("avg") or {}
        lat = avg.get("lat")
        lon = avg.get("lon")
        ns = int((result or {}).get("n_samples") or 0)

        if lat is None or lon is None or ns <= 0:
            failed.append({"port": port, "error": "Respuesta inválida (sin avg o sin n_samples)"})
            continue

        ok_items.append({"port": port, "lat": float(lat), "lon": float(lon), "n_samples": ns})

    if not ok_items:
        raise HTTPException(
            status_code=500,
            detail={"message": "No se pudo obtener fix válido de ningún GPS", "failed": failed},
        )

    # ✅ Promedio ponderado por n_samples
    total_samples = sum(x["n_samples"] for x in ok_items)
    lat_avg = sum(x["lat"] * x["n_samples"] for x in ok_items) / total_samples
    lon_avg = sum(x["lon"] * x["n_samples"] for x in ok_items) / total_samples

    return {
        "ok": True,
        "avg": {"lat": lat_avg, "lon": lon_avg},
        "n_devices_total": len(gps_ports),
        "n_devices_ok": len(ok_items),
        "n_samples_total": total_samples,
        "devices_ok": [{"port": x["port"], "n_samples": x["n_samples"]} for x in ok_items],
        "devices_failed": failed,
    }

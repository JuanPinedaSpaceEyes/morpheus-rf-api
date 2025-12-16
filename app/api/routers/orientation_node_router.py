# app/routers/spectrum_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math

router = APIRouter(prefix="/orientation", tags=["orientation"])


# Modelo de respuesta
class OrientationResponse(BaseModel):
    heading: float
    status: str
    message: str


# ThreadPool para ejecutar código bloqueante (comunicación serial)
executor = ThreadPoolExecutor(max_workers=2)


@router.get("/", response_model=OrientationResponse)
async def get_sensor_orientation():
    """
    Obtiene el heading del sensor BNO055.

    Llama a get_orientation() que:
    - Lee 5 muestras del sensor
    - Calcula la media circular
    - Retorna h_avg (heading promedio)
    """
    try:
        # Importar la función
        from app.util.orientation import get_orientation

        # Ejecutar en thread separado (serial.Serial es bloqueante)
        loop = asyncio.get_event_loop()
        h_avg = await loop.run_in_executor(executor, get_orientation)

        # Validar resultado
        if h_avg is None or math.isnan(h_avg):
            raise HTTPException(
                status_code=500,
                detail="No se pudo obtener lectura válida del sensor"
            )

        return OrientationResponse(
            heading=round(h_avg, 2),
            status="success",
            message="Heading obtenido correctamente"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer sensor: {str(e)}"
        )
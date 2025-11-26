from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.util.dual_bladerf_capture import DualBladeRFService

router = APIRouter(prefix="/rfCaptureDualBladerf", tags=["rfCaptureDualBladerf"])


class ConfigBladeRF(BaseModel):
    serial_one: str
    serial_two: str
    center_freq_bladeOne: float
    center_freq_bladeTwo: float
    gain_db: int

dual_bladerf_service: Optional[DualBladeRFService] = None


@router.post("/start")
async def start_capture(initConfig: ConfigBladeRF):
    global dual_bladerf_service


    if dual_bladerf_service is None:

        dual_bladerf_service = DualBladeRFService(
            serial_24=initConfig.serial_one,
            serial_58=initConfig.serial_two,
            center_freq_24=initConfig.center_freq_bladeOne * 1e9,
            center_freq_58=initConfig.center_freq_bladeTwo * 1e9,
            sample_rate=10e6,
            gain_db=initConfig.gain_db,
        )


        try:
            dual_bladerf_service.init_devices()
        except RuntimeError as e:
            #
            dual_bladerf_service = None
            raise HTTPException(status_code=500, detail=str(e))

    try:
        dual_bladerf_service.start_capture()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "started",
        "running": dual_bladerf_service.running,
    }


@router.post("/stop")
async def stop_capture():
    global dual_bladerf_service

    if dual_bladerf_service is None:
        raise HTTPException(status_code=400, detail="Service is not initialized.")

    dual_bladerf_service.stop_capture()

    return {
        "status": "stopped",
        "running": dual_bladerf_service.running,
    }

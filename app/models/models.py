# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Peak(BaseModel):
    frequency_hz: float = Field(..., description="Frecuencia absoluta del pico (Hz)")
    power_db: float = Field(..., description="Potencia estimada (dB)")

class PeaksBlock(BaseModel):
    block_id: int
    capture_time_sec: float
    noise_floor_db: float
    max_power_db: float
    peaks: List[Peak]

class Status(BaseModel):
    configured: bool
    blocks_processed: int
    sample_rate_hz: float
    center_freq_hz: float
    last_error: Optional[str] = None

class PowerStatus(BaseModel):
    powered: bool
    method: str
    message: str

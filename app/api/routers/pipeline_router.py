# app/pipeline_router.py
from __future__ import annotations

import io
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, Body
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import asyncio
import time
import threading
import math
import numpy as np


router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# ---------- Config ----------
PIPELINE_PATH = Path(__file__).resolve().parents[2] / "ml" / "pipeline.py"
LOG_DIR = Path(os.getenv("PIPELINE_LOG_DIR", "/tmp/morpheus_pipeline"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
CWD = Path(os.getenv("PIPELINE_CWD", str(PIPELINE_PATH.parent)))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")  # opcional: fuerza intérprete

# Imagen "clásica" por compatibilidad (si tu pipeline guarda last.png)
LAST_IMG_PATH = Path(os.getenv("PIPELINE_LAST_IMG", "/tmp/morpheus_pipeline/last.png"))

# ---------- Estado ----------
class _State:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.started_at: Optional[float] = None
        self.log_file: Optional[Path] = None
        self.lock = threading.Lock()

state = _State()

# ---------- Helpers ----------
def _start_pipeline(args: Optional[List[str]] = None):
    if not PIPELINE_PATH.exists():
        raise HTTPException(500, f"pipeline.py no existe en {PIPELINE_PATH}")

    with state.lock:
        if state.proc and state.proc.poll() is None:
            raise HTTPException(409, "El pipeline ya está corriendo.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"pipeline_{ts}.log"
        log_f = open(log_path, "a", buffering=1)

        cmd = [PYTHON_BIN, "-u", str(PIPELINE_PATH)]
        if args:
            cmd += args

        env = os.environ.copy()

        proc = subprocess.Popen(
            cmd,
            cwd=str(CWD),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        state.proc = proc
        state.started_at = time.time()
        state.log_file = log_path

        return {
            "status": "started",
            "pid": proc.pid,
            "log_file": str(log_path),
            "cwd": str(CWD),
            "cmd": cmd,
        }

def _status():
    with state.lock:
        running = state.proc is not None and state.proc.poll() is None
        rc = None if not state.proc else state.proc.poll()
        return {
            "running": running,
            "pid": None if not state.proc else state.proc.pid,
            "returncode": rc,
            "uptime_sec": None if not running else int(time.time() - (state.started_at or time.time())),
            "log_file": None if not state.log_file else str(state.log_file),
            "started_at": state.started_at,
            "pipeline_path": str(PIPELINE_PATH),
        }

def _tail_log(path: Path, kb: int) -> str:
    if not path.exists():
        return ""
    n = kb * 1024
    with open(path, "rb") as f:
        try:
            f.seek(-n, io.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode(errors="replace")
    return data

def _stop(timeout: float = 5.0):
    with state.lock:
        if not state.proc or state.proc.poll() is not None:
            return {"status": "not_running"}

        state.proc.terminate()
        try:
            state.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            state.proc.kill()

        rc = state.proc.poll()
        pid = state.proc.pid
        state.proc = None
        return {"status": "stopped", "pid": pid, "returncode": rc}

class PsdFrameModel(BaseModel):
    start_hz: float = Field(..., description="Frecuencia de inicio del primer bin (Hz, absoluta)")
    bin_hz: float = Field(..., description="Ancho de cada bin en Hz")
    bins: List[float] = Field(..., description="Potencia por bin en dB")
    capture_time_sec: Optional[float] = Field(None, description="Epoch seconds (float)")
    drone_id: Optional[str] = None
    fft_size: Optional[int] = None
    hop_samples: Optional[int] = None
    ds: Optional[int] = None
    schema_version: Optional[str] = None

# ====== PREDICCIONES (hub + endpoints) =======================================
class PredResultModel(BaseModel):
    label: str = Field(..., description="Nombre de la clase predicha")

class _PredHub:
    def __init__(self):
        self._last: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def set_last(self, pred: Dict[str, Any]) -> None:
        with self._lock:
            self._last = pred

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last) if self._last is not None else None

pred_hub = _PredHub()

def _topk(pred: Dict[str, Any], k: int) -> Dict[str, Any]:
    """Adjunta 'topk' si hay 'probs' disponible."""
    probs = pred.get("probs")
    classes = pred.get("classes") or []
    if not isinstance(probs, list) or not probs:
        return pred
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    pred = dict(pred)  # copia superficial
    pred["topk"] = [
        {"id": i, "label": classes[i] if i < len(classes) else str(i), "prob": float(probs[i])}
        for i in order
    ]
    return pred

class _PsdHub:
    def __init__(self):
        self._last: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def set_last(self, frame: Dict[str, Any]) -> None:
        """Guardar el último frame (puede llamarse desde cualquier hilo)."""
        with self._lock:
            self._last = frame

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            # Devolver una copia superficial por seguridad
            return dict(self._last) if self._last is not None else None


psd_hub = _PsdHub()

def _slice_frame_by_hz(frame: Dict[str, Any], min_hz: Optional[float], max_hz: Optional[float]) -> Dict[str, Any]:
    if min_hz is None and max_hz is None:
        return frame

    start_hz = float(frame["start_hz"])
    bin_hz = float(frame["bin_hz"])
    bins = frame["bins"]
    n = len(bins)

    lo_idx = 0 if min_hz is None else int(math.floor((min_hz - start_hz) / bin_hz))
    hi_idx = n if max_hz is None else int(math.ceil((max_hz - start_hz) / bin_hz))

    lo_idx = max(0, lo_idx)
    hi_idx = min(n, hi_idx)
    if hi_idx <= lo_idx:
        # No overlap → devolver frame vacío mínimo coherente
        return {
            **frame,
            "bins": [],
            "fft_size": 0,
        }

    sliced = dict(frame)
    sliced["bins"] = bins[lo_idx:hi_idx]
    sliced["start_hz"] = start_hz + lo_idx * bin_hz
    sliced["fft_size"] = len(sliced["bins"])
    return sliced

def _compute_psd_db(x: np.ndarray, nfft: int = 4096, window: str = "hann") -> np.ndarray:
    """
    x: array complejo 1D (IQ, baseband) np.complex64/complex128
    Retorna PSD en dB (log |FFT|), length=nfft, centrada (fftshift).
    """
    if x.ndim != 1:
        raise ValueError("x debe ser 1D complejo")

    # Seleccionar segmento
    seg = x[:nfft]
    if seg.shape[0] < nfft:
        seg = np.pad(seg, (0, nfft - seg.shape[0]))

    # Ventana
    if window == "hann":
        w = np.hanning(nfft)
    elif window == "hamming":
        w = np.hamming(nfft)
    else:
        w = np.ones(nfft)

    seg = seg * w
    X = np.fft.fftshift(np.fft.fft(seg, n=nfft))
    psd = 20.0 * np.log10(np.abs(X) + 1e-12)
    return psd

def make_psd_frame(
    x: np.ndarray,
    center_hz: float,
    sample_rate: float,
    nfft: int = 4096,
    *,
    drone_id: Optional[str] = None,
    hop_samples: Optional[int] = None,
    ds: Optional[int] = None,
    schema_version: str = "1.0",
) -> Dict[str, Any]:

    psd_db = _compute_psd_db(x=x, nfft=nfft, window="hann")
    start_hz = float(center_hz - sample_rate / 2.0)
    bin_hz = float(sample_rate / nfft)
    frame: Dict[str, Any] = dict(
        start_hz=start_hz,
        bin_hz=bin_hz,
        bins=psd_db.astype(float).tolist(),
        capture_time_sec=time.time(),
        drone_id=drone_id,
        fft_size=nfft,
        hop_samples=hop_samples,
        ds=ds,
        schema_version=schema_version,
    )
    return frame


# ---------- Endpoints ----------
@router.get("/run", tags=["pipeline"])
def run_get():
    return _start_pipeline(args=None)

@router.get("/status", tags=["pipeline"])
def status_get():
    return _status()

@router.get("/logs", tags=["pipeline"], response_class=PlainTextResponse)
def logs_get(tail_kb: int = Query(1024, ge=1, le=1024)):
    if not state.log_file:
        return PlainTextResponse("")
    return PlainTextResponse(_tail_log(state.log_file, tail_kb))

@router.post("/stop", tags=["pipeline"])
def stop_post(timeout_sec: float = Query(5.0, ge=1.0, le=60.0)):
    return _stop(timeout=timeout_sec)

@router.websocket("/ws/psd")
async def ws_psd(
    ws: WebSocket,
    interval_ms: int = Query(200, ge=50, le=5000, description="Período de envío (ms)"),
    min_hz: Optional[float] = Query(None, description="Recorte inferior (Hz)"),
    max_hz: Optional[float] = Query(None, description="Recorte superior (Hz)"),
):
    await ws.accept()
    try:
        last_sent_ts = 0.0
        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            frame = psd_hub.get_last()
            if not frame:
                # Aún no hay datos: sigue esperando.
                continue

            # Para evitar mandar el mismo frame demasiadas veces, dejamos esto opcional.
            # Si tu frame tiene capture_time_sec, lo usamos de “id temporal”.
            cap_ts = float(frame.get("capture_time_sec") or 0.0)
            if cap_ts and cap_ts <= last_sent_ts:
                continue

            to_send = _slice_frame_by_hz(frame, min_hz, max_hz)
            await ws.send_json(to_send)
            last_sent_ts = cap_ts if cap_ts else time.time()
    except Exception:
        # Cierra silenciosamente: el cliente puede desconectarse.
        return

@router.get("/psd/latest", response_model=PsdFrameModel)
def get_latest_psd(
    min_hz: Optional[float] = Query(None, description="Opcional: frecuencia mínima para recorte (Hz)"),
    max_hz: Optional[float] = Query(None, description="Opcional: frecuencia máxima para recorte (Hz)"),
):
    """
    Devuelve el último PsdFrame disponible (producido por el pipeline vía psd_hub.set_last()).
    """
    last = psd_hub.get_last()
    if not last:
        raise HTTPException(status_code=404, detail="No hay PSD aún (psd_hub vacío).")
    return _slice_frame_by_hz(last, min_hz, max_hz)

@router.get("/psd/snapshot", response_model=PsdFrameModel)
def snapshot_psd(
    center_hz: float = Query(..., description="Frecuencia central absoluta (Hz)"),
    sample_rate: float = Query(..., description="Frecuencia de muestreo (Hz)"),
    nfft: int = Query(4096, ge=256, le=65536),
):
    """
    Utilidad de test: genera un PSD sintético (sin leer SDR) con picos para validar front.
    NO usa el hub. Útil para probar el canvas.
    """
    # Señal sintética: dos tonos + ruido (baseband)
    t = np.arange(nfft) / sample_rate
    sig = (
        0.8 * np.exp(1j * 2 * np.pi * ( + 0.12 * sample_rate) * t) +   # pico desplazado
        0.6 * np.exp(1j * 2 * np.pi * ( - 0.18 * sample_rate) * t)     # otro pico
    )
    noise = (np.random.randn(nfft) + 1j * np.random.randn(nfft)) * 0.25
    x = (sig + noise).astype(np.complex64)

    frame = make_psd_frame(x=x, center_hz=center_hz, sample_rate=sample_rate, nfft=nfft, schema_version="1.0")
    return frame

@router.post("/psd/ingest", status_code=202)
def ingest_psd(frame: PsdFrameModel = Body(...)):

    psd_hub.set_last(frame.dict())
    return {"ok": True}

@router.websocket("/ws/pred")
async def ws_pred(
    ws: WebSocket,
    interval_ms: int = Query(200, ge=50, le=5000),
    topk: int = Query(3, ge=1, le=10),
):
    await ws.accept()
    try:
        last_sent_ts = 0.0
        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            pred = pred_hub.get_last()
            if not pred:
                continue
            ts = float(pred.get("capture_time_sec") or 0.0)
            if ts and ts <= last_sent_ts:
                continue
            await ws.send_json(_topk(pred, topk))
            last_sent_ts = ts if ts else time.time()
    except Exception:
        return

@router.get("/pred/latest", response_model=PredResultModel)
def get_latest_pred():
    pred = pred_hub.get_last()
    if not pred:
        raise HTTPException(status_code=404, detail="No hay predicciones aún (pred_hub vacío).")
    return pred

@router.post("/pred/ingest", status_code=202)
def ingest_pred(pred: PredResultModel = Body(...)):
    pred_hub.set_last(pred.dict())
    return {"ok": True}

@router.get("/pred/mock")
def pred_mock():
    # Ejemplo sintético
    classes = ["DJI Mini 4K", "Jammer", "Noise"]
    probs = np.array([0.15, 0.70, 0.15], dtype=np.float32).tolist()
    pred = {
        "capture_time_sec": time.time(),
        "label_id": 1,
        "label": classes[1],
        "confidence": probs[1],
        "probs": probs,
        "classes": classes,
        "meta": {"note": "mock"},
    }
    pred_hub.set_last(pred)
    return pred
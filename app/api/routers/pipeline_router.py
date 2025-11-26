from __future__ import annotations

import sys
import io
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import time
import threading
import math

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, Body
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# 🔹 Importamos la función que corre el pipeline en este mismo proceso
from app.ml.pipeline import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# ---------- Config ----------
PIPELINE_PATH = Path(__file__).resolve().parents[2] / "ml" / "pipeline.py"
LOG_DIR = Path(os.getenv("PIPELINE_LOG_DIR", "/tmp/morpheus_pipeline"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
CWD = Path(os.getenv("PIPELINE_CWD", str(PIPELINE_PATH.parent)))
LAST_SPEC_PATH = Path(os.getenv("PIPELINE_LAST_SPEC", "/tmp/morpheus_pipeline/last_spectrogram.png"))
LAST_DOA_PATH = Path(os.getenv("PIPELINE_LAST_DOA", "/tmp/morpheus_pipeline/last_doa.png"))
LAST_IMG_PATH = Path(os.getenv("PIPELINE_LAST_IMG", "/tmp/morpheus_pipeline/last.png"))

# ---------- Estado: modo single-thread (legacy compatible) ----------
class _State:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.started_at: Optional[float] = None
        self.log_file: Optional[Path] = None  # si algún día redirigimos stdout
        self.lock = threading.Lock()


state = _State()


# ---------- Estado: multi-thread (1 pipeline por bladeRF) ----------
@dataclass
class PipelineThreadInfo:
    name: str
    dev_id: Optional[str]
    device_name: str
    thread: threading.Thread
    stop_event: threading.Event
    started_at: float
    log_file: Optional[Path] = None  # reservado por si quieres logs por blade


class _MultiState:
    def __init__(self):
        self.threads: Dict[str, PipelineThreadInfo] = {}
        self.lock = threading.Lock()


multi_state = _MultiState()


# ---------- Modelos de entrada ----------
class RunDeviceModel(BaseModel):
    name: str = Field(..., description="ID interno: '24', '58', 'blade_1', etc.")
    dev_id: Optional[str] = Field(
        None,
        description=(
            "Valor para BLADERF_DEVICE (serial/ID del bladeRF). "
            "Si es None, pipeline.py elegirá el primero disponible."
        ),
    )


# ---------- Helper para logs ----------
def _tail_log(path: Path, kb: int) -> str:
    if not path or not path.exists():
        return ""
    n = kb * 1024
    with open(path, "rb") as f:
        try:
            f.seek(-n, io.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode(errors="replace")
    return data


# ---------- Modo single-thread (único pipeline) ----------
def _start_pipeline(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Inicia un único pipeline en un hilo.
    El parámetro args se mantiene por compatibilidad pero no se usa.
    """
    if not PIPELINE_PATH.exists():
        raise HTTPException(500, f"pipeline.py no existe en {PIPELINE_PATH}")

    with state.lock:
        if state.thread and state.thread.is_alive():
            raise HTTPException(409, "El pipeline ya está corriendo.")

        stop_event = threading.Event()
        t = threading.Thread(
            target=run_pipeline,
            kwargs={"dev_id": None, "device_name": None, "stop_event": stop_event},
            daemon=True,
            name="pipeline_main",
        )
        state.thread = t
        state.stop_event = stop_event
        state.started_at = time.time()
        state.log_file = None  # si luego quieres redirigir stdout, se puede usar
        t.start()

        return {
            "status": "started",
            "thread_name": t.name,
            "log_file": None,
            "cwd": str(CWD),
            "cmd": ["thread", "run_pipeline"],
        }


def _status() -> Dict[str, Any]:
    with state.lock:
        running = state.thread is not None and state.thread.is_alive()
        return {
            "running": running,
            "thread_name": state.thread.name if state.thread else None,
            "returncode": None,
            "uptime_sec": None
            if not running
            else int(time.time() - (state.started_at or time.time())),
            "log_file": str(state.log_file) if state.log_file else None,
            "started_at": state.started_at,
            "pipeline_path": str(PIPELINE_PATH),
        }


def _stop(timeout: float = 5.0) -> Dict[str, Any]:
    with state.lock:
        t = state.thread
        stop_event = state.stop_event
        if not t or not t.is_alive():
            return {"status": "not_running"}

    # Pedimos al hilo que pare
    if stop_event:
        stop_event.set()

    t.join(timeout=timeout)
    still_running = t.is_alive()

    with state.lock:
        if not still_running:
            state.thread = None
            state.stop_event = None

    return {
        "status": "stopped" if not still_running else "still_running",
        "thread_name": t.name,
        "returncode": None,
    }


# ---------- Modo multi-thread: 1 pipeline por bladeRF ----------
def _start_pipeline_for(name: str, dev_id: Optional[str]) -> Dict[str, Any]:
    if not name:
        raise HTTPException(400, "name no puede ser vacío")

    with multi_state.lock:
        existing = multi_state.threads.get(name)
        if existing and existing.thread.is_alive():
            raise HTTPException(
                409,
                f"El pipeline '{name}' ya está corriendo (thread={existing.thread.name}).",
            )

        stop_event = threading.Event()
        t = threading.Thread(
            target=run_pipeline,
            kwargs={"dev_id": dev_id, "device_name": name, "stop_event": stop_event},
            daemon=True,
            name=f"pipeline_{name}",
        )
        started_at = time.time()
        info = PipelineThreadInfo(
            name=name,
            dev_id=dev_id,
            device_name=name,
            thread=t,
            stop_event=stop_event,
            started_at=started_at,
            log_file=None,
        )
        multi_state.threads[name] = info
        t.start()

        return {
            "status": "started",
            "name": name,
            "dev_id": dev_id,
            "thread_name": t.name,
            "log_file": None,
            "cwd": str(CWD),
        }


def _status_all() -> Dict[str, Any]:
    with multi_state.lock:
        out: Dict[str, Any] = {}
        for name, info in multi_state.threads.items():
            running = info.thread.is_alive()
            out[name] = {
                "running": running,
                "thread_name": info.thread.name,
                "returncode": None,
                "dev_id": info.dev_id,
                "uptime_sec": None
                if not running
                else int(time.time() - info.started_at),
                "log_file": str(info.log_file) if info.log_file else None,
            }
        return out


def _stop_one(name: str, timeout: float = 5.0) -> Dict[str, Any]:
    with multi_state.lock:
        info = multi_state.threads.get(name)
        if not info or not info.thread.is_alive():
            return {"status": "not_running", "name": name}
        t = info.thread
        stop_event = info.stop_event

    if stop_event:
        stop_event.set()
    t.join(timeout=timeout)
    still_running = t.is_alive()

    with multi_state.lock:
        if not still_running:
            multi_state.threads.pop(name, None)

    return {
        "status": "stopped" if not still_running else "still_running",
        "name": name,
        "thread_name": t.name,
        "returncode": None,
    }


def _stop_all(timeout: float = 5.0) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    with multi_state.lock:
        names = list(multi_state.threads.keys())
    for name in names:
        results[name] = _stop_one(name=name, timeout=timeout)
    return results


# ================== MODELOS Y HUBS DE PSD / PRED / DOA / SPEC ==================

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


class DoaResultModel(BaseModel):
    angle_deg: float = Field(..., description="Ángulo estimado (grados, [-90,90])")


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


class _DoaHub:
    def __init__(self):
        self._last: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def set_last(self, doa: Dict[str, Any]) -> None:
        with self._lock:
            self._last = doa

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last) if self._last is not None else None


doa_hub = _DoaHub()


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


class _SpecHub:
    def __init__(self):
        self._last: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def set_last(self, spec: Dict[str, Any]) -> None:
        with self._lock:
            self._last = spec

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last) if self._last is not None else None


spec_hub = _SpecHub()


class SpecFrameModel(BaseModel):
    label: str
    timestamp: float
    shape: List[int]
    n_fft: int
    hop_length: int
    sample_rate: int
    pmin: float
    pmax: float
    data: List[List[List[float]]]


# ============================ ENDPOINTS CONTROL PIPELINE ============================

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

@router.post("/run-device", tags=["pipeline"])
def run_device(body: RunDeviceModel):
    return _start_pipeline_for(
        name=body.name,
        dev_id=body.dev_id,
    )

@router.get("/status-all", tags=["pipeline"])
def status_all():
    return _status_all()


@router.post("/stop-device", tags=["pipeline"])
def stop_device(
        name: str = Query(..., description="ID interno del pipeline (ej: '24', '58')"),
        timeout_sec: float = Query(5.0, ge=1.0, le=60.0),
):
    return _stop_one(name=name, timeout=timeout_sec)


@router.post("/stop-all", tags=["pipeline"])
def stop_all(
    timeout_sec: float = Query(5.0, ge=1.0, le=60.0),
):
    return _stop_all(timeout=timeout_sec)

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


# ============================ ENDPOINTS PRED ============================

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


# ============================ ENDPOINTS DOA ============================

@router.websocket("/ws/doa")
async def ws_doa(
    ws: WebSocket,
    interval_ms: int = Query(200, ge=50, le=5000, description="Período de envío (ms)")
):
    await ws.accept()
    try:
        last_sent_ts = 0.0
        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            doa = doa_hub.get_last()
            if not doa:
                continue
            ts = float(doa.get("capture_time_sec") or 0.0)
            if ts and ts <= last_sent_ts:
                continue
            await ws.send_json(doa)
            last_sent_ts = ts if ts else time.time()
    except Exception:
        return


@router.get("/doa/latest", response_model=DoaResultModel)
def get_latest_doa():
    doa = doa_hub.get_last()
    if not doa:
        raise HTTPException(status_code=404, detail="No hay DOA aún (doa_hub vacío).")
    return doa


@router.post("/doa/ingest", status_code=202)
def ingest_doa(doa: DoaResultModel = Body(...)):
    doa_hub.set_last(doa.dict())
    return {"ok": True}


# ============================ ENDPOINTS SPEC ============================

@router.post("/spec/ingest", status_code=202)
def ingest_spec(frame: SpecFrameModel = Body(...)):
    spec_hub.set_last(frame.model_dump())
    return {"status": "ok"}


@router.websocket("/ws/spec")
async def ws_spec(
    ws: WebSocket,
    interval_ms: int = Query(200, ge=50, le=5000, description="Período de envío (ms)")
):
    await ws.accept()
    try:
        last_sent_ts = 0.0
        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            frame = spec_hub.get_last()
            if not frame:
                continue

            ts = float(frame.get("timestamp") or 0.0)
            if ts and ts <= last_sent_ts:
                continue

            await ws.send_json(frame)
            last_sent_ts = ts if ts else time.time()
    except Exception:
        return

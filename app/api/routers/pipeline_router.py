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
from bladerf import _bladerf

# 🔹 Importamos la función que corre el pipeline en este mismo proceso
from app.ml.pipeline import run_pipeline

# ✅ GPS detect
from app.util.node_positition import detect_gps_ports

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# ---------- Config ----------
PIPELINE_PATH = Path(__file__).resolve().parents[2] / "ml" / "pipeline.py"
LOG_DIR = Path(os.getenv("PIPELINE_LOG_DIR", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/logs/pipeline"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / os.getenv("PIPELINE_LOG_FILE", "pipeline.log")
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
    gps_port: Optional[str] = None  # ✅ NUEVO: puerto GPS asignado a este blade
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
    gps_port: Optional[str] = Field(
        None,
        description="Puerto serial GPS a asignar a este pipeline (ej: /dev/ttyUSB0 o /dev/cu.PL2303...).",
    )


# ---------- Helpers para detectar todos los bladeRF conectados ----------

def _devinfo_serial_str(info) -> str:

    s = getattr(info, "serial", "")
    if isinstance(s, (bytes, bytearray)):
        return s.decode("ascii", errors="ignore")
    return str(s)


def _detect_bladerf_devices() -> list[dict]:
    if _bladerf is None:
        raise HTTPException(
            status_code=500,
            detail="El módulo bladerf no está disponible en este backend.",
        )

    try:
        devinfos = _bladerf.get_device_list()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error llamando a bladerf.get_device_list(): {e}",
        )

    devices = []
    if not devinfos:
        return devices

    for info in devinfos:
        serial = _devinfo_serial_str(info)
        devices.append(
            {
                "serial": serial,
                "info": repr(info),  # por si quieres ver bus/instancia/etc.
            }
        )
    return devices


# ---------- Helpers GPS ----------
def _detect_gps_ports_or_empty(
    *,
    baud: int = 4800,
    sniff_time_s: float = 3.0,
    read_timeout_s: float = 0.5,
) -> List[str]:
    try:
        ports = detect_gps_ports(baud=baud, sniff_time_s=sniff_time_s, read_timeout_s=read_timeout_s)
        return ports or []
    except Exception as e:
        # No tumbar el endpoint por esto; dejamos el error explícito arriba si se requiere
        print("[GPS] Error detectando puertos:", e)
        return []


def _assign_gps_ports_to_blades(blade_serials: List[str], gps_ports: List[str]) -> Dict[str, str]:
    """
    Asigna 1 GPS por blade, de manera determinística (ordenando).
    Retorna: serial_blade -> gps_port
    """
    blade_serials_sorted = sorted([s for s in blade_serials if s])
    gps_ports_sorted = sorted(gps_ports)

    if len(gps_ports_sorted) < len(blade_serials_sorted):
        raise HTTPException(
            status_code=409,
            detail=f"No hay suficientes GPS para enlazar 1:1. blades={len(blade_serials_sorted)} gps={len(gps_ports_sorted)}",
        )

    return {blade_serials_sorted[i]: gps_ports_sorted[i] for i in range(len(blade_serials_sorted))}


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

class _NodeMap:
    def __init__(self):
        self.by_serial: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

node_map = _NodeMap()


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
        state.log_file = None
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
            "uptime_sec": None if not running else int(time.time() - (state.started_at or time.time())),
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
def _start_pipeline_for(
    name: str,
    dev_id: Optional[str],
    *,
    assign_gps: bool = False,
    gps_port: Optional[str] = None,
    gps_baud: int = 4800,
    node_port: Optional[str] = None,
    node_side: Optional[str] = None,
) -> Dict[str, Any]:
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

        # ✅ kwargs seguros: SOLO lo que run_pipeline acepta
        kwargs: Dict[str, Any] = {
            "dev_id": dev_id,
            "device_name": name,
            "stop_event": stop_event,
        }
        if node_port is not None:
            kwargs["node_port"] = node_port
        if node_side is not None:
            kwargs["node_side"] = node_side

        # ✅ SOLO si assign_gps=True
        if assign_gps and gps_port:
            kwargs["gps_port"] = gps_port
            kwargs["gps_baud"] = int(gps_baud)

        t = threading.Thread(
            target=run_pipeline,
            kwargs=kwargs,
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
            gps_port=(gps_port if (assign_gps and gps_port) else None),  # ✅ FIX
            log_file=None,
        )
        multi_state.threads[name] = info
        t.start()

        return {
            "status": "started",
            "name": name,
            "dev_id": dev_id,
            "thread_name": t.name,
            "cwd": str(CWD),
            "assign_gps": assign_gps,
            "gps_port": gps_port if assign_gps else None,
            "gps_baud": int(gps_baud) if assign_gps else None,
            "node_port": node_port,
            "node_side": node_side,
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
                "gps_port": info.gps_port,  # ✅ NUEVO
                "uptime_sec": None if not running else int(time.time() - info.started_at),
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
    angle_deg: float
    blade_serial: Optional[str] = None
    center_freq_hz: Optional[float] = None
    psr_db: Optional[float] = None
    capture_time_sec: Optional[float] = None
    node_side: Optional[str] = None  # "left" | "right"
    node_name: Optional[str] = None  # "blade_1", etc.


class PredResultModel(BaseModel):
    label: str = Field(..., description="Nombre de la clase predicha")
    confidence: Optional[float] = Field(None, description="Confianza principal asociada a la detección (0-1, opcional)")
    timestamp: Optional[float] = Field(None, description="Tiempo de la predicción en segundos desde epoch (opcional)")
    prediction_mode: Optional[str] = Field(None, description="Modo de predicción ('two_stage' o 'single_stage')")
    center_freq: Optional[float] = Field(None, description="Frecuencia central (Hz) usada en la captura que generó esta predicción")
    mode: Optional[str] = Field(None, description="Modo del pipeline durante esta predicción: 'scan' o 'track'")
    binary: Optional[Dict[str, Any]] = Field(None, description="Detalle del clasificador binario (si aplica)")
    multiclass: Optional[Dict[str, Any]] = Field(None, description="Detalle del clasificador multiclase (si aplica)")


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
    probs = pred.get("probs")
    classes = pred.get("classes") or []
    if not isinstance(probs, list) or not probs:
        return pred
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    pred = dict(pred)
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
        with self._lock:
            self._last = frame

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last) if self._last is not None else None


psd_hub = _PsdHub()


# ✅ DOA HUB: último global + último POR blade_serial
class _DoaHub:
    def __init__(self):
        self._last: Optional[Dict[str, Any]] = None
        self._last_by_blade: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def set_last(self, doa: Dict[str, Any]) -> None:
        with self._lock:
            self._last = doa

    def set_last_for(self, blade_serial: str, doa: Dict[str, Any]) -> None:
        if not blade_serial:
            return
        with self._lock:
            self._last_by_blade[str(blade_serial)] = doa

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last) if self._last is not None else None

    def get_last_for(self, blade_serial: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            doa = self._last_by_blade.get(str(blade_serial))
            return dict(doa) if doa is not None else None

    def get_last_all(self) -> Dict[str, Any]:
        with self._lock:
            return {k: dict(v) for k, v in self._last_by_blade.items()}


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
        return {**frame, "bins": [], "fft_size": 0}

    sliced = dict(frame)
    sliced["bins"] = bins[lo_idx:hi_idx]
    sliced["start_hz"] = start_hz + lo_idx * bin_hz
    sliced["fft_size"] = len(sliced["bins"])
    return sliced


def _compute_psd_db(x: np.ndarray, nfft: int = 4096, window: str = "hann") -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("x debe ser 1D complejo")

    seg = x[:nfft]
    if seg.shape[0] < nfft:
        seg = np.pad(seg, (0, nfft - seg.shape[0]))

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
    if not LOG_FILE.exists():
        return PlainTextResponse("")
    return PlainTextResponse(_tail_log(LOG_FILE, tail_kb))

@router.post("/stop", tags=["pipeline"])
def stop_post(timeout_sec: float = Query(5.0, ge=1.0, le=60.0)):
    return _stop(timeout=timeout_sec)

@router.post("/run-device", tags=["pipeline"])
def run_device(body: RunDeviceModel):
    return _start_pipeline_for(
        name=body.name,
        dev_id=body.dev_id,
        assign_gps=bool(body.gps_port),     # ✅ FIX
        gps_port=body.gps_port,
    )

@router.post("/run-all-devices", tags=["pipeline"])
def run_all_devices(
    name_prefix: str = Query(
        "blade",
        description="Prefijo para el name lógico de cada pipeline (ej: 'blade' -> blade_1, blade_2, ...)",
    ),
    assign_gps: bool = Query(
        True,
        description="Si True, detecta puertos GPS y asigna 1:1 a cada blade. Si False, no asigna gps_port.",
    ),
    gps_baud: int = Query(4800, ge=300, le=921600, description="Baud rate para detectar GPS"),
):
    """
    Lanza un pipeline por cada bladeRF detectado.
    Ahora opcionalmente enlaza 1 GPS ↔ 1 blade y pasa gps_port al run_pipeline.
    Además pasa node_side/node_port para amarrar LEFT/RIGHT.
    """
    devices = _detect_bladerf_devices()
    if not devices:
        raise HTTPException(
            status_code=404,
            detail="No se encontró ningún dispositivo bladeRF conectado.",
        )

    # Asignación GPS 1:1 (serial -> port)
    serials = [d["serial"] for d in devices if d.get("serial")]
    gps_ports: List[str] = []
    gps_map: Dict[str, str] = {}

    if assign_gps:
        gps_ports = _detect_gps_ports_or_empty(baud=gps_baud)
        if not gps_ports:
            raise HTTPException(
                status_code=404,
                detail="assign_gps=True pero no se detectó ningún GPS (NMEA).",
            )
        gps_map = _assign_gps_ports_to_blades(serials, gps_ports)

    results: dict[str, dict] = {}

    # Copiamos los nombres ya usados para evitar colisiones
    with multi_state.lock:
        used_names = set(multi_state.threads.keys())

    # determinismo: orden por serial
    devices_sorted = sorted(devices, key=lambda d: d.get("serial") or "")

    for idx, dev in enumerate(devices_sorted, start=1):
        serial = dev["serial"]

        # nombre base tipo 'blade_1', 'blade_2', ...
        base_name = f"{name_prefix}_{idx}"
        name = base_name

        # Evitar chocar con algún name ya existente
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        used_names.add(name)

        assigned_gps = gps_map.get(serial) if assign_gps else None

        # LEFT/RIGHT
        side = "left" if idx == 1 else "right" if idx == 2 else f"aux_{idx}"
        node_port = ("NODE_LEFT" if side == "left" else "NODE_RIGHT" if side == "right" else None)
        node_side = (side if side in ("left", "right") else None)

        with node_map.lock:
            node_map.by_serial[serial] = {"node_side": node_side, "node_name": name}

        try:
            start_res = _start_pipeline_for(
                name=name,
                dev_id=serial,
                assign_gps=assign_gps,
                gps_port=assigned_gps,
                gps_baud=gps_baud,
                node_port=node_port,
                node_side=node_side,
            )
            results[name] = {
                "serial": serial,
                "gps_port": assigned_gps,
                "node_side": node_side,
                "node_port": node_port,
                "status": "started",
                "detail": start_res,
            }
        except HTTPException as e:
            results[name] = {
                "serial": serial,
                "gps_port": assigned_gps,
                "node_side": node_side,
                "node_port": node_port,
                "status": "error",
                "http_status": e.status_code,
                "detail": e.detail,
            }
        except Exception as e:
            results[name] = {
                "serial": serial,
                "gps_port": assigned_gps,
                "node_side": node_side,
                "node_port": node_port,
                "status": "error",
                "http_status": 500,
                "detail": str(e),
            }

    return {
        "count": len(devices_sorted),
        "devices": devices_sorted,
        "assign_gps": assign_gps,
        "gps_ports_detected": gps_ports,
        "gps_map_serial_to_port": gps_map,
        "pipelines": results,
    }



@router.get("/status-all", tags=["pipeline"])
def status_all():
    return _status_all()


@router.post("/stop-device", tags=["pipeline"])
def stop_device(
    name: str = Query(..., description="ID interno del pipeline (ej: 'blade_1', 'blade_2')"),
    timeout_sec: float = Query(5.0, ge=1.0, le=60.0),
):
    return _stop_one(name=name, timeout=timeout_sec)


@router.post("/stop-all", tags=["pipeline"])
def stop_all(timeout_sec: float = Query(5.0, ge=1.0, le=60.0)):
    return _stop_all(timeout=timeout_sec)


# ============================ ENDPOINTS PSD ============================
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
        0.8 * np.exp(1j * 2 * np.pi * (+0.12 * sample_rate) * t) +
        0.6 * np.exp(1j * 2 * np.pi * (-0.18 * sample_rate) * t)
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
async def ws_doa(ws: WebSocket):
    """
    WebSocket DOA:
    - NO requiere query params.
    - Envía mensajes JSON mínimos por cada blade cuando haya DOA nuevo:
        { "blade_serial": "SERIAL", "angle_deg": 12.3 }
    """
    await ws.accept()

    # ✅ dedupe por blade
    last_sent_ts_by_blade: Dict[str, float] = {}

    # ✅ período interno fijo (no expuesto por query)
    SLEEP_S = 0.2

    try:
        while True:
            await asyncio.sleep(SLEEP_S)

            all_last = doa_hub.get_last_all() or {}  # {serial: doa_dict}
            if not all_last:
                continue

            for serial, doa in all_last.items():
                if not doa:
                    continue

                s = str(serial)
                ts = float(doa.get("capture_time_sec") or 0.0)
                last_ts = last_sent_ts_by_blade.get(s, 0.0)

                # ✅ si no hay ts o no avanzó, no reenviamos
                if ts and ts <= last_ts:
                    continue

                angle = float(doa.get("angle_deg") or 0.0)

                # ✅ SOLO lo mínimo que necesitas en el front
                await ws.send_json(
                    {
                        "blade_serial": s,
                        "angle_deg": angle,
                    }
                )

                last_sent_ts_by_blade[s] = ts if ts else time.time()

    except WebSocketDisconnect:
        return
    except Exception:
        return


@router.get("/doa/latest", response_model=float)
def get_latest_doa(
    blade_serial: Optional[str] = Query(
        None,
        description="Si se envía, devuelve solo el ángulo DOA de ese blade_serial; si no, devuelve el último global (solo ángulo).",
    ),
):
    """
    Devuelve SOLO el ángulo (float).
    """
    doa = doa_hub.get_last_for(blade_serial) if blade_serial else doa_hub.get_last()
    if not doa:
        raise HTTPException(status_code=404, detail="No hay DOA aún (doa_hub vacío).")
    return float(doa.get("angle_deg") or 0.0)


@router.get("/doa/all/latest")
def get_all_latest_doa() -> Dict[str, float]:
    """
    Devuelve SOLO el ángulo por blade_serial:
      { "SERIAL1": 12.3, "SERIAL2": -40.1, ... }
    """
    all_last = doa_hub.get_last_all()  # {serial: doa_dict}
    out: Dict[str, float] = {}
    for serial, doa in (all_last or {}).items():
        if not doa:
            continue
        out[str(serial)] = float(doa.get("angle_deg") or 0.0)
    return out


@router.post("/doa/ingest", status_code=202)
def ingest_doa(doa: DoaResultModel = Body(...)):
    payload = doa.model_dump()

    # ✅ Asegura timestamp para deduplicación y orden
    payload["capture_time_sec"] = float(payload.get("capture_time_sec") or time.time())

    # ✅ Completa node_side/node_name desde node_map si no vienen
    serial = str(payload.get("blade_serial") or "").strip()
    if serial:
        with node_map.lock:
            nm = node_map.by_serial.get(serial) or {}
        payload.setdefault("node_side", nm.get("node_side"))
        payload.setdefault("node_name", nm.get("node_name"))

    # (Opcional) meta informativa
    meta = payload.get("meta") or {}
    if serial:
        meta.setdefault("blade_serial", serial)
    if payload.get("center_freq_hz") is not None:
        meta.setdefault("center_freq_hz", payload["center_freq_hz"])
    payload["meta"] = meta

    # ✅ guardamos último global y último por blade
    doa_hub.set_last(payload)
    if serial:
        doa_hub.set_last_for(serial, payload)

    return {"ok": True}

# ============================ ENDPOINTS SPEC ============================
@router.post("/spec/ingest", status_code=202)
def ingest_spec(frame: SpecFrameModel = Body(...)):
    spec_hub.set_last(frame.model_dump())
    return {"status": "ok"}


@router.websocket("/ws/spec")
async def ws_spec(
    ws: WebSocket,
    interval_ms: int = Query(200, ge=50, le=5000, description="Período de envío (ms)"),
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

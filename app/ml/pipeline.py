import os
import sys
import ctypes
from pathlib import Path
from datetime import datetime
import time
import threading
import shutil
import subprocess
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import math
import inspect

import numpy as np
import requests
import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import Spectrogram
from scipy.signal import correlate
from bladerf import _bladerf
from collections import Counter, defaultdict
from app.util.orientation import get_orientation
from app.util.triangulation import estimate_emitter_latlon_enu_ls

# ================================================================
# Cargar libbladeRF
# ================================================================
lib_path = "/opt/homebrew/lib/libbladeRF.dylib"
if os.path.exists(lib_path):
    ctypes.cdll.LoadLibrary(lib_path)
    os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(lib_path) + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
else:
    print(f"⚠️ No se encontró la librería en {lib_path}")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ================================================================
# Helpers de serial
# ================================================================
def _devinfo_serial_str(info) -> str:
    s = getattr(info, "serial", "")
    if isinstance(s, (bytes, bytearray)):
        return s.decode("ascii", errors="ignore")
    return str(s)


# ================================================================
# Configuración
# ================================================================
TWO_STAGE_PREDICTION = os.getenv("TWO_STAGE_PREDICTION", "1") == "1"
BINARY_CONFIDENCE_THRESHOLD = float(os.getenv("BINARY_CONFIDENCE_THRESHOLD", "0.4"))

CLASS_DICTS = {
    "binary": {0: "Noise", 1: "Drone"},
    "multiclass": {
        0: "DJI Inspire 2",
        1: "DJI Mini 4K",
        2: "DJI Mavic 2 Air S",
        3: "DJI Mavic Mini",
        4: "DJI Mavic Pro",
        5: "DJI Mavic Pro 2",
        6: "DJI Phantom 4",
        7: "Jammer",
        8: "Parrot Disco",
    },
}

MODEL_CONFIGS = {
    "binary": {
        "path": os.getenv(
            "BINARY_MODEL_PATH",
            "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/model_traced_binary.pt",
        )
    },
    "multiclass": {
        "path": os.getenv(
            "MULTICLASS_MODEL_PATH",
            "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/model_traced.pt",
        )
    },
}

SDR_CONFIG = {
    "sample_rate": 40e6,
    "center_freq": 2440000000,
    "step_freq": 20_000_000,
    "gain": 30,
    "num_samples": int(3e6),
    "min_freq": 2410000000,
    "max_freq": 2470000000,
    "bandwidth_divisor": 2,
    "num_buffers": 16,
    "buffer_size": 8192,
    "num_transfers": 8,
    "stream_timeout": 3500,
    "bytes_per_sample": 4,
}

SPEC_CONFIG = {
    "n_fft": 512,
    "win_length": 512,
    "hop_length": 5860,
    "window_fn": torch.hann_window,
    "power": None,
    "normalized": False,
    "center": False,
    "onesided": False,
}

VIS_CONFIG = {
    "sample_freq": SDR_CONFIG["sample_rate"],
    "n_fft": SPEC_CONFIG["n_fft"],
    "win_length": SPEC_CONFIG["win_length"],
    "hop_length": SPEC_CONFIG["hop_length"],
    "pmin": -50,
    "pmax": 30,
    "figsize": (16, 4),
    "show_stats": True,
}

SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(
    os.getenv("PIPELINE_PLOT_DIR", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/plots")
)
LAST_SPEC_PATH = Path(os.getenv("PIPELINE_LAST_SPEC", "/tmp/morpheus_pipeline/last_spectrogram.png"))
LAST_DOA_PATH = Path(os.getenv("PIPELINE_LAST_DOA", "/tmp/morpheus_pipeline/last_doa.png"))

if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LAST_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_DOA_PATH.parent.mkdir(parents=True, exist_ok=True)

INGEST_URL = os.getenv("PSD_INGEST_URL", "http://127.0.0.1:8000/pipeline/psd/ingest")
PRED_INGEST_URL = os.getenv("PRED_INGEST_URL", "http://127.0.0.1:8000/pipeline/pred/ingest")
DOA_INGEST_URL = os.getenv("DOA_INGEST_URL", "http://127.0.0.1:8000/pipeline/doa/ingest")
SPEC_INGEST_URL = os.getenv("SPEC_INGEST_URL", "http://127.0.0.1:8000/pipeline/spec/ingest")

DOA_CONFIG = {
    "speed_of_light": 299_792_458.0,
    "antenna_spacing_cm": 6.14,
    "block_size": 4096,
    "angles": np.linspace(-90, 90, 721),
    "num_samples": 4096,
    "num_buffers": 32,
    "buffer_size": 8192,
    "num_transfers": 16,
    "dynamic_range_db": 15,
    "military_green": "#4B9920",
}

C0 = 299_792_458.0  # velocidad de la luz (m/s)

SAMPLE_RATE = 40e6  # Hz
CENTER_FREQ = 2_440_000_000  # Hz
GAIN_DB = 30  # dB  (-15 a 60)

N_FFT = 512  # tamaño FFT para espectrograma
WIN_LEN = 512
HOP_LEN = 5860

# Nº de muestras para espectrograma (1 canal)
NUM_SAMPLES_SPEC = 3_000_000  # ~75 ms a 40 MS/s

# --- Ventana temporal deseada para DOA (MVDR) ---
DOA_WINDOW_MS = 10.0  # cambia aquí: 5, 10, 20, 50 ms, etc.

# nº total de muestras de DOA (por canal), en función de DOA_WINDOW_MS
NUM_SAMPLES_DOA = int((DOA_WINDOW_MS / 1000.0) * SAMPLE_RATE)
NUM_SAMPLES_DOA = max(NUM_SAMPLES_DOA, 4096)  # al menos 4096 muestras

# Geometría ULA 2 antenas
D_CM = 6.25  # separación entre antenas en cm
D_M = D_CM / 100.0  # en metros
FC_HZ = 2.44e9  # frecuencia central para DOA

# DOA / MVDR params
BLOCK_SIZE_MVDR = 4096
ANGLES_DEG = np.linspace(-90, 90, 721)  # malla fina
DIAG_LOAD = 1e-3


NODE_GPS_ALL_CAPTURE_URL = os.getenv(
    "NODE_GPS_ALL_CAPTURE_URL",
    "http://127.0.0.1:8000/node/gps/all/capture",
)
NODE_GPS_SAMPLES = int(os.getenv("NODE_GPS_SAMPLES", "5"))
NODE_GPS_REFRESH_S = float(os.getenv("NODE_GPS_REFRESH_S", "3.0"))

PIPELINE_NODE_PORT = os.getenv("PIPELINE_NODE_PORT")  # e.g. NODE_LEFT | NODE_RIGHT
# Alternativa por lado:
PIPELINE_NODE_SIDE = (os.getenv("PIPELINE_NODE_SIDE", "") or "").strip().lower()  # left|right

# Si NO llegan 2 nodos o quieres fallback tipo frontend:
NODE_OFFSET_M = float(os.getenv("NODE_OFFSET_M", "10.0"))  # 10m izquierda / 10m derecha



# ============================
# GPS FALLBACK (ubicación "quemada")
# ============================
FALLBACK_CITY = os.getenv("FALLBACK_CITY", "Medellín, Colombia")
FALLBACK_LAT = float(os.getenv("FALLBACK_LAT", "6.244203"))
FALLBACK_LON = float(os.getenv("FALLBACK_LON", "-75.581212"))


# ================================================================
# Multi-Drone Detection Config
# ================================================================
DRONE_TIMEOUT_SEC = float(os.getenv("DRONE_TIMEOUT_SEC", "5.0"))  # Tiempo sin ver un dron para considerarlo "perdido"
MIN_DETECTIONS_CONFIRM = int(os.getenv("MIN_DETECTIONS_CONFIRM", "2"))  # Detecciones mínimas para confirmar dron
MULTI_DRONE_INGEST_URL = os.getenv("MULTI_DRONE_INGEST_URL", "http://127.0.0.1:8000/pipeline/drones/ingest")


@dataclass
class DroneDetection:
    """Representa una detección de dron en una frecuencia específica."""
    drone_id: str  # ID único del dron (generado)
    frequency_hz: float  # Frecuencia donde se detectó
    drone_class: str  # Clase predicha (DJI Mini 4K, etc.)
    confidence: float  # Confianza de la predicción
    doa_angle: Optional[float]  # Ángulo DOA (None si no se pudo calcular)
    doa_psr_db: Optional[float]  # Calidad del DOA (Peak-to-Sidelobe Ratio)
    first_seen: float  # Timestamp de primera detección
    last_seen: float  # Timestamp de última detección
    detection_count: int  # Número de veces detectado
    consecutive_detections: int  # Detecciones consecutivas actuales
    status: str  # "tentative", "confirmed", "lost"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "frequency_hz": self.frequency_hz,
            "drone_class": self.drone_class,
            "confidence": self.confidence,
            "doa_angle": self.doa_angle,
            "doa_psr_db": self.doa_psr_db,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "detection_count": self.detection_count,
            "consecutive_detections": self.consecutive_detections,
            "status": self.status,
            "age_sec": time.time() - self.first_seen,
            "time_since_last_sec": time.time() - self.last_seen,
        }


class MultiDroneTracker:
    """
    Gestiona múltiples detecciones de drones en diferentes frecuencias.

    Estrategia HÍBRIDA (Barrido + Verificación):
    ============================================
    1. BARRIDO: Descubre nuevos drones con predicción completa
    2. VERIFICACIÓN: Solo DOA en frecuencias conocidas (rápido)

    Ciclo de vida de un dron:
    - "tentative" → Primera(s) detección(es)
    - "confirmed" → Suficientes detecciones consecutivas
    - "lost" → Demasiados fallos de verificación o timeout
    """

    def __init__(self, timeout_sec: float = DRONE_TIMEOUT_SEC,
                 min_confirmations: int = MIN_DETECTIONS_CONFIRM,
                 max_verification_failures: int = 3):
        self.drones: Dict[float, DroneDetection] = {}  # freq_hz -> DroneDetection
        self.timeout_sec = timeout_sec
        self.min_confirmations = min_confirmations
        self.max_verification_failures = max_verification_failures
        self.drone_counter = 0  # Para generar IDs únicos
        self.verification_failures: Dict[float, int] = {}  # freq -> contador de fallos
        self.lock = threading.Lock()

    def _generate_drone_id(self) -> str:
        self.drone_counter += 1
        return f"DRONE_{self.drone_counter:04d}"

    def get_known_frequencies(self) -> List[float]:
        """
        Retorna lista de frecuencias donde hay drones conocidos.
        Usado para la fase de VERIFICACIÓN.
        """
        with self.lock:
            return list(self.drones.keys())

    def get_drone_at_frequency(self, frequency_hz: float) -> Optional[DroneDetection]:
        """Retorna el dron en una frecuencia específica (si existe)."""
        with self.lock:
            existing_freq = self._find_nearby_frequency(frequency_hz)
            if existing_freq is not None:
                return self.drones[existing_freq]
            return None

    def update_doa_verification(self, frequency_hz: float, doa_angle: float,
                                doa_psr_db: float) -> Optional[DroneDetection]:
        """
        Actualiza SOLO el DOA de un dron existente (verificación rápida).
        No requiere predicción completa, solo confirma que el dron sigue ahí.

        Returns:
            DroneDetection actualizada o None si no existe
        """
        now = time.time()

        with self.lock:
            existing_freq = self._find_nearby_frequency(frequency_hz)

            if existing_freq is None:
                return None

            drone = self.drones[existing_freq]
            drone.last_seen = now
            drone.doa_angle = doa_angle
            drone.doa_psr_db = doa_psr_db
            drone.detection_count += 1
            drone.consecutive_detections += 1

            # Resetear contador de fallos de verificación
            self.verification_failures[existing_freq] = 0

            # Promover a "confirmed" si tiene suficientes detecciones
            if (drone.status == "tentative" and
                    drone.consecutive_detections >= self.min_confirmations):
                drone.status = "confirmed"
                print(f"[MultiDrone] ✅ Dron CONFIRMADO: {drone.drone_id} en {frequency_hz / 1e6:.1f} MHz")

            return drone

    def mark_verification_failed(self, frequency_hz: float) -> bool:
        """
        Marca que la verificación DOA falló para esta frecuencia.

        Returns:
            True si el dron fue eliminado por demasiados fallos
        """
        with self.lock:
            existing_freq = self._find_nearby_frequency(frequency_hz)

            if existing_freq is None:
                return False

            # Incrementar contador de fallos
            self.verification_failures[existing_freq] = \
                self.verification_failures.get(existing_freq, 0) + 1

            failures = self.verification_failures[existing_freq]
            drone = self.drones[existing_freq]
            drone.consecutive_detections = 0

            print(f"[MultiDrone] ⚠️  Verificación fallida para {drone.drone_id} "
                  f"({failures}/{self.max_verification_failures})")

            # Eliminar si hay demasiados fallos
            if failures >= self.max_verification_failures:
                drone.status = "lost"
                print(f"[MultiDrone] ❌ Dron PERDIDO (verificación): {drone.drone_id} "
                      f"en {existing_freq / 1e6:.1f} MHz")
                del self.drones[existing_freq]
                del self.verification_failures[existing_freq]
                return True

            return False

    def update_detection(self, frequency_hz: float, drone_class: str,
                         confidence: float, doa_angle: Optional[float] = None,
                         doa_psr_db: Optional[float] = None) -> DroneDetection:
        """
        Actualiza o crea una detección de dron en la frecuencia dada.
        Usado en la fase de BARRIDO (predicción completa).

        Returns:
            DroneDetection actualizada
        """
        now = time.time()

        with self.lock:
            # Buscar si ya existe un dron en esta frecuencia (con tolerancia)
            existing_freq = self._find_nearby_frequency(frequency_hz)

            if existing_freq is not None:
                # Actualizar dron existente
                drone = self.drones[existing_freq]
                drone.last_seen = now
                drone.detection_count += 1
                drone.consecutive_detections += 1
                drone.drone_class = drone_class
                drone.confidence = confidence

                # Resetear contador de fallos
                self.verification_failures[existing_freq] = 0

                # Actualizar DOA si se proporcionó uno válido
                if doa_angle is not None:
                    drone.doa_angle = doa_angle
                    drone.doa_psr_db = doa_psr_db

                # Promover a "confirmed" si tiene suficientes detecciones
                if (drone.status == "tentative" and
                        drone.consecutive_detections >= self.min_confirmations):
                    drone.status = "confirmed"
                    print(f"[MultiDrone] ✅ Dron CONFIRMADO: {drone.drone_id} en {frequency_hz / 1e6:.1f} MHz")

                return drone
            else:
                # Crear nuevo dron
                drone_id = self._generate_drone_id()
                drone = DroneDetection(
                    drone_id=drone_id,
                    frequency_hz=frequency_hz,
                    drone_class=drone_class,
                    confidence=confidence,
                    doa_angle=doa_angle,
                    doa_psr_db=doa_psr_db,
                    first_seen=now,
                    last_seen=now,
                    detection_count=1,
                    consecutive_detections=1,
                    status="tentative"
                )
                self.drones[frequency_hz] = drone
                self.verification_failures[frequency_hz] = 0
                print(
                    f"[MultiDrone] 🆕 Nuevo dron detectado: {drone_id} en {frequency_hz / 1e6:.1f} MHz ({drone_class})")
                return drone

    def _find_nearby_frequency(self, frequency_hz: float, tolerance_hz: float = 5e6) -> Optional[float]:
        """Busca una frecuencia cercana en el tracker (dentro de la tolerancia)."""
        for freq in self.drones.keys():
            if abs(freq - frequency_hz) <= tolerance_hz:
                return freq
        return None

    def mark_no_detection(self, frequency_hz: float):
        """
        Marca que NO se detectó dron en esta frecuencia durante barrido.
        """
        with self.lock:
            existing_freq = self._find_nearby_frequency(frequency_hz)
            if existing_freq is not None:
                self.drones[existing_freq].consecutive_detections = 0

    def cleanup_lost_drones(self) -> List[DroneDetection]:
        """
        Elimina drones que no se han visto en timeout_sec segundos.

        Returns:
            Lista de drones eliminados
        """
        now = time.time()
        lost = []

        with self.lock:
            freqs_to_remove = []
            for freq, drone in self.drones.items():
                if now - drone.last_seen > self.timeout_sec:
                    drone.status = "lost"
                    lost.append(drone)
                    freqs_to_remove.append(freq)
                    print(f"[MultiDrone] ❌ Dron PERDIDO (timeout): {drone.drone_id} en {freq / 1e6:.1f} MHz")

            for freq in freqs_to_remove:
                del self.drones[freq]
                if freq in self.verification_failures:
                    del self.verification_failures[freq]

        return lost

    def get_active_drones(self) -> List[DroneDetection]:
        """Retorna lista de drones activos (tentative + confirmed)."""
        with self.lock:
            return list(self.drones.values())

    def get_confirmed_drones(self) -> List[DroneDetection]:
        """Retorna solo drones confirmados."""
        with self.lock:
            return [d for d in self.drones.values() if d.status == "confirmed"]

    def has_known_drones(self) -> bool:
        """Retorna True si hay drones conocidos para verificar."""
        with self.lock:
            return len(self.drones) > 0

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen del estado actual."""
        with self.lock:
            active = list(self.drones.values())
            return {
                "total_active": len(active),
                "confirmed": len([d for d in active if d.status == "confirmed"]),
                "tentative": len([d for d in active if d.status == "tentative"]),
                "drones": [d.to_dict() for d in active],
                "timestamp": time.time(),
            }


# ================================================================
# Modelos (cargados una vez)
# ================================================================
try:
    if TWO_STAGE_PREDICTION:
        binary_model = torch.jit.load(MODEL_CONFIGS["binary"]["path"], map_location="cpu")
        binary_model.eval()
        print(f"[Model] Modelo binario cargado: {MODEL_CONFIGS['binary']['path']}")

        multiclass_model = torch.jit.load(MODEL_CONFIGS["multiclass"]["path"], map_location="cpu")
        multiclass_model.eval()
        print(f"[Model] Modelo multiclase cargado: {MODEL_CONFIGS['multiclass']['path']}")
    else:
        model = torch.jit.load(MODEL_CONFIGS["multiclass"]["path"], map_location="cpu")
        model.eval()
        print(f"[Model] Modelo único cargado: {MODEL_CONFIGS['multiclass']['path']}")
except Exception as e:
    print(f"[Model] Error cargando modelos: {e}")
    sys.exit(1)


# ================================================================
# Helpers PSD / SPEC / PRED / DOA / GPS
# ================================================================
def _compute_psd_db(x: np.ndarray, nfft: int = 4096) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("x debe ser 1D complejo")
    seg = x[:nfft]
    if seg.shape[0] < nfft:
        seg = np.pad(seg, (0, nfft - seg.shape[0]))
    w = np.hanning(nfft)
    X = np.fft.fftshift(np.fft.fft(seg * w, n=nfft))
    return 10.0 * np.log10(np.abs(X) + 1e-12)


def publish_psd(
        x_complex: np.ndarray,
        center_hz: float,
        sample_rate: float,
        nfft: int = 4096,
        drone_id: Optional[str] = None,
) -> None:
    psd_db = _compute_psd_db(x_complex.astype(np.complex64), nfft=nfft)
    frame = {
        "start_hz": float(center_hz - sample_rate / 2.0),
        "bin_hz": float(sample_rate / nfft),
        "bins": psd_db.astype(float).tolist(),
        "capture_time_sec": time.time(),
        "drone_id": drone_id,
        "fft_size": nfft,
        "schema_version": "1.0",
    }
    try:
        requests.post(INGEST_URL, json=frame, timeout=0.7)
    except Exception as e:
        print("[pipeline] ingest error:", e)


def publish_doa(
    angle_deg: float,
    *,
    blade_serial: str,
    center_freq_hz: Optional[float] = None,
    psr_db: Optional[float] = None,
) -> None:
    body: Dict[str, Any] = {
        "angle_deg": float(angle_deg),
        "blade_serial": str(blade_serial),
        "capture_time_sec": time.time(),
    }
    if center_freq_hz is not None:
        body["center_freq_hz"] = float(center_freq_hz)
    if psr_db is not None:
        body["psr_db"] = float(psr_db)

    try:
        requests.post(DOA_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_doa HTTP error:", e)


def publish_multi_drone(tracker: MultiDroneTracker, blade_name: str = "unknown") -> None:
    """
    Publica el estado completo de todos los drones detectados.
    """
    summary = tracker.get_summary()
    summary["blade_id"] = blade_name
    try:
        requests.post(MULTI_DRONE_INGEST_URL, json=summary, timeout=0.7)
    except Exception as e:
        print("[pipeline] publish_multi_drone HTTP error:", e)


def publish_drone_detection(drone: DroneDetection, blade_name: str = "unknown") -> None:
    """
    Publica una detección individual de dron (para actualizaciones en tiempo real).
    """
    body = {
        "blade_id": blade_name,
        "drone": drone.to_dict(),
        "timestamp": time.time(),
    }
    try:
        # Usar el mismo endpoint de predicción pero con info extendida
        requests.post(PRED_INGEST_URL, json={
            "label": drone.drone_class,
            "confidence": drone.confidence,
            "center_freq": drone.frequency_hz,
            "mode": "continuous_scan",
            "drone_id": drone.drone_id,
            "doa_angle": drone.doa_angle,
            "status": drone.status,
            "timestamp": time.time(),
        }, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_drone_detection HTTP error:", e)


def build_spectrogram_frame(spectrogram: torch.Tensor, class_name: str) -> dict:
    spec_np = spectrogram.detach().cpu().numpy()
    C, F, T = spec_np.shape
    pmin = float(np.nanmin(spec_np))
    pmax = float(np.nanmax(spec_np))
    return {
        "label": class_name,
        "timestamp": time.time(),
        "shape": [C, F, T],
        "n_fft": SPEC_CONFIG["n_fft"],
        "hop_length": SPEC_CONFIG["hop_length"],
        "sample_rate": SDR_CONFIG["sample_rate"],
        "pmin": pmin,
        "pmax": pmax,
        "data": spec_np.tolist(),
    }


def get_connected_blades() -> List[Dict[str, Any]]:
    devinfos = _bladerf.get_device_list()
    devices: List[Dict[str, Any]] = []
    for info in devinfos:
        devices.append({
            "serial": _devinfo_serial_str(info),
            "backend": getattr(info, "backend", None),
            "usb_bus": getattr(info, "usb_bus", None),
            "usb_addr": getattr(info, "usb_addr", None),
        })
    return devices

def count_connected_blades() -> int:
    return len(_bladerf.get_device_list())

def count_unique_blades_by_serial() -> int:
    devinfos = _bladerf.get_device_list()
    serials = [_devinfo_serial_str(i) for i in devinfos]
    return len(set(serials))


def _normalize_bearing_deg(x: float) -> float:
    x = float(x) % 360.0
    return x + 360.0 if x < 0 else x


def _destination(lon_deg: float, lat_deg: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """
    Destino geodésico simple (esfera). Retorna (lon, lat) en grados.
    """
    R = 6371000.0
    brng = math.radians(_normalize_bearing_deg(bearing_deg))
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)

    dr = distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2),
    )

    return (math.degrees(lon2), math.degrees(lat2))


def _fetch_all_nodes_positions_http(n_samples: int, timeout_s: float = 1.5) -> dict:
    """
    Llama al endpoint que trae todas las posiciones de nodos GPS.
    Espera un JSON tipo:
      { count: N, devices: [ {port, ok, data:{avg:{lat,lon}}, n_samples}, ... ] }
    """
    url = f"{NODE_GPS_ALL_CAPTURE_URL}/{int(n_samples)}"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _parse_nodes_payload(payload: dict) -> dict:
    """
    Retorna dict: port -> {"lat": float, "lon": float}
    Solo para devices ok con lat/lon.
    """
    out = {}
    for dev in (payload or {}).get("devices", []) or []:
        port = dev.get("port") or "UNKNOWN"
        ok = bool(dev.get("ok"))
        data = dev.get("data") or {}
        avg = data.get("avg") or {}
        lat = avg.get("lat")
        lon = avg.get("lon")
        if ok and lat is not None and lon is not None:
            out[str(port)] = {"lat": float(lat), "lon": float(lon)}
    return out


def _mean_latlon(pos_by_port: dict) -> tuple[float, float] | tuple[None, None]:
    if not pos_by_port:
        return None, None
    lats = [v["lat"] for v in pos_by_port.values()]
    lons = [v["lon"] for v in pos_by_port.values()]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _compute_left_right_from_avg(avg_lat: float, avg_lon: float, heading_deg: float, offset_m: float) -> dict:
    """
    Genera posiciones sintéticas:
      left  = avg desplazado heading-90
      right = avg desplazado heading+90
    Retorna dict con llaves NODE_LEFT/NODE_RIGHT.
    """
    axis = _normalize_bearing_deg(heading_deg)
    left_lon, left_lat = _destination(avg_lon, avg_lat, axis - 90.0, offset_m)
    right_lon, right_lat = _destination(avg_lon, avg_lat, axis + 90.0, offset_m)
    return {
        "NODE_LEFT": {"lat": float(left_lat), "lon": float(left_lon)},
        "NODE_RIGHT": {"lat": float(right_lat), "lon": float(right_lon)},
    }


def _pick_node_position_for_this_blade(
    blade_name: str,
    pos_by_port: dict,
    *,
    heading_deg: float = 0.0,
    offset_m: float = NODE_OFFSET_M,
    node_port: str | None = None,
    node_side: str | None = None,
) -> tuple[float | None, float | None, str]:
    """
    Decide la posición (lat, lon) de ESTE blade.

    Prioridad:
      1) node_port explícito (PIPELINE_NODE_PORT o param)
      2) node_side left/right (PIPELINE_NODE_SIDE o param) buscando NODE_LEFT/NODE_RIGHT
      3) heurística por nombre del blade (contiene "left"/"right")
      4) si no hay 2 nodos claros: promedio + sintetiza LEFT/RIGHT a ±offset_m con heading
      5) fallback: si hay al menos 1 nodo, usar el primero
    """
    node_port = node_port or PIPELINE_NODE_PORT
    node_side = (node_side or PIPELINE_NODE_SIDE or "").strip().lower()

    # 1) Port explícito
    if node_port and node_port in pos_by_port:
        p = pos_by_port[node_port]
        return p["lat"], p["lon"], f"port:{node_port}"

    # 2) Side explícito
    if node_side in ("left", "right"):
        key = "NODE_LEFT" if node_side == "left" else "NODE_RIGHT"
        if key in pos_by_port:
            p = pos_by_port[key]
            return p["lat"], p["lon"], f"side:{node_side} (real:{key})"

    # 3) Heurística por nombre
    bn = (blade_name or "").lower()
    if "left" in bn and "NODE_LEFT" in pos_by_port:
        p = pos_by_port["NODE_LEFT"]
        return p["lat"], p["lon"], "name:left (real:NODE_LEFT)"
    if "right" in bn and "NODE_RIGHT" in pos_by_port:
        p = pos_by_port["NODE_RIGHT"]
        return p["lat"], p["lon"], "name:right (real:NODE_RIGHT)"

    # 4) Sintetizar LEFT/RIGHT desde promedio si no están
    avg_lat, avg_lon = _mean_latlon(pos_by_port)
    if avg_lat is not None and avg_lon is not None:
        synth = _compute_left_right_from_avg(avg_lat, avg_lon, heading_deg, offset_m)
        if node_side in ("left", "right"):
            key = "NODE_LEFT" if node_side == "left" else "NODE_RIGHT"
            p = synth[key]
            return p["lat"], p["lon"], f"side:{node_side} (synthetic)"
        # Si no hay side, pero el nombre sugiere:
        if "left" in bn:
            p = synth["NODE_LEFT"]
            return p["lat"], p["lon"], "name:left (synthetic)"
        if "right" in bn:
            p = synth["NODE_RIGHT"]
            return p["lat"], p["lon"], "name:right (synthetic)"
        # Si nada: devolver el promedio
        return avg_lat, avg_lon, "avg(all_nodes)"

    # 5) fallback: primer nodo disponible
    if pos_by_port:
        first_port = sorted(pos_by_port.keys())[0]
        p = pos_by_port[first_port]
        return p["lat"], p["lon"], f"fallback:first_port:{first_port}"

    return None, None, "no_gps_available"


def _safe_call_triangulation(fn, **kwargs):
    """
    Llama estimate_emitter_latlon_enu_ls filtrando kwargs por signature (para no romper).
    """
    try:
        sig = inspect.signature(fn)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**filtered)
    except Exception as e:
        print("[Triangulation] error:", e)
        return None

# ============================
# TRIANGULATION SHARED BUFFER (entre hilos de blades)
# ============================
TRI_LOCK = threading.Lock()

TRI_TTL_S = float(os.getenv("TRI_TTL_S", "1.0"))            # ventana de tiempo para considerar DOAs "simultáneos"
TRI_MIN_NODES = int(os.getenv("TRI_MIN_NODES", "2"))        # mínimo de nodos para triangular
TRI_FREQ_TOL_HZ = float(os.getenv("TRI_FREQ_TOL_HZ", "5e6"))# tolerancia por frecuencia (ej 5 MHz)

# bucket_freq -> serial -> obs
TRI_OBS: Dict[float, Dict[str, Dict[str, float]]] = {}

def _tri_bucket(freq_hz: float) -> float:
    return float(round(freq_hz / TRI_FREQ_TOL_HZ) * TRI_FREQ_TOL_HZ)

def tri_update_and_solve(
    *,
    freq_hz: float,
    blade_serial: str,
    doa_local_deg: float,
    node_lat: Optional[float],
    node_lon: Optional[float],
    heading_deg: float = 0.0,
):
    """
    1) Guarda (DOA, lat, lon, heading) del blade actual en un buffer compartido.
    2) Si hay >= TRI_MIN_NODES observaciones recientes para la misma freq (bucket),
       llama estimate_emitter_latlon_enu_ls y retorna TriangulationResult.
    """
    if node_lat is None or node_lon is None:
        return None

    now = time.time()
    key = _tri_bucket(freq_hz)

    with TRI_LOCK:
        bucket = TRI_OBS.setdefault(key, {})
        bucket[str(blade_serial)] = {
            "ts": float(now),
            "doa": float(doa_local_deg),
            "lat": float(node_lat),
            "lon": float(node_lon),
            "heading": float(heading_deg),
        }

        # limpiar stale
        for s in list(bucket.keys()):
            if now - bucket[s]["ts"] > TRI_TTL_S:
                bucket.pop(s, None)

        if len(bucket) < TRI_MIN_NODES:
            return None

        obs = list(bucket.values())

    # IMPORTANTE: este doa_local_deg debe ser consistente con tu heading (0=N, 90=E) según tu doc
    return _safe_call_triangulation(
        estimate_emitter_latlon_enu_ls,
        n_active=len(obs),
        doa_local_deg=[o["doa"] for o in obs],
        node_lat_deg=[o["lat"] for o in obs],
        node_lon_deg=[o["lon"] for o in obs],
        heading_deg=[o["heading"] for o in obs],
        earth_radius_m=6_371_000.0,
    )

# ============================
# ORIENTACIÓN (solo una vez, cacheada por proceso)
# ============================
_ORI_LOCK = threading.Lock()
_ORI_HEADING_DEG: Optional[float] = None

def get_fixed_heading_deg() -> float:
    """
    Lee el heading del BNO055 UNA sola vez (por proceso) y lo reutiliza.
    """
    global _ORI_HEADING_DEG

    with _ORI_LOCK:
        if _ORI_HEADING_DEG is not None:
            return _ORI_HEADING_DEG

        try:
            samples = int(os.getenv("BNO055_SAMPLES", "5"))
            h = get_orientation(samples=samples)  # <-- tu función retorna float
            if h is None or (isinstance(h, float) and math.isnan(h)):
                raise ValueError("heading inválido (NaN/None)")

            _ORI_HEADING_DEG = _normalize_bearing_deg(float(h))
            print(f"[Orientation] ✅ Heading fijo: {_ORI_HEADING_DEG:.2f}° (samples={samples})")
        except Exception as e:
            _ORI_HEADING_DEG = 0.0
            print(f"[Orientation] ⚠️ No se pudo leer BNO055, usando 0.0° | error: {e}")

        return _ORI_HEADING_DEG


# =============================================================================
# Utilidades generales
# =============================================================================

def wavelength(fc_hz: float) -> float:
    """Longitud de onda λ = c / f."""
    return C0 / fc_hz


def preprocess_iq(X: np.ndarray,
                  *,
                  demean: bool = True,
                  normalize: bool = False) -> np.ndarray:
    """
    Limpieza básica de IQ para MVDR.

    X: array complejo de forma (K, M)
       K = snapshots en el tiempo
       M = nº de sensores / canales

    - demean: elimina DC por canal
    - normalize: iguala RMS por canal
    """
    if X.ndim != 2 or not np.iscomplexobj(X):
        raise ValueError(f"Expected complex array shaped (K, M). "
                         f"Got {X.shape}, complex={np.iscomplexobj(X)}")

    Xp = X.astype(np.complex64, copy=False)

    if demean:
        # media por columna (canal)
        Xp = Xp - np.mean(Xp, axis=0, keepdims=True)

    if normalize:
        rms = np.sqrt(np.mean(np.abs(Xp) ** 2, axis=0, keepdims=True)) + 1e-12
        Xp = Xp / rms

    return Xp


# =============================================================================
# Funciones de MVDR
# =============================================================================

def steering_vector(theta_deg: float, M: int, d_lambda: float) -> np.ndarray:
    """
    Vector director ULA:
      a_m(θ) = exp(-j 2π d/λ m sin θ),  m = 0..M-1
    """
    m = np.arange(M)[:, None]  # (M,1)
    return np.exp(-1j * 2.0 * np.pi * d_lambda * m * np.sin(np.deg2rad(theta_deg)))


def mvdr_block(X_block: np.ndarray,
               angles: np.ndarray,
               d_lambda: float,
               diag_load: float = 1e-3) -> np.ndarray:
    """
    MVDR sobre un bloque X_block.

    X_block: (K, M) snapshots complejos
    angles:  array de ángulos en grados
    d_lambda: espaciamiento normalizado d/λ
    """
    X_block = np.asarray(X_block, dtype=np.complex128)
    K, M = X_block.shape

    # Covarianza MxM
    R = (X_block.conj().T @ X_block) / max(1, K)

    # Carga diagonal para estabilidad numérica
    delta = diag_load * (np.trace(R).real / M)
    R = R + delta * np.eye(M)

    # Inversa
    R_inv = np.linalg.inv(R)

    P = np.empty(len(angles), dtype=float)
    for i, ang in enumerate(angles):
        a = steering_vector(ang, M, d_lambda)  # (M,1)
        denom = (a.conj().T @ R_inv @ a).item()
        P[i] = 1.0 / max(np.real(denom), 1e-12)

    return P


def estimate_doa_mvdr(X_full: np.ndarray,
                      fc_hz: float,
                      d_m: float,
                      block_size: int,
                      angles_deg: np.ndarray,
                      diag_load: float):
    """
    Ejecuta MVDR por bloques y estima el DOA (máximo del espectro).

    X_full: (K, 2) complejo (K snapshots, 2 antenas)
    Devuelve: P_mvdr (espectro medio), theta_hat (DOA), psr_db (peak/median en dB)
    """
    lam = wavelength(fc_hz)
    d_lambda = d_m / lam

    num_blocks = X_full.shape[0] // block_size
    acc = np.zeros(len(angles_deg), dtype=float)

    for b in range(num_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        Xb = X_full[sl, :]  # (K, 2)
        acc += mvdr_block(Xb, angles_deg, d_lambda, diag_load=diag_load)

    P_mvdr = acc / max(1, num_blocks)

    # Relación pico/mediana como métrica de claridad
    P_lin = np.asarray(P_mvdr, float)
    psr_db = 10 * np.log10(P_lin.max() / (np.median(P_lin) + 1e-12))

    # Búsqueda de máximo con refinamiento parabólico
    ang = np.asarray(angles_deg, float)
    step = ang[1] - ang[0]

    edge = 2
    i_search = np.arange(edge, len(P_lin) - edge)
    i0 = i_search[np.argmax(P_lin[i_search])]

    if 0 < i0 < len(P_lin) - 1:
        y1, y2, y3 = P_lin[i0 - 1], P_lin[i0], P_lin[i0 + 1]
        denom = (y1 - 2 * y2 + y3)
        delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
    else:
        delta = 0.0

    theta_hat = ang[i0] + delta * step

    return P_mvdr, theta_hat, psr_db, d_lambda


class transform_spectrogram(torch.nn.Module):
    def __init__(self, device, n_fft=512, win_length=512, hop_length=5860,
                 window_fn=torch.hann_window, power=None, normalized=False,
                 center=False, onesided=False):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided,
        ).to(device=device)
        self.win_length = win_length
        self.epsilon = 1e-12

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        iq_signal = iq_signal[0, :] + 1j * iq_signal[1, :]
        iq_signal = iq_signal - iq_signal.mean()
        spec_complex = self.spec(iq_signal)
        spec_real = spec_complex.real
        spec_imag = spec_complex.imag
        spec_magnitude = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + self.epsilon)
        spec_db = 10 * torch.log10(spec_magnitude + self.epsilon)
        return spec_db


def two_stage_predict(binary_model, multiclass_model, spectrogram, binary_threshold=0.5,
                      binary_results_list=None, multiclass_results_list=None, window_size=5):
    spec_input = spectrogram.type(torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        binary_outputs = binary_model(spec_input).view(-1)
        binary_probs = torch.sigmoid(binary_outputs)
        binary_pred = (binary_probs > binary_threshold).float()
        binary_class_idx = int(binary_pred.item())
        binary_class_name = CLASS_DICTS["binary"][binary_class_idx]
        binary_confidence = binary_probs.item() if binary_class_idx == 1 else 1 - binary_probs.item()

    binary_result = {
        "class_name": binary_class_name,
        "confidence": binary_confidence,
        "raw_prob": binary_probs.item(),
    }

    _, binary_result_smoothed, binary_results_list = smooth_predictions(
        binary_result, binary_results_list, window_size, adaptive=True
    )

    multiclass_result = None
    multiclass_result_smoothed = None
    final_class_name = binary_result_smoothed['class_name']

    if binary_result_smoothed['class_name'] == 'Drone':
        with torch.no_grad():
            multiclass_outputs = multiclass_model(spec_input)
            multiclass_probs = torch.softmax(multiclass_outputs, dim=1)
            multiclass_pred = multiclass_outputs.argmax(dim=1)
            multiclass_class_idx = int(multiclass_pred.item())
            multiclass_class_name = CLASS_DICTS["multiclass"][multiclass_class_idx]
            multiclass_confidence = multiclass_probs[0, multiclass_class_idx].item()

        multiclass_result = {
            "class_name": multiclass_class_name,
            "confidence": multiclass_confidence,
            "class_idx": multiclass_class_idx,
            "all_probs": multiclass_probs[0].cpu().numpy().tolist(),
        }

        _, multiclass_result_smoothed, multiclass_results_list = smooth_predictions(
            multiclass_result, multiclass_results_list, window_size, adaptive=False
        )

        final_class_name = multiclass_result_smoothed['class_name']

    else:
        # Add None to maintain temporal alignment
        if multiclass_results_list is None:
            multiclass_results_list = []
        multiclass_results_list.append(None)

    return (final_class_name, binary_result, binary_result_smoothed,
            multiclass_result, multiclass_result_smoothed,
            binary_results_list, multiclass_results_list)


# ----------------------------------
# Moving mode and confidence ponderation prediction
# -----------------------------------
def smooth_predictions(result, results_list, window_size=5, adaptive=True, max_history=None):
    """
    Perform moving mode prediction over a list of results.
    Handles None values in history to maintain temporal alignment.
    Uses adaptive window sizing to respond faster to state changes.

    Args:
        result: dict with 'class_name' and 'confidence' for current prediction
        results_list: List of dicts with 'class_name' and 'confidence' (may contain None)
        window_size: Maximum size of the moving window
        adaptive: If True, use smaller window when detecting changes
        max_history: Maximum history length to prevent memory leak (default: window_size * 3)

    Returns:
        tuple: (original_result, corrected_result, updated_results_list)
    """
    if results_list is None:
        results_list = []

    # Add current result to history
    results_list.append(result)

    # Trim history to prevent unbounded growth
    if max_history is None:
        max_history = window_size * 3  # Keep 3x window size as buffer
    if len(results_list) > max_history:
        results_list = results_list[-max_history:]

    # Get recent results window and filter out None values
    recent_results = results_list[-window_size:]
    valid_results = [r for r in recent_results if r is not None]

    # Minimum valid predictions required for smoothing
    MIN_VALID_PREDICTIONS = 2
    if len(valid_results) < MIN_VALID_PREDICTIONS:
        return result, result.copy(), results_list

    # Adaptive window: detect if current prediction differs from recent consensus
    # ONLY apply if adaptive=True (binary stage uses this, multiclass doesn't)
    effective_window_used = len(valid_results)  # Track actual window size used

    if adaptive and len(valid_results) >= 2:
        # Calculate weighted consensus from history (EXCLUDING current prediction)
        prev_results = valid_results[:-1]
        prev_scores = defaultdict(float)

        for idx, res in enumerate(prev_results):
            class_name = res['class_name']
            confidence = res['confidence']
            # Use same weighting as main scoring
            recency_weight = min((idx + 1) / len(prev_results), 0.85)
            prev_scores[class_name] += confidence * recency_weight

        # Get weighted consensus
        if prev_scores:
            prev_consensus = max(prev_scores, key=prev_scores.get)
            current_class = result['class_name']

            # If current differs from weighted consensus, use smaller window
            if current_class != prev_consensus:
                effective_window_size = min(3, window_size)
                recent_results = results_list[-effective_window_size:]
                valid_results = [r for r in recent_results if r is not None]
                effective_window_used = len(valid_results)

                # Re-check minimum after window reduction
                if len(valid_results) < MIN_VALID_PREDICTIONS:
                    return result, result.copy(), results_list

    # Calculate weighted scores for each class (use valid_results directly)
    class_scores = defaultdict(float)

    for idx, res in enumerate(valid_results):
        class_name = res['class_name']
        confidence = res['confidence']

        # Recency weight: more recent = higher weight
        recency_weight = min((idx + 1) / len(valid_results), 0.85)

        # Combined score: confidence * recency
        class_scores[class_name] += confidence * recency_weight

    scores_str = ", ".join([f"{k}: {v:.4f}" for k, v in class_scores.items()])
    print(f"[Suavizado] {scores_str}")

    # Find class with highest combined score
    if class_scores:
        best_class = max(class_scores, key=class_scores.get)

        # Calculate average confidence for the best class only
        # Filter predictions that match best_class
        best_class_predictions = [r for r in valid_results if r['class_name'] == best_class]

        if best_class_predictions:
            # Weighted average of confidences for winning class
            weighted_conf_sum = 0
            weight_sum = 0
            for idx, res in enumerate(valid_results):
                if res['class_name'] == best_class:
                    recency_weight = min((idx + 1) / len(valid_results), 0.85)
                    weighted_conf_sum += res['confidence'] * recency_weight
                    weight_sum += recency_weight

            # This is the weighted average confidence for the winning class
            normalized_confidence = weighted_conf_sum / weight_sum if weight_sum > 0 else 0
        else:
            normalized_confidence = 0

        result_corrected = {
            'class_name': best_class,
            'confidence': normalized_confidence,  # Now properly in [0, 1]
            'raw_confidence': result['confidence'],  # Original model confidence
            'raw_prob': result.get('raw_prob'),
            'all_probs': result.get('all_probs'),
            'raw_score': class_scores[best_class],  # Unnormalized score for debugging
            'window_size_used': effective_window_used
        }
    else:
        result_corrected = result.copy()

    return result, result_corrected, results_list


def visualize_spectrogram(
        spectrogram: torch.Tensor,
        class_name: str,
        blade_id: str,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 5860,
        sample_freq: float = 40e6,
        pmin: Optional[float] = None,
        pmax: Optional[float] = None,
        show_stats: bool = True,
        figsize: tuple = (16, 4),
) -> None:
    if blade_id is None:
        blade_id = os.getenv("PIPELINE_DEVICE_NAME", "default_blade")

    if spectrogram.ndim == 3:
        if spectrogram.shape[0] == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="none")
            fig.suptitle(f"{class_name} | {blade_id}")
        elif spectrogram.shape[0] == 1:
            fig, ax_single = plt.subplots(1, 1, figsize=figsize)
            axes = [ax_single]
            fig.suptitle(f"{class_name} | {blade_id}")
        else:
            raise ValueError(f"Spectrogram shape {spectrogram.shape} doesn't match expected dimensions")
    else:
        raise ValueError(f"Expected 3D spectrogram, got shape {spectrogram.shape}")

    spectrogram_np = spectrogram.detach().cpu().numpy()
    n_freq_bins = spectrogram_np.shape[1]
    n_time_bins = spectrogram_np.shape[2]

    time_bin_duration = hop_length / sample_freq
    time_duration_ms = np.arange(n_time_bins) * time_bin_duration * 1e3

    freqs = np.fft.fftfreq(n_fft, 1 / sample_freq)
    sorted_indices = np.argsort(freqs)
    freqs = freqs[sorted_indices]
    freqs_in_mhz = freqs / 1e6

    if pmin is None:
        pmin = np.nanmin(spectrogram_np)
    if pmax is None:
        pmax = np.nanmax(spectrogram_np)

    if len(freqs_in_mhz) != n_freq_bins:
        freqs_in_mhz = np.linspace(freqs_in_mhz[0], freqs_in_mhz[-1], n_freq_bins)

    t = time_duration_ms
    f = freqs_in_mhz

    for ch in range(spectrogram_np.shape[0]):
        axes[ch].imshow(
            spectrogram_np[ch],
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], f[0], f[-1]],
            interpolation="nearest",
            vmin=pmin,
            vmax=pmax,
        )

    plt.tight_layout()

    if show_stats:
        print(
            f"[Visualización] Min={np.nanmin(spectrogram_np):.2f}, "
            f"Max={np.nanmax(spectrogram_np):.2f}, "
            f"Mean={np.nanmean(spectrogram_np):.2f}, "
            f"Std={np.nanstd(spectrogram_np):.2f}"
        )

    if SAVE_PLOTS:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = PLOT_DIR / f"spectrogram_{ts}.png"
            fig.savefig(out_path, dpi=120)
            shutil.copy2(out_path, LAST_SPEC_PATH)
            print(f"[Visualización] Espectrograma guardado: {out_path}")
        except Exception as e:
            print("[Visualización] savefig error:", e)

    plt.close(fig)


def publish_pred(
        label_name: str,
        confidence: Optional[float] = None,
        binary_info: Optional[dict] = None,
        multiclass_info: Optional[dict] = None,
        center_freq: Optional[float] = None,
        mode: Optional[str] = None,
) -> None:
    """
    Publica la predicción actual hacia el endpoint HTTP del backend.

    Args:
        label_name: Nombre de la clase final ("Noise", "DJI Mini 4K", etc.).
        confidence: Confianza principal asociada a la detección (típicamente binaria).
        binary_info: Dict opcional con detalles del clasificador binario suavizado.
        multiclass_info: Dict opcional con detalles del clasificador multiclase suavizado.
        center_freq: Frecuencia central (Hz) usada en esta captura.
        mode: Modo del pipeline en esta iteración ("scan" o "track").
    """
    body = {"label": label_name, "timestamp": time.time()}
    if confidence is not None:
        body["confidence"] = float(confidence)

    if center_freq is not None:
        body["center_freq"] = float(center_freq)

    if mode is not None:
        body["mode"] = str(mode)

    if TWO_STAGE_PREDICTION:
        body["prediction_mode"] = "two_stage"
        if binary_info:
            body["binary"] = binary_info
        if multiclass_info:
            body["multiclass"] = multiclass_info
    else:
        body["prediction_mode"] = "single_stage"
    try:
        requests.post(PRED_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_pred HTTP error:", e)


def publish_spec(frame: dict) -> None:
    try:
        requests.post(SPEC_INGEST_URL, json=frame, timeout=0.7)
    except Exception as e:
        print("[pipeline] publish_spec HTTP error:", e)


# ================================================================
# bladeRF helper
# ================================================================
def _open_bladerf_from_env(dev_id: Optional[str]):
    try:
        devinfos = _bladerf.get_device_list()
    except Exception as e:
        print(f"[SDR] Error llamando a get_device_list(): {e}")
        raise RuntimeError("get_device_list failed") from e

    if not devinfos:
        print("[SDR] No se encontraron dispositivos bladeRF en get_device_list().")
        try:
            out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
            print("[SDR] bladeRF-cli -p stdout:\n", out.stdout)
            print("[SDR] bladeRF-cli -p stderr:\n", out.stderr)
        except Exception as e2:
            print("[SDR] No pude ejecutar bladeRF-cli -p:", e2)
        raise RuntimeError("No bladeRF devices found")

    if not dev_id:
        raise RuntimeError("BLADERF_DEVICE no definido")

    serial_prefix = dev_id.strip()
    print(f"[SDR] BLADERF_DEVICE='{serial_prefix}', buscando por prefijo de serial...")

    for info in devinfos:
        serial_str = _devinfo_serial_str(info)
        print(f"   - encontrado dispositivo con serial={serial_str}")
        if serial_str.startswith(serial_prefix):

            # =========================
            # Configuración de TRIGGER vía bladeRF-cli (rol SLAVE en J51-1 RX)
            # =========================
            try:
                # Comando base
                cli_cmd = ["bladeRF-cli"]
                # Si tienes BLADERF_DEVICE definido, úsalo también aquí
                if serial_str:
                    device_arg = f"*:serial={serial_str}"
                    cli_cmd += ["-d", device_arg]
                # Configurar trigger en J51-1 como SLAVE de RX
                # Equivalente a: bladerf_trigger_init + trig.role = SLAVE + bladerf_trigger_arm
                cli_cmd += ["-e", "trigger J51-1 rx slave"]
                print("[pipeline] Configurando trigger J51-1 RX como SLAVE mediante bladeRF-cli...")
                res = subprocess.run(cli_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print("[pipeline] ADVERTENCIA: fallo al configurar trigger con bladeRF-cli")
                    print("[pipeline] stdout:\n", res.stdout)
                    print("[pipeline] stderr:\n", res.stderr)
                else:
                    print("[pipeline] Trigger configurado correctamente como SLAVE en J51-1 RX.")
            except FileNotFoundError:
                print(
                    "[pipeline] ADVERTENCIA: bladeRF-cli no encontrado en el PATH. Se omite configuración de trigger.")
            except Exception as e:
                print("[pipeline] ADVERTENCIA: Error inesperado al configurar trigger vía bladeRF-cli:", e)
            print(f"[SDR] Abriendo bladeRF con serial que empieza por '{serial_prefix}'")
            try:
                dev = _bladerf.BladeRF(devinfo=info)
                print(f"[SDR] Dispositivo abierto: {dev}")
                return dev, serial_str
            except Exception as e:
                print(f"[SDR] Error abriendo dispositivo con ese serial: {e}")
                raise RuntimeError("Failed to open bladeRF") from e

    print(f"[SDR] No se encontró ningún dispositivo cuyo serial empiece por '{serial_prefix}'")
    try:
        out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
        print("[SDR] bladeRF-cli -p stdout:\n", out.stdout)
        print("[SDR] bladeRF-cli -p stderr:\n", out.stderr)
    except Exception as e2:
        print("[SDR] No pude ejecutar bladeRF-cli -p:", e2)
    raise RuntimeError("No bladeRF matching BLADERF_DEVICE")


def configure_common_channels(sdr: _bladerf.BladeRF):
    rx0 = sdr.Channel(_bladerf.CHANNEL_RX(0))
    rx1 = sdr.Channel(_bladerf.CHANNEL_RX(1))

    for ch in (rx0, rx1):
        ch.frequency = CENTER_FREQ
        ch.sample_rate = SAMPLE_RATE
        ch.bandwidth = SAMPLE_RATE / 2
        ch.gain_mode = _bladerf.GainMode.Manual
        ch.gain = GAIN_DB

    return rx0, rx1


def capture_single_channel(sdr: _bladerf.BladeRF, num_samples: int) -> np.ndarray:
    """Captura IQ de un solo canal en modo RX_X1."""
    bytes_per_sample = 4  # I int16 + Q int16
    buf = bytearray(1024 * bytes_per_sample)

    x = np.zeros(num_samples, dtype=np.complex64)
    num_samples_read = 0

    while True:
        if num_samples > 0 and num_samples_read == num_samples:
            break
        elif num_samples > 0:
            num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
        else:
            num = len(buf) // bytes_per_sample

        sdr.sync_rx(buf, num)

        samples = np.frombuffer(buf, dtype=np.int16, count=num * 2)
        samples = samples[0::2] + 1j * samples[1::2]
        samples /= 2048.0

        x[num_samples_read:num_samples_read + num] = samples[:num]
        num_samples_read += num

    return x


def capture_dual_channel(sdr: _bladerf.BladeRF, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Captura IQ de dos canales en modo RX_X2: I0,Q0,I1,Q1."""
    bytes_per_sample = 8  # I0,Q0,I1,Q1 (4 int16)
    buf = bytearray(1024 * bytes_per_sample)

    x1 = np.zeros(num_samples, dtype=np.complex64)
    x2 = np.zeros(num_samples, dtype=np.complex64)
    num_samples_read = 0

    while True:
        if num_samples > 0 and num_samples_read == num_samples:
            break
        elif num_samples > 0:
            num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
        else:
            num = len(buf) // bytes_per_sample

        sdr.sync_rx(buf, num)

        raw_i16 = np.frombuffer(buf, dtype=np.int16, count=num * 4)
        raw = raw_i16.reshape(num, 4)

        s1 = (raw[:, 0] + 1j * raw[:, 1]) / 2048.0
        s2 = (raw[:, 2] + 1j * raw[:, 3]) / 2048.0

        x1[num_samples_read:num_samples_read + num] = s1
        x2[num_samples_read:num_samples_read + num] = s2
        num_samples_read += num

    return x1, x2


# ================================================================
# run_pipeline: para usarlo desde FastAPI en hilos
# ================================================================
def run_pipeline(
    dev_id: Optional[str] = None,
    device_name: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
    node_port: Optional[str] = None,
    node_side: Optional[str] = None,
    gps_port: Optional[str] = None,     # ✅ NUEVO
    gps_baud: int = 4800,               # ✅ NUEVO
) -> None:
    """
    Bucle principal del pipeline. Bloqueante. Pensado para correrse en un hilo.

    COMPORTAMIENTO HÍBRIDO (Barrido + Verificación DOA):
    ====================================================

    Cada iteración tiene DOS FASES:

    FASE 1 - VERIFICACIÓN (rápida):
    --------------------------------
    Para cada frecuencia donde hay un dron conocido:
    - Sintoniza la frecuencia
    - Ejecuta SOLO DOA (sin predicción completa)
    - Si DOA válido: actualiza ángulo y confirma que sigue ahí
    - Si DOA inválido: incrementa contador de fallos

    FASE 2 - BARRIDO (descubrimiento):
    -----------------------------------
    - Avanza a la siguiente frecuencia del barrido secuencial
    - Captura completa + Espectrograma + Predicción
    - Si detecta dron nuevo: lo agrega al tracker + DOA
    - Continúa con el ciclo

    Ventajas:
    - Verificación rápida de drones conocidos (solo DOA ~100ms)
    - Descubrimiento continuo de nuevos drones
    - Actualización frecuente de ángulos DOA
    - No se "ciega" en una sola frecuencia
    """

    # Parámetros DOA
    NUM_READS_VERIFY = 2  # Lecturas MVDR para verificación (rápido)
    NUM_READS_DISCOVER = 3  # Lecturas MVDR para descubrimiento
    PSR_MIN_DB = 8.0  # Umbral de calidad DOA
    WINDOW_WIDTH = 10.0  # Grados para clustering de ángulos
    MAX_VERIFICATION_FAILURES = int(os.getenv("MAX_VERIFICATION_FAILURES", "3"))

    effective_dev_id = dev_id or os.getenv("BLADERF_DEVICE")
    blade_name = device_name or os.getenv("PIPELINE_DEVICE_NAME", effective_dev_id or "default_blade")

    fixed_heading_deg = get_fixed_heading_deg()
    # ✅ Cargar GPS al inicio (o fallback Medellín)


    gps_cache = {
        "last_fetch": 0.0,
        "pos_by_port": {},  # port -> {lat, lon}
        "last_debug": "init",
        "node_lat": None,
        "node_lon": None,
        "node_src": "init",
    }

    def refresh_gps_cache(force: bool = False):
        now = time.time()
        if (not force) and (now - gps_cache["last_fetch"] < NODE_GPS_REFRESH_S):
            return

        heading_deg = fixed_heading_deg

        try:
            payload = _fetch_all_nodes_positions_http(NODE_GPS_SAMPLES, timeout_s=1.5)
            pos_by_port = _parse_nodes_payload(payload)

            # ✅ Si no hay nodos reales, usamos Medellín y sintetizamos LEFT/RIGHT
            if not pos_by_port:
                pos_by_port = _compute_left_right_from_avg(
                    FALLBACK_LAT,
                    FALLBACK_LON,
                    heading_deg,
                    NODE_OFFSET_M,
                )
                src_note = f"fallback:{FALLBACK_CITY} (synthetic LEFT/RIGHT)"
            else:
                src_note = "gps:nodes_http"

            node_lat, node_lon, src = _pick_node_position_for_this_blade(
                blade_name=blade_name,
                pos_by_port=pos_by_port,
                heading_deg=heading_deg,
                offset_m=NODE_OFFSET_M,
                node_port=node_port,
                node_side=node_side,
            )

            # ✅ Si por alguna razón sigue None, forzamos fallback Medellín promedio
            if node_lat is None or node_lon is None:
                synth = _compute_left_right_from_avg(FALLBACK_LAT, FALLBACK_LON, heading_deg, NODE_OFFSET_M)
                node_lat, node_lon, src2 = _pick_node_position_for_this_blade(
                    blade_name=blade_name,
                    pos_by_port=synth,
                    heading_deg=heading_deg,
                    offset_m=NODE_OFFSET_M,
                    node_port=node_port,
                    node_side=node_side,
                )
                src = f"{src2} | fallback:{FALLBACK_CITY}"
                pos_by_port = synth
                src_note = f"fallback:{FALLBACK_CITY} (forced)"

            gps_cache["last_fetch"] = now
            gps_cache["pos_by_port"] = pos_by_port
            gps_cache["node_lat"] = node_lat
            gps_cache["node_lon"] = node_lon
            gps_cache["node_src"] = f"{src_note} | {src}"
            gps_cache["last_debug"] = "ok"

        except Exception as e:
            # ✅ Si el HTTP falla, igual caemos a Medellín y seguimos vivos
            pos_by_port = _compute_left_right_from_avg(FALLBACK_LAT, FALLBACK_LON, heading_deg, NODE_OFFSET_M)
            node_lat, node_lon, src = _pick_node_position_for_this_blade(
                blade_name=blade_name,
                pos_by_port=pos_by_port,
                heading_deg=heading_deg,
                offset_m=NODE_OFFSET_M,
                node_port=node_port,
                node_side=node_side,
            )

            gps_cache["last_fetch"] = now
            gps_cache["pos_by_port"] = pos_by_port
            gps_cache["node_lat"] = node_lat
            gps_cache["node_lon"] = node_lon
            gps_cache["node_src"] = f"http_error:{e} | fallback:{FALLBACK_CITY} | {src}"
            gps_cache["last_debug"] = f"http_error:{e}"

    refresh_gps_cache(force=True)


    print(f"[Pipeline] run_pipeline(dev_id={effective_dev_id}, name={blade_name})")
    print(f"[Pipeline] MODO: HÍBRIDO (Barrido + Verificación DOA)")

    # Parámetros de configuración
    SCAN_WINDOW = int(os.getenv("SCAN_WINDOW_SIZE", "1"))
    CLEANUP_INTERVAL = int(os.getenv("DRONE_CLEANUP_INTERVAL", "10"))

    # Frecuencias de escaneo
    min_freq = int(SDR_CONFIG["min_freq"])
    max_freq = int(SDR_CONFIG["max_freq"])
    step_freq = int(SDR_CONFIG["step_freq"])


    if step_freq <= 0:
        raise RuntimeError("SDR_CONFIG['step_freq'] debe ser > 0")

    scan_freqs = list(range(min_freq, max_freq + 1, step_freq))
    if not scan_freqs:
        scan_freqs = [int(SDR_CONFIG["center_freq"])]

    print(f"[Pipeline] Frecuencias de escaneo: {[f'{f / 1e6:.0f} MHz' for f in scan_freqs]}")
    print(f"[Pipeline] Total de frecuencias: {len(scan_freqs)}")
    print(f"[Pipeline] Timeout de drones: {DRONE_TIMEOUT_SEC} seg")
    print(f"[Pipeline] Max fallos verificación: {MAX_VERIFICATION_FAILURES}")

    # ========================================
    # Inicializar Multi-Drone Tracker
    # ========================================
    drone_tracker = MultiDroneTracker(
        timeout_sec=DRONE_TIMEOUT_SEC,
        min_confirmations=MIN_DETECTIONS_CONFIRM,
        max_verification_failures=MAX_VERIFICATION_FAILURES
    )

    # Estado del barrido
    scan_index = 0
    iteration_count = 0

    try:
        sdr, blade_serial = _open_bladerf_from_env(effective_dev_id)
    except Exception as e:
        print(f"[Pipeline] No se pudo abrir bladeRF ({blade_name}): {e}")
        return

    # Configurar PLL
    sdr.set_pll_refclk(int(10e6))
    sdr.set_pll_enable(True)
    for _ in range(50):
        if sdr.get_pll_lock_state():
            break
        time.sleep(0.1)

    rx0, rx1 = configure_common_channels(sdr)

    # Config canales
    rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))
    rx1 = sdr.Channel(_bladerf.CHANNEL_RX(1)) #-----> pendiente, rx1 deja de ser RX(1)

    for ch in (rx1, rx_ch):
        ch.sample_rate = SDR_CONFIG["sample_rate"]
        ch.bandwidth = SDR_CONFIG["sample_rate"] / SDR_CONFIG["bandwidth_divisor"]
        ch.gain_mode = _bladerf.GainMode.Manual
        ch.gain = SDR_CONFIG["gain"]

    rx1.enable = False
    rx_ch.enable = True

    buf = bytearray(1024 * SDR_CONFIG["bytes_per_sample"])
    transform = transform_spectrogram(device="cpu", **SPEC_CONFIG)

    print(f"[Pipeline] Iniciando recepción ({blade_name}) (Two-stage: {TWO_STAGE_PREDICTION})")
    print(f"[Pipeline] ═══════════════════════════════════════════════════")

    # Historial para suavizado de predicciones (por frecuencia)
    binary_history_per_freq: Dict[float, List] = defaultdict(list)
    multiclass_history_per_freq: Dict[float, List] = defaultdict(list)

    # refresh_gps_cache(force=True)

    # ========================================
    # Función auxiliar: Ejecutar DOA en una frecuencia
    # ========================================
    def execute_doa_at_frequency(freq_hz: float, num_reads: int) -> tuple[Optional[float], Optional[float]]:
        """
        Ejecuta DOA en la frecuencia especificada.
        Returns: (doa_angle, doa_psr) o (None, None) si falla
        """
        # Configurar para dual-channel

        time.sleep(0.2)
        # rx0.enable = False
        # rx1.enable = False
        # rx_ch.enable = False
        #
        # for ch in (rx0, rx1):
        #     ch.frequency = int(freq_hz)
        #
        # sdr.sync_config(
        #     layout=_bladerf.ChannelLayout.RX_X2,
        #     fmt=_bladerf.Format.SC16_Q11,
        #     num_buffers=32,
        #     buffer_size=16384,
        #     num_transfers=16,
        #     stream_timeout=3500,
        # )
        # rx0.enable = True
        # rx1.enable = True

        spectra_acc = None
        angles_list = []
        valid_reads = 0
        last_psr = 0.0

        for _ in range(num_reads):
            x1, x2 = capture_dual_channel(sdr, NUM_SAMPLES_DOA)

            # Normalizar
            x1 /= np.sqrt(np.mean(np.abs(x1) ** 2) + 1e-12)
            x2 /= np.sqrt(np.mean(np.abs(x2) ** 2) + 1e-12)

            X_full = np.stack([x1, x2], axis=1)
            X_full = preprocess_iq(X_full, demean=True, normalize=True)

            P_mvdr, theta_hat, psr_db, _ = estimate_doa_mvdr(
                X_full,
                fc_hz=FC_HZ,
                d_m=D_M,
                block_size=BLOCK_SIZE_MVDR,
                angles_deg=ANGLES_DEG,
                diag_load=DIAG_LOAD,
            )

            last_psr = psr_db

            if psr_db < PSR_MIN_DB:
                continue

            if spectra_acc is None:
                spectra_acc = P_mvdr.copy()
            else:
                spectra_acc += P_mvdr

            angles_list.append(theta_hat)
            valid_reads += 1

        if valid_reads == 0:
            return None, last_psr

        # Clustering de ángulos
        angles_arr = np.array(angles_list)
        best_count = 0
        best_mask = None

        for center in angles_arr:
            lower = center - WINDOW_WIDTH / 2.0
            upper = center + WINDOW_WIDTH / 2.0
            mask = (angles_arr >= lower) & (angles_arr <= upper)
            count = mask.sum()
            if count > best_count:
                best_count = count
                best_mask = mask

        cluster_angles = angles_arr[best_mask]
        doa_angle = float(cluster_angles.mean())

        return doa_angle, last_psr

    # ========================================
    # LOOP PRINCIPAL
    # ========================================
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print(f"[Pipeline] stop_event recibido, saliendo ({blade_name})")
                break

            iteration_count += 1
            iteration_start = time.time()



            node_lat_deg = gps_cache["node_lat"]
            node_lon_deg = gps_cache["node_lon"]
            node_src = gps_cache["node_src"]

            # ══════════════════════════════════════════════════════════════
            # FASE 1: VERIFICACIÓN DE DRONES CONOCIDOS (solo DOA)
            # ══════════════════════════════════════════════════════════════
            known_frequencies = drone_tracker.get_known_frequencies()

            if known_frequencies:
                time.sleep(0.5)
                print(f"\n[{blade_name}] ══ FASE VERIFICACIÓN ({len(known_frequencies)} drones) ══")

                for freq in known_frequencies:
                    drone = drone_tracker.get_drone_at_frequency(freq)
                    if drone is None:
                        continue

                    verify_start = time.time()
                    print(f"[{blade_name}] 🔍 Verificando {drone.drone_id} @ {freq / 1e6:.0f} MHz...", end=" ")

                    # Ejecutar solo DOA (rápido)
                    doa_angle, doa_psr = execute_doa_at_frequency(freq, NUM_READS_VERIFY)

                    if doa_angle is not None:
                        # DOA válido - actualizar tracker
                        time.sleep(0.5)
                        drone_tracker.update_doa_verification(freq, doa_angle, doa_psr)
                        print(f"✅ DOA: {doa_angle:.1f}° ({time.time() - verify_start:.2f}s)")

                        # Publicar DOA actualizado
                        publish_doa(
                            doa_angle,
                            blade_serial=blade_serial,
                            center_freq_hz=freq,
                            psr_db=doa_psr,
                        )
                        publish_drone_detection(drone, blade_name)

                        tri = tri_update_and_solve(
                            freq_hz=freq,
                            blade_serial=blade_serial,
                            doa_local_deg=doa_angle,
                            node_lat=node_lat_deg,
                            node_lon=node_lon_deg,
                            heading_deg=fixed_heading_deg
                        )
                        if tri:
                            print(
                                f"[{blade_name}] 📍 TRIANGULACIÓN @ {freq / 1e6:.0f} MHz → lat={tri.lat_deg:.7f}, lon={tri.lon_deg:.7f}")
                        #     # aquí ya tienes (lat, lon) del emisor
                    # else:
                    #     # DOA inválido - marcar fallo
                    #     was_removed = drone_tracker.mark_verification_failed(freq)
                    #     if was_removed:
                    #         print(f"❌ ELIMINADO (demasiados fallos)")
                    #     else:
                    #         print(f"⚠️  Fallo (PSR: {doa_psr:.1f} dB)")

            # ══════════════════════════════════════════════════════════════
            # FASE 2: BARRIDO (descubrimiento de nuevos drones)
            # ══════════════════════════════════════════════════════════════
            center_freq = scan_freqs[scan_index]
            scan_index = (scan_index + 1) % len(scan_freqs)

            # Detectar inicio de nuevo ciclo de barrido
            is_new_sweep = (scan_index == 0)
            if is_new_sweep and iteration_count > 1:
                print(f"\n[{blade_name}] ═══════════ NUEVO CICLO DE BARRIDO ═══════════")
                summary = drone_tracker.get_summary()
                if summary["total_active"] > 0:
                    print(f"[{blade_name}] 📊 Drones activos: {summary['total_active']} "
                          f"(confirmados: {summary['confirmed']}, tentative: {summary['tentative']})")
                    for d in summary["drones"]:
                        status_icon = "✅" if d["status"] == "confirmed" else "⏳"
                        doa_str = f"DOA: {d['doa_angle']:.1f}°" if d["doa_angle"] is not None else "DOA: N/A"
                        print(
                            f"    {status_icon} {d['drone_id']}: {d['drone_class']} @ {d['frequency_hz'] / 1e6:.0f} MHz | {doa_str}")

            scan_start = time.time()
            print(f"\n[{blade_name}] ══ FASE BARRIDO → {center_freq / 1e6:.0f} MHz (iter {iteration_count}) ══")

            # ========================================
            # Configurar SDR para single-channel
            # ========================================
            rx0.enable = False
            rx1.enable = False
            rx_ch.enable = False
            sdr.sync_config(
                layout=_bladerf.ChannelLayout.RX_X1,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=SDR_CONFIG["num_buffers"],
                buffer_size=SDR_CONFIG["buffer_size"],
                num_transfers=SDR_CONFIG["num_transfers"],
                stream_timeout=SDR_CONFIG["stream_timeout"],
            )

            for ch in (rx1, rx_ch):
                ch.frequency = center_freq

            rx1.enable = False
            rx_ch.enable = True

            # ========================================
            # Captura de muestras IQ
            # ========================================
            x = np.zeros(SDR_CONFIG["num_samples"], dtype=np.complex64)
            num_samples_read = 0

            while num_samples_read < SDR_CONFIG["num_samples"]:
                num = min(
                    len(buf) // SDR_CONFIG["bytes_per_sample"],
                    SDR_CONFIG["num_samples"] - num_samples_read,
                )
                sdr.sync_rx(buf, num)
                samples = np.frombuffer(buf, dtype=np.int16)
                samples = samples[0::2] + 1j * samples[1::2]
                samples /= 2048.0
                x[num_samples_read:num_samples_read + num] = samples[0:num]
                num_samples_read += num

            # ========================================
            # Espectrograma + Predicción
            # ========================================
            I = np.real(x)
            Q = np.imag(x)
            sample = torch.tensor(np.stack([I, Q], axis=0))
            spec = transform(sample)

            binary_results_list = binary_history_per_freq[center_freq]
            multiclass_results_list = multiclass_history_per_freq[center_freq]

            if TWO_STAGE_PREDICTION:
                (
                    final_class_name,
                    binary_result,
                    binary_result_smoothed,
                    multiclass_result,
                    multiclass_result_smoothed,
                    binary_results_list,
                    multiclass_results_list,
                ) = two_stage_predict(
                    binary_model,
                    multiclass_model,
                    spec,
                    BINARY_CONFIDENCE_THRESHOLD,
                    binary_results_list,
                    multiclass_results_list,
                    window_size=SCAN_WINDOW,
                )

                binary_history_per_freq[center_freq] = binary_results_list
                multiclass_history_per_freq[center_freq] = multiclass_results_list

                print(f"[{blade_name}] [Stage 1] {binary_result['class_name']} "
                      f"(conf={binary_result['confidence']:.4f})")

                if multiclass_result:
                    print(f"[{blade_name}] [Stage 2] {multiclass_result['class_name']} "
                          f"(conf={multiclass_result['confidence']:.4f})")

                drone_predict = final_class_name
                drone_confidence = (
                    binary_result_smoothed["confidence"]
                    if binary_result_smoothed is not None
                    else binary_result["confidence"]
                )

                print(f"[{blade_name}] [Final] {drone_predict} (conf={drone_confidence:.4f})")

            else:
                with torch.no_grad():
                    spec_input = spec.unsqueeze(0).unsqueeze(0)
                    outputs = model(spec_input)
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)

                drone_predict = CLASS_DICTS["multiclass"][preds.item()]
                drone_confidence = probs[0, preds.item()].item()
                print(f"[{blade_name}] [Prediction] {drone_predict} (conf={drone_confidence:.4f})")

            # ========================================
            # PSD → backend
            # ========================================
            publish_psd(x, center_freq, SDR_CONFIG["sample_rate"], nfft=1024, drone_id=drone_predict)

            # ========================================
            # Spectrogram → backend
            # ========================================
            try:
                spec_frame = build_spectrogram_frame(
                    spectrogram=spec.unsqueeze(0),
                    class_name=drone_predict,
                )
                publish_spec(spec_frame)
            except Exception as e:
                print("[pipeline] publish_spec error:", e)

            # ========================================
            # LÓGICA DE DETECCIÓN
            # ========================================
            drone_present = drone_predict not in ["Noise", "Jammer"]
            doa_angle = None
            doa_psr = None

            if drone_present:
                # Verificar si es un dron NUEVO o ya conocido
                existing_drone = drone_tracker.get_drone_at_frequency(center_freq)

                if existing_drone is None:
                    # NUEVO DRON - ejecutar DOA completo

                    print(f"[{blade_name}] 🎯 NUEVO DRON en {center_freq / 1e6:.0f} MHz - Ejecutando DOA...")
                    doa_angle, doa_psr = execute_doa_at_frequency(center_freq, NUM_READS_DISCOVER)

                    if doa_angle is not None:
                        print(f"[{blade_name}] [DOA] Ángulo: {doa_angle:.1f}°")
                        publish_doa(
                            doa_angle,
                            blade_serial=blade_serial,
                            center_freq_hz=center_freq,
                            psr_db=doa_psr,
                        )

                        tri = tri_update_and_solve(
                            freq_hz=center_freq,
                            blade_serial=blade_serial,
                            doa_local_deg=doa_angle,
                            node_lat=node_lat_deg,
                            node_lon=node_lon_deg,
                            heading_deg=fixed_heading_deg
                        )
                        if tri:
                            print(
                                f"[{blade_name}] 📍 TRIANGULACIÓN @ {freq / 1e6:.0f} MHz → lat={tri.lat_deg:.7f}, lon={tri.lon_deg:.7f}")
                            # aquí ya tienes (lat, lon) del emisor
                    else:
                        print(f"[{blade_name}] [DOA] Sin lecturas válidas")

                    # Agregar al tracker
                    detected_drone = drone_tracker.update_detection(
                        frequency_hz=center_freq,
                        drone_class=drone_predict,
                        confidence=drone_confidence,
                        doa_angle=doa_angle,
                        doa_psr_db=doa_psr
                    )
                    publish_drone_detection(detected_drone, blade_name)
                else:
                    # DRON CONOCIDO - solo actualizar predicción (DOA ya se hizo en fase verificación)
                    print(f"[{blade_name}] 📡 Dron conocido {existing_drone.drone_id} confirmado en barrido")
                    drone_tracker.update_detection(
                        frequency_hz=center_freq,
                        drone_class=drone_predict,
                        confidence=drone_confidence,
                        doa_angle=existing_drone.doa_angle,  # Mantener DOA existente
                        doa_psr_db=existing_drone.doa_psr_db
                    )

                # Publicar predicción
                publish_pred(
                    label_name=drone_predict,
                    confidence=drone_confidence,
                    binary_info=binary_result_smoothed if TWO_STAGE_PREDICTION else None,
                    multiclass_info=multiclass_result_smoothed if TWO_STAGE_PREDICTION else None,
                    center_freq=float(center_freq),
                    mode="hybrid_scan",
                )

                # Visualizar espectrograma
                label_for_plot = f"{drone_predict} ({drone_confidence:.2f}) @ {center_freq / 1e6:.0f}MHz"
                visualize_spectrogram(
                    spectrogram=spec.unsqueeze(0),
                    class_name=label_for_plot,
                    blade_id=blade_name,
                    **VIS_CONFIG,
                )
            else:
                # No hay dron - publicar Noise/Jammer
                publish_pred(
                    label_name=drone_predict,
                    confidence=drone_confidence,
                    binary_info=binary_result_smoothed if TWO_STAGE_PREDICTION else None,
                    multiclass_info=multiclass_result_smoothed if TWO_STAGE_PREDICTION else None,
                    center_freq=float(center_freq),
                    mode="hybrid_scan",
                )

            # ========================================
            # Limpieza periódica
            # ========================================
            if iteration_count % CLEANUP_INTERVAL == 0:
                lost_drones = drone_tracker.cleanup_lost_drones()
                if lost_drones:
                    print(f"[{blade_name}] 🧹 Limpieza: {len(lost_drones)} dron(es) perdidos por timeout")

            # ========================================
            # Publicar estado completo
            # ========================================
            # publish_multi_drone(drone_tracker, blade_name)

            # Tiempo total de iteración
            iteration_elapsed = time.time() - iteration_start
            scan_elapsed = time.time() - scan_start
            print(f"[{blade_name}] ⏱️  Barrido: {scan_elapsed:.2f}s | Total iteración: {iteration_elapsed:.2f}s")

    except Exception as e:
        import traceback
        print(f"[Pipeline] Error inesperado en {blade_name}: {e}")
        traceback.print_exc()
    finally:
        print(f"[Pipeline] Cerrando SDR ({blade_name})...")
        try:
            rx1.enable = False
            rx_ch.enable = False
            sdr.close()
            print(f"[Pipeline] SDR cerrado correctamente ({blade_name})")
        except Exception as e:
            print(f"[Pipeline] Error al cerrar SDR ({blade_name}): {e}")
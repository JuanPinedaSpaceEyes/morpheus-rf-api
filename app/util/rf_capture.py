# app/rf_capture.py
import threading
import time
import numpy as np
from typing import Optional, List, Tuple

import sys, os, shutil, subprocess
from pathlib import Path
from importlib import import_module, reload



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_realtime_module():
    mod = import_module("app.util.realtime_peaks_final")
    try:
        mod = reload(mod)
    except Exception:
        pass
    required = ["BladeRFCapture", "compute_psd", "find_peaks",
                "RX_FREQ", "SAMPLE_RATE", "FFT_SIZE", "BLOCK_SAMPLES",
                "THRESH_ABOVE_MED_DB", "MIN_SEP_HZ", "TOP_N"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        available = ", ".join(sorted([a for a in dir(mod) if not a.startswith("_")]))
        raise ImportError(
            f"El módulo realtime_peaks_final no define {missing}. "
            f"Disponibles: {available}."
        )
    return mod

_rt = _load_realtime_module()
BladeRFCapture       = getattr(_rt, "BladeRFCapture")
compute_psd          = getattr(_rt, "compute_psd")
find_peaks           = getattr(_rt, "find_peaks")
RX_FREQ              = float(getattr(_rt, "RX_FREQ"))
SAMPLE_RATE          = float(getattr(_rt, "SAMPLE_RATE"))
FFT_SIZE             = int(getattr(_rt, "FFT_SIZE"))
BLOCK_SAMPLES        = int(getattr(_rt, "BLOCK_SAMPLES"))
THRESH_ABOVE_MED_DB  = float(getattr(_rt, "THRESH_ABOVE_MED_DB"))
MIN_SEP_HZ           = float(getattr(_rt, "MIN_SEP_HZ"))
TOP_N                = int(getattr(_rt, "TOP_N"))

# ---------- Sonda de "power state" (encendido/presente) ----------
CLI_TIMEOUT_SEC = 3.0

def _in_flatpak() -> bool:
    return os.path.exists("/.flatpak-info") or "FLATPAK_ID" in os.environ or "/app" in os.environ.get("PATH", "")

def _spawn_host_prefix() -> List[str]:
    if _in_flatpak() and shutil.which("flatpak-spawn"):
        return ["flatpak-spawn", "--host"]
    return []

def _child_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = "/usr/local/bin:/usr/bin:/snap/bin:" + env.get("PATH", "")
    return env

def _resolve_cli_name() -> Optional[str]:
    env_cli = (os.environ.get("BLADERF_CLI") or "").strip()
    if env_cli and os.path.exists(env_cli):
        return os.path.basename(env_cli) if _in_flatpak() else env_cli
    search_path = "/usr/local/bin:/usr/bin:/snap/bin:" + os.environ.get("PATH", "")
    for name in ("bladeRF-cli", "bladerf-cli"):
        p = shutil.which(name, path=search_path)
        if p:
            return os.path.basename(p) if _in_flatpak() else p
    return None

def _probe_with_cli() -> Tuple[bool, str, str]:
    cli = _resolve_cli_name()
    if not cli:
        return False, "bladerf-cli", "CLI no encontrado en PATH ni BLADERF_CLI"
    argv = _spawn_host_prefix() + [cli, "-p"]
    try:
        p = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=_child_env(), timeout=CLI_TIMEOUT_SEC
        )
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out:

            if any(s in out.lower() for s in ("bladerf", "nuand", "serial", "product")) and \
               "no devices available" not in out.lower():
                return True, "bladerf-cli -p", out.splitlines()[0]
            if "no devices available" in out.lower():
                return False, "bladerf-cli -p", "No hay dispositivos disponibles"
        return False, "bladerf-cli -p", out or f"returncode={p.returncode}"
    except subprocess.TimeoutExpired:
        return False, "bladerf-cli -p", f"timeout>{CLI_TIMEOUT_SEC}s"
    except Exception as e:
        return False, "bladerf-cli -p", f"error: {e}"

def _probe_with_lsusb() -> Tuple[bool, str, str]:
    lsusb = shutil.which("lsusb") or "/usr/bin/lsusb"
    argv = _spawn_host_prefix() + [lsusb]
    try:
        p = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=_child_env(), timeout=CLI_TIMEOUT_SEC
        )
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out:

            if "2cf0:" in out.lower() or "nuand" in out.lower() or "bladerf" in out.lower():

                for line in out.splitlines():
                    if "2cf0:" in line.lower() or "nuand" in line.lower() or "bladerf" in line.lower():
                        return True, "lsusb", line.strip()
            return False, "lsusb", "No se encontró 2cf0/nuand/bladeRF"
        return False, "lsusb", out or f"returncode={p.returncode}"
    except subprocess.TimeoutExpired:
        return False, "lsusb", f"timeout>{CLI_TIMEOUT_SEC}s"
    except Exception as e:
        return False, "lsusb", f"error: {e}"

def probe_power_state() -> dict:
    ok, method, msg = _probe_with_cli()
    if ok or ("CLI no encontrado" not in msg and "timeout" not in msg and "error" not in msg):
        return {"powered": ok, "method": method, "message": msg}

    ok2, m2, msg2 = _probe_with_lsusb()
    return {"powered": ok2, "method": m2, "message": msg2}


class CaptureService:

    def __init__(self):
        self._cap = BladeRFCapture()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Banda activa inicial (por defecto: RX_FREQ y SAMPLE_RATE)
        self.center_hz = float(RX_FREQ)
        self.sample_rate = float(SAMPLE_RATE)

        self.block_id = 0
        self.last_block: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.configured = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="rf_capture_thread", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def set_band(self, center_hz: float, sample_rate: float) -> bool:
        """Cambia la banda activa y reconfigura el BladeRF."""
        self.center_hz = center_hz
        self.sample_rate = sample_rate
        self.configured = self._cap.configure(freq=center_hz, samplerate=sample_rate)
        return self.configured

    def _loop(self):
        # Configura la banda activa actual
        self.configured = self._cap.configure(freq=self.center_hz, samplerate=self.sample_rate)
        if not self.configured:
            self.last_error = "BladeRF no pudo configurarse. Revisa permisos/CLI."
            return

        while not self._stop.is_set():
            try:
                t0 = time.time()
                samples = self._cap.capture_block(BLOCK_SAMPLES)
                if samples is None or len(samples) == 0:
                    self.last_error = "Bloque vacío/None en captura"
                    time.sleep(0.1)
                    continue

                psd = compute_psd(samples, FFT_SIZE)
                peaks, noise_floor, max_power = find_peaks(
                    psd, self.sample_rate, self.center_hz,
                    THRESH_ABOVE_MED_DB, MIN_SEP_HZ, TOP_N
                )

                self.block_id += 1
                self.last_block = {
                    "block_id": self.block_id,
                    "capture_time_sec": time.time() - t0,
                    "noise_floor_db": float(noise_floor),
                    "max_power_db": float(max_power),
                    "peaks": [(float(f), float(p)) for (f, p) in peaks],
                    "psd": {
                        "start_hz": float(self.center_hz - self.sample_rate / 2),
                        "bin_hz": float(self.sample_rate / FFT_SIZE),
                        "bins": np.asarray(psd, dtype=np.float32).tolist(),
                    },
                }
                self.last_error = None

            except Exception as e:
                self.last_error = str(e)
                time.sleep(0.2)

    def get_status(self):
        return dict(
            configured=self.configured,
            blocks_processed=self.block_id,
            sample_rate_hz=self.sample_rate,
            center_freq_hz=self.center_hz,
            last_error=self.last_error,
        )

    def get_last_block(self):
        return self.last_block

    def get_power_state(self) -> dict:
        return probe_power_state()

    # helpers para FastAPI
    def get_status(self):
        # Usa la banda actual (self.band_index)
        band = self.scan_bands[self.band_index]
        return dict(
            configured=self.configured,
            blocks_processed=self.block_id,
            sample_rate_hz=band["sample_rate"],
            center_freq_hz=band["center_hz"],
            last_error=self.last_error,
        )

    def get_last_block(self):
        return self.last_block

    def get_power_state(self) -> dict:
        return probe_power_state()
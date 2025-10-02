# app/drone_detector_bladerf.py
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from collections import defaultdict
import threading
from typing import Optional, Tuple


# ANSI color codes para salida opcional (consola)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def bladerf_present() -> Tuple[bool, str]:
    """Devuelve (presente, primer_renglon_salida) usando `bladeRF-cli -p`."""
    try:
        p = subprocess.run(
            ["bladeRF-cli", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        if p.returncode == 0 and re.search(r"(bladeRF|nuand)", p.stdout, re.IGNORECASE):
            first = p.stdout.splitlines()[0] if p.stdout else "device found"
            return True, first
        return False, p.stdout.strip() or p.stderr.strip()
    except Exception as e:
        return False, f"err: {e}"


class DroneDetectorBladeRF:
    """
    Detector basado en tu script, adaptado para ejecutarse como servicio con bladeRF.

    Modo real:
      - Requiere un proceso externo (pipeline) que imprima líneas con este formato:
        [YYYY-mm-dd HH:MM:SS.mmm] <channel:int> <power:int> <MAC:AA:BB:CC:DD:EE:FF> <payload>
      - Si tu pipeline no incluye timestamp al inicio, se le inyecta uno automáticamente.

    Ejemplos de capture_cmd:
      - "python3 /opt/flows/ieee80211_probe_stdout.py --center 2.437e9 --samp-rate 20e6"
      - "/usr/bin/bash -lc 'mi_binario --opciones'"
    """
    def __init__(self, capture_cmd: Optional[str] = None):
        # OUIs de drones (tu lista)
        self.drone_ouis = {
            '60:60:1F': 'DJI Technology',
            '34:D2:62': 'DJI Technology',
            'A0:14:3D': 'DJI Technology',
            '90:3A:E6': 'DJI Technology',
            '7C:E9:D3': 'DJI Technology',
            '48:1C:B9': 'DJI Technology',
            'B0:E1:C5': 'DJI Technology',
            '00:00:00': 'G9 Drone',   # TODO: actualizar con OUI real si lo tienes
            'AC:DE:48': 'Generic Drone Vendor',
        }

        self.detected_devices = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'signal_strength': [],
            'channel': set(),
            'payload_samples': [],
            'packet_count': 0,
            'vendor': 'Unknown',
            'is_drone': False
        })

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._simulate = True
        self.start_time: Optional[datetime] = None

        # --- Nuevo: comando externo para captura real ---
        # Prioridad: parámetro > variable de entorno > None
        self.capture_cmd: Optional[str] = capture_cmd or os.getenv("DRONE_CAPTURE_CMD")
        # Proceso lanzado en modo real
        self._proc: Optional[subprocess.Popen] = None

    # ---------------- Lógica de parseo y actualización ----------------

    def parse_mac_from_payload(self, payload: str):
        mac_pattern = r'([0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5})'
        return re.findall(mac_pattern, payload)

    def identify_vendor(self, mac: str):
        oui = mac[:8].upper()
        for drone_oui, vendor in self.drone_ouis.items():
            if oui.startswith(drone_oui.upper()):
                return vendor, True
        return 'Unknown Vendor', False

    def update_device(self, mac: str, channel: int, power: int, payload: str, timestamp: str):
        vendor, is_drone = self.identify_vendor(mac)
        with self._lock:
            device = self.detected_devices[mac]
            device['vendor'] = vendor
            device['is_drone'] = is_drone
            device['last_seen'] = timestamp
            device['packet_count'] += 1
            if device['first_seen'] is None:
                device['first_seen'] = timestamp
                if is_drone:
                    self.alert_drone_detection(mac, vendor)
            device['channel'].add(channel)
            device['signal_strength'].append(power)
            if len(device['payload_samples']) < 5:
                device['payload_samples'].append(payload[:20])

    def alert_drone_detection(self, mac: str, vendor: str):
        # Mensaje a consola (no bloquea API)
        print(
            f"\n{Colors.RED}🚁 ¡DRONE DETECTADO!{Colors.ENDC} "
            f"{Colors.YELLOW}{mac}{Colors.ENDC} "
            f"{Colors.CYAN}{vendor}{Colors.ENDC} "
            f"{Colors.GREEN}{datetime.now().strftime('%H:%M:%S')}{Colors.ENDC}\n"
        )

    def process_capture_line(self, line: str):
        try:
            parts = line.strip().split()
            if len(parts) >= 6:
                # parts: [timestamp_parts] channel power mac payload
                timestamp = parts[0] + ' ' + parts[1] if len(parts[0]) > 0 else datetime.now().strftime('[%Y-%m-%d %H:%M:%S.%f]')
                channel = int(parts[2]) if parts[2].isdigit() else 0
                power_level = int(parts[3]) if parts[3].isdigit() else 0
                mac_address = parts[4]
                payload = parts[5] if len(parts) > 5 else ''
                if re.match(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$', mac_address):
                    self.update_device(mac_address, channel, power_level, payload, timestamp)
                # MACs adicionales en payload
                for mac_match in re.findall(r'([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})', payload):
                    if mac_match != mac_address:
                        self.update_device(mac_match, channel, power_level, payload, timestamp)
        except Exception:
            pass

    # ---------------- Captura (simulada por defecto) ----------------

    def _simulate_loop(self):
        sample_data = [
            "[2022-04-29 10:41:19.840]    81  1  01:E7:E7:E7:E7  F3",
            "[2022-04-29 10:41:19.848]    81  1  60:60:1F:AA:BB:CC  F7",  # DJI
            "[2022-04-29 10:41:19.864]    81  1  01:E7:E7:E7:E7  FB",
            "[2022-04-29 10:41:30.974]    82  1  34:D2:62:11:22:33  FF",  # DJI
            "[2022-04-29 10:41:38.333]    82  1  02:E7:E7:E7:E7  FF",
            "[2022-04-29 10:41:52.818]    82  1  AA:BB:CC:DD:EE:FF  FB",
        ]
        idx = 0
        while self._running.is_set():
            line = sample_data[idx % len(sample_data)]
            line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]" + line[line.find(']')+1:]
            self.process_capture_line(line)
            idx += 1
            time.sleep(0.1)

    # ---------------- Captura REAL: lee de un proceso externo ----------------

    def _bladerf_capture_loop(self):
        """
        Ejecuta `self.capture_cmd` y lee stdout línea a línea.
        Cada línea debe venir (o se le inyecta) en formato:
          [timestamp] <channel> <power> <MAC> <payload>
        """
        if not self.capture_cmd:
            print(f"{Colors.RED}[ERR]{Colors.ENDC} No se definió 'capture_cmd' para captura real. Usa el parámetro o DRONE_CAPTURE_CMD.")
            self._running.clear()
            return

        args = shlex.split(self.capture_cmd)
        print(f"{Colors.CYAN}[INFO]{Colors.ENDC} Iniciando proceso de captura: {self.capture_cmd}")

        try:
            self._proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # line-buffered
            )
        except Exception as e:
            print(f"{Colors.RED}[ERR]{Colors.ENDC} No pude iniciar el proceso de captura: {e}")
            self._running.clear()
            self._proc = None
            return

        try:
            assert self._proc.stdout is not None
            while self._running.is_set():
                line = self._proc.stdout.readline()
                if not line:
                    # Proceso terminó o no hay datos nuevos
                    if self._proc.poll() is not None:
                        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} Proceso de captura salió con código {self._proc.returncode}")
                        break
                    time.sleep(0.05)
                    continue

                # Asegurar timestamp al inicio si no viene incluido
                s = line.strip()
                if not s.startswith('['):
                    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {s}"
                self.process_capture_line(s)

        finally:
            # Cierre limpio del proceso si sigue vivo
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
            except Exception:
                pass
            self._proc = None

    # ---------------- API del servicio ----------------

    def start(self, simulate: bool = True) -> dict:
        present, detail = bladerf_present()
        if not simulate:
            # En modo real exigimos hardware y comando
            if not present:
                msg = f"bladeRF no detectado: {detail}"
                print(f"{Colors.RED}[ERR]{Colors.ENDC} {msg}")
                return {"running": False, "error": msg}
            if not self.capture_cmd:
                msg = "capture_cmd no definido. Usa parámetro en constructor o variable DRONE_CAPTURE_CMD."
                print(f"{Colors.RED}[ERR]{Colors.ENDC} {msg}")
                return {"running": False, "error": msg}

        if self._thread and self._thread.is_alive():
            return {"running": True, "message": "Detector ya está en ejecución"}

        self._simulate = simulate
        self._running.set()
        self.start_time = datetime.now()

        target = self._simulate_loop if simulate else self._bladerf_capture_loop
        self._thread = threading.Thread(target=target, name="DroneDetectorBladeRF", daemon=True)
        self._thread.start()
        return {"running": True, "mode": "simulation" if simulate else "real"}

    def stop(self) -> dict:
        self._running.clear()
        # Detener hilo
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

        # Detener proceso real si sigue activo
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None

        return {"running": False}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def summary(self) -> dict:
        with self._lock:
            runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
            drones = {mac: info for mac, info in self.detected_devices.items() if info['is_drone']}
            others = {mac: info for mac, info in self.detected_devices.items() if not info['is_drone']}
            # Resumen compacto para API
            simple_drones = []
            for mac, info in drones.items():
                avg_signal = (sum(info['signal_strength']) / len(info['signal_strength'])) if info['signal_strength'] else 0
                simple_drones.append({
                    "mac": mac,
                    "vendor": info['vendor'],
                    "packets": info['packet_count'],
                    "avg_signal": round(avg_signal, 1),
                    "channels": sorted(list(info['channel'])),
                    "last_seen": info['last_seen'],
                })
            return {
                "running": self.is_running(),
                "mode": "simulation" if self._simulate else "real",
                "runtime_seconds": int(runtime),
                "total_devices": len(self.detected_devices),
                "drones_count": len(drones),
                "others_count": len(others),
                "drones": simple_drones[:20],
            }

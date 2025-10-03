# app/drone_detector_bladerf.py
"""
Drone detector service for Morpheus RF API.

Adapted from the user-provided script: keeps simulation mode and a
'capture_cmd' hook for reading lines from an external pipeline (SoapySDR/GNU-Radio flow)
that emits textual lines with: [timestamp] <channel> <power> <MAC> <payload>.

Provides a class DroneDetectorBladeRF with methods expected by app/main.py:
 - start(simulate: bool=True) -> dict
 - stop() -> dict
 - is_running() -> bool
 - summary() -> dict
 - save_report() -> None

To run in real mode you must set the capture command either via the constructor
or the environment variable DRONE_CAPTURE_CMD (e.g. a python script that demodulates 802.11).
"""

from __future__ import annotations
import os
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime
from collections import defaultdict
from typing import Optional, Tuple

try:
    from tabulate import tabulate
except Exception:
    tabulate = None  # optional

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


def _bladerf_probe_cli() -> Tuple[bool, str]:
    """Simple probe for bladeRF CLI presence; used only to check device in real mode."""
    cmd = os.getenv("BLADERF_CLI") or shutil_which("bladeRF-cli") or shutil_which("bladerf-cli")
    if not cmd:
        return False, "bladeRF CLI not found"
    try:
        p = subprocess.run([cmd, "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        out = (p.stdout or p.stderr or "").strip()
        if p.returncode == 0 and ("bladeRF" in out or "nuand" in out.lower()):
            return True, out.splitlines()[0] if out else "bladeRF present"
        if "no devices available" in out.lower():
            return False, "no bladeRF devices available"
        return False, out or f"returncode={p.returncode}"
    except Exception as e:
        return False, f"error: {e}"

def shutil_which(name):
    """small wrapper to avoid importing shutil globally at top-level in some contexts"""
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None


class DroneDetectorBladeRF:
    """
    Detector adaptable a tu API:

    - simulate=True: run a built-in simulator.
    - simulate=False: runs an external capture command (DRONE_CAPTURE_CMD env var or constructor param)
      which must output lines in the expected format; otherwise the detector will error out.
    """

    def __init__(self, capture_cmd: Optional[str] = None):
        # OUIs de drones (extend as needed)
        self.drone_ouis = {
            '60:60:1F': 'DJI Technology',
            '34:D2:62': 'DJI Technology',
            'A0:14:3D': 'DJI Technology',
            '90:3A:E6': 'DJI Technology',
            '7C:E9:D3': 'DJI Technology',
            '48:1C:B9': 'DJI Technology',
            'B0:E1:C5': 'DJI Technology',
            '00:12:1C': 'Parrot SA',
            '90:03:B7': 'Parrot SA',
            '00:60:37': 'Skydio',
        }

        # OUIs comunes (no-drones) para mejor clasificación
        self.common_ouis = {
            '00:1B:63': 'Apple Inc.',
            '3C:15:C2': 'Apple Inc.',
            'AC:BC:32': 'Apple Inc.',
            '88:66:5A': 'Apple Inc.',
            '10:8C:CF': 'Samsung Electronics',
            'B8:27:EB': 'Raspberry Pi Foundation',
            'F0:D1:A9': 'Google Inc.',
            '48:D7:05': 'Amazon Technologies',
        }

        self.detected_devices = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'signal_strength': [],
            'frequency_or_channel': set(),
            'payload_samples': [],
            'packet_count': 0,
            'vendor': 'Unknown',
            'is_drone': False,
            'signal_type': 'Unknown'
        })

        # threading/sync primitives
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._simulate = True
        self.start_time: Optional[datetime] = None

        # capture command: constructor param > env var
        self.capture_cmd = capture_cmd or os.getenv("DRONE_CAPTURE_CMD")
        self._proc: Optional[subprocess.Popen] = None

        # stats
        self.total_packets = 0

    # ---------------- utilities ----------------
    def detect_signal_type(self, frequency_hz: str) -> str:
        try:
            freq_mhz = float(frequency_hz) / 1e6
            if 2400 <= freq_mhz <= 2500:
                return 'WiFi 2.4GHz/BT/Drones'
            if 5000 <= freq_mhz <= 6000:
                return 'WiFi 5GHz/Drones'
            if 900 <= freq_mhz <= 928:
                return 'ISM 900MHz/LoRa'
            return f'Unknown ({freq_mhz:.1f}MHz)'
        except Exception:
            return 'Unknown'

    def identify_vendor(self, mac: str) -> Tuple[str, bool]:
        oui = (mac or "")[:8].upper()
        for d_oui, vendor in self.drone_ouis.items():
            if oui.startswith(d_oui.upper()):
                return vendor, True
        for c_oui, vendor in self.common_ouis.items():
            if oui.startswith(c_oui.upper()):
                return vendor, False
        return 'Unknown Device', False

    # ---------------- parsing & updating ----------------
    def parse_mac_from_payload(self, payload: str):
        pattern = r'([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})'
        return [m.upper() for (m,) in re.findall(pattern, payload)]

    def update_device(self, mac: str, freq_or_channel: str, power: int, payload: str, timestamp: str):
        vendor, is_drone = self.identify_vendor(mac)
        with self._lock:
            self.total_packets += 1
            d = self.detected_devices[mac]
            d['vendor'] = vendor
            d['is_drone'] = is_drone
            d['last_seen'] = timestamp
            d['packet_count'] += 1
            if d['first_seen'] is None:
                d['first_seen'] = timestamp
                if is_drone:
                    self._print_alert(mac, vendor, timestamp)
            d['frequency_or_channel'].add(freq_or_channel)
            d['signal_strength'].append(int(power))
            if len(d['payload_samples']) < 5 and payload:
                d['payload_samples'].append(payload[:32])

    def _print_alert(self, mac: str, vendor: str, timestamp: str):
        print(
            f"\n{Colors.RED}🚁 DRONE DETECTADO{Colors.ENDC} "
            f"{Colors.YELLOW}{mac}{Colors.ENDC} "
            f"{Colors.CYAN}{vendor}{Colors.ENDC} "
            f"{Colors.GREEN}{timestamp}{Colors.ENDC}\n"
        )

    def process_capture_line(self, line: str):
        """
        Expected line forms:
         - "[2025-10-02 12:10:01.123] 2437000000 -45 60:60:1F:AA:BB:CC payload..."
         - or lines without timestamp, these will be prepended with current timestamp.
         - older format also accepted: timestamp split in two tokens; parser is forgiving.
        """
        try:
            if not line:
                return
            s = line.strip()
            if not s.startswith('['):
                s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {s}"

            # split tokens after timestamp
            # remove leading [timestamp]
            try:
                ts_end = s.index(']') + 1
                ts = s[1:ts_end-1]
                rest = s[ts_end:].strip()
            except ValueError:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                rest = s

            parts = rest.split(None, 4)  # up to 5 parts: freq/channel, power, mac, payload...
            if len(parts) >= 3:
                freq_or_channel = parts[0]
                power_s = parts[1]
                mac_candidate = parts[2].upper()
                payload = parts[3] if len(parts) > 3 else ''

                # normalize power
                try:
                    power = int(re.sub(r'[^\d\-]', '', power_s))
                except Exception:
                    power = 0

                # MAC validation
                if re.match(r'^([0-9A-F]{2}:){5}[0-9A-F]{2}$', mac_candidate):
                    self.update_device(mac_candidate, freq_or_channel, power, payload, ts)
                else:
                    # maybe MAC inside payload
                    macs = self.parse_mac_from_payload(payload)
                    for m in macs:
                        self.update_device(m, freq_or_channel, power, payload, ts)
        except Exception:
            # keep service robust
            return

    # ---------------- simulation loop ----------------
    def _simulate_loop(self):
        devices = [
            {'mac': '60:60:1F:AA:BB:CC', 'freq':'2437000000', 'power': -45, 'rate': 0.8},
            {'mac': '34:D2:62:11:22:33', 'freq':'5180000000', 'power': -52, 'rate': 0.6},
            {'mac': '3C:15:C2:A1:B2:C3', 'freq':'2437000000', 'power': -35, 'rate': 1.0},
            {'mac': '88:66:5A:D4:E5:F6', 'freq':'5180000000', 'power': -40, 'rate': 0.9},
            {'mac': '10:8C:CF:77:88:99', 'freq':'2462000000', 'power': -55, 'rate': 0.7},
            {'mac': 'B8:27:EB:98:76:54', 'freq':'2437000000', 'power': -65, 'rate': 0.3},
        ]
        import random
        idx = 0
        while self._running.is_set():
            device = random.choices(devices, weights=[d['rate'] for d in devices])[0]
            power = device['power'] + random.randint(-8, 8)
            line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {device['freq']} {power} {device['mac']} SIMPAYLOAD"
            self.process_capture_line(line)
            if idx % 20 == 0:
                # keep last state printed if someone runs interactively
                self._maybe_print_summary()
            idx += 1
            time.sleep(0.05)

    # ---------------- real capture loop (external process) ----------------
    def _capture_loop_from_cmd(self):
        if not self.capture_cmd:
            print(f"{Colors.RED}[ERR]{Colors.ENDC} capture_cmd no definido. Establece DRONE_CAPTURE_CMD o pasa capture_cmd al constructor.")
            self._running.clear()
            return

        args = shlex.split(self.capture_cmd)
        print(f"{Colors.CYAN}[INFO]{Colors.ENDC} Iniciando captura externa: {self.capture_cmd}")
        try:
            self._proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        except Exception as e:
            print(f"{Colors.RED}[ERR]{Colors.ENDC} No pude iniciar proceso de captura: {e}")
            self._running.clear()
            return

        try:
            assert self._proc.stdout is not None
            while self._running.is_set():
                line = self._proc.stdout.readline()
                if not line:
                    # process ended or nothing new
                    if self._proc.poll() is not None:
                        print(f"{Colors.YELLOW}[WARN]{Colors.ENDC} Proceso de captura finalizó (code {self._proc.returncode})")
                        break
                    time.sleep(0.05)
                    continue
                # ensure timestamp at start
                s = line.strip()
                if not s.startswith('['):
                    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {s}"
                self.process_capture_line(s)
        finally:
            # ensure process cleanup
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

    # ---------------- printing / summary / report ----------------
    def _maybe_print_summary(self):
        # minimal console summary for interactive sessions
        if tabulate is None:
            return
        with self._lock:
            drones = {m:i for m,i in self.detected_devices.items() if i['is_drone']}
            others = {m:i for m,i in self.detected_devices.items() if not i['is_drone']}
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BOLD}DroneDetector (sim={self._simulate}) running={self.is_running()}{Colors.ENDC}")
            print(f"Total packets: {self.total_packets} Devices: {len(self.detected_devices)} Drones: {len(drones)}")
            if drones:
                rows = []
                for mac, info in drones.items():
                    avg = sum(info['signal_strength'])/len(info['signal_strength']) if info['signal_strength'] else 0
                    rows.append([mac, info['vendor'], info['packet_count'], f"{avg:.1f}dBm", ",".join(info['frequency_or_channel'])])
                print(tabulate(rows, headers=['MAC','Vendor','Pkts','Avg','Ch/Freq']))
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

    def save_report(self) -> str:
        """Save a compact text report and return filename."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sdr_report_{timestamp}.txt"
        try:
            with open(filename, 'w') as f:
                f.write("SDR SPECTRUM ANALYZER REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Runtime: {(datetime.now() - self.start_time) if self.start_time else 0}\n")
                f.write(f"Total packets: {self.total_packets}\n")
                f.write(f"Total devices: {len(self.detected_devices)}\n\n")
                drones = {mac:info for mac,info in self.detected_devices.items() if info['is_drone']}
                f.write(f"DRONES: {len(drones)}\n")
                f.write("-"*80 + "\n")
                for mac, info in drones.items():
                    avg = sum(info['signal_strength'])/len(info['signal_strength']) if info['signal_strength'] else 0
                    f.write(f"MAC: {mac}\nVendor: {info['vendor']}\nPackets: {info['packet_count']}\nAvg signal: {avg:.1f} dBm\nChannels/freq: {','.join(info['frequency_or_channel'])}\n\n")
            print(f"{Colors.GREEN}✓ Report saved: {filename}{Colors.ENDC}")
            return filename
        except Exception as e:
            print(f"{Colors.RED}[ERR] saving report: {e}{Colors.ENDC}")
            return ""

    # ---------------- control API ----------------
    def start(self, simulate: bool = True) -> dict:
        """
        Start detector in background thread.
        If simulate is False, capture_cmd must be defined and bladeRF (or equivalent) must be present.
        """
        # if already running, return status
        if self._thread and self._thread.is_alive():
            return {"running": True, "message": "Detector already running", "mode": "simulation" if self._simulate else "real"}

        self._simulate = bool(simulate)
        # if real, validate presence
        if not self._simulate:
            # optional probe for bladeRF CLI presence
            ok, msg = _bladerf_probe_cli()
            if not ok:
                return {"running": False, "error": f"bladeRF probe failed: {msg}"}
            if not self.capture_cmd:
                return {"running": False, "error": "capture_cmd not defined (DRONE_CAPTURE_CMD env var or constructor param)"}

        # start thread
        self._running.set()
        self.start_time = datetime.now()
        worker = self._simulate_loop if self._simulate else self._capture_loop_from_cmd
        self._thread = threading.Thread(target=worker, name="DroneDetectorBladeRF", daemon=True)
        self._thread.start()
        return {"running": True, "mode": "simulation" if self._simulate else "real"}

    def stop(self) -> dict:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        # ensure subprocess is terminated
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
        # save a report snapshot
        self.save_report()
        return {"running": False}

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def summary(self) -> dict:
        with self._lock:
            runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            drones = {mac:info for mac,info in self.detected_devices.items() if info['is_drone']}
            others = {mac:info for mac,info in self.detected_devices.items() if not info['is_drone']}
            simple_drones = []
            for mac, info in drones.items():
                avg = (sum(info['signal_strength'])/len(info['signal_strength'])) if info['signal_strength'] else 0
                simple_drones.append({
                    "mac": mac,
                    "vendor": info['vendor'],
                    "packets": info['packet_count'],
                    "avg_signal": round(avg, 1),
                    "channels": sorted(list(info['frequency_or_channel'])),
                    "last_seen": info['last_seen'],
                })
            return {
                "running": self.is_running(),
                "mode": "simulation" if self._simulate else "real",
                "runtime_seconds": int(runtime),
                "total_devices": len(self.detected_devices),
                "drones_count": len(drones),
                "others_count": len(others),
                "drones": simple_drones[:50],
            }

# app/gps_capture.py
from __future__ import annotations

import time
from typing import List, Optional

import serial
from serial.tools import list_ports
import pynmea2

# ===== CONFIG =====
DEFAULT_PORT = "/dev/cu.PL2303G-USBtoUART10"
DEFAULT_BAUD = 4800

DEFAULT_FIX_TIMEOUT_S = 300        # wait for first valid fix
DEFAULT_N_SAMPLES = 5              # samples to average
DEFAULT_CAPTURE_TIMEOUT_S = 60     # time limit to gather N samples after fix
DEFAULT_SAMPLE_SPACING_S = 0.8     # avoid duplicate epoch messages
DEFAULT_STATUS_EVERY_S = 2         # progress while waiting for fix
# ==================


def _list_serial_ports() -> List[dict]:
    ports: List[dict] = []
    for p in list_ports.comports():
        ports.append(
            {
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "vid": getattr(p, "vid", None),
                "pid": getattr(p, "pid", None),
            }
        )
    return ports


def _looks_like_gps_port(info: dict) -> bool:

    dev = info["device"]
    desc = (info.get("description") or "").lower()
    hwid = (info.get("hwid") or "").lower()


    gps_keywords = ["gps", "gnss", "u-blox", "ublox"]
    usb_serial_keywords = ["pl2303", "usb-serial", "usb serial", "cp210", "ch340", "ch341"]

    if any(k in desc for k in gps_keywords):
        return True
    if any(k in desc for k in usb_serial_keywords):
        return True
    if "gps" in hwid or "gnss" in hwid:
        return True


    if dev.startswith("/dev/ttyUSB") or dev.startswith("/dev/ttyACM") or dev.startswith("/dev/cu."):
        return True
    if dev.upper().startswith("COM"):
        return True

    return False


def detect_gps_ports(
    baud: int = DEFAULT_BAUD,
    sniff_time_s: float = 3.0,
    read_timeout_s: float = 0.5,
) -> List[str]:
    candidates = [p for p in _list_serial_ports() if _looks_like_gps_port(p)]
    gps_ports: List[str] = []

    for info in candidates:
        dev = info["device"]
        try:
            with serial.Serial(dev, baud, timeout=read_timeout_s) as ser:
                start = time.time()
                while time.time() - start < sniff_time_s:
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    if not line:
                        continue

                    if line.startswith("$") and any(
                        tag in line for tag in ("GGA", "RMC", "GLL", "GSA", "GSV")
                    ):
                        print(f"[DETECT] {dev} parece GPS (NMEA: {line[:30]}...)")
                        gps_ports.append(dev)
                        break
        except (serial.SerialException, OSError):
            continue

    return gps_ports


def extract_fix(msg):
    st = getattr(msg, "sentence_type", "")

    # RMC: válido cuando status == 'A'
    if st == "RMC" and getattr(msg, "status", "") == "A":
        if msg.latitude and msg.longitude:
            return float(msg.latitude), float(msg.longitude)

    # GGA: válido cuando gps_qual > 0
    if st == "GGA":
        try:
            gps_qual = int(getattr(msg, "gps_qual", 0))
        except ValueError:
            gps_qual = 0
        if gps_qual > 0 and msg.latitude and msg.longitude:
            return float(msg.latitude), float(msg.longitude)

    return None


def capture_avg_fix(
    port: Optional[str] = None,
    baud: int = DEFAULT_BAUD,
    fix_timeout_s: float = DEFAULT_FIX_TIMEOUT_S,
    n_samples: int = DEFAULT_N_SAMPLES,
    capture_timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
    sample_spacing_s: float = DEFAULT_SAMPLE_SPACING_S,
    status_every_s: float = DEFAULT_STATUS_EVERY_S,
) -> dict:
    if port is None:
        gps_ports = detect_gps_ports(baud=baud)
        if not gps_ports:
            raise RuntimeError(
                "No se encontró ningún receptor GNSS. Verifica las conexiones USB "
                "o especifica un puerto manualmente."
            )
        if len(gps_ports) > 1:
            print(f"[INFO] Se encontraron múltiples GPS: {gps_ports}. Usando {gps_ports[0]}")
        port_to_use = gps_ports[0]
    else:
        port_to_use = port

    print(f"[INFO] Usando puerto {port_to_use} a {baud} baudios.")

    samples = []

    with serial.Serial(port_to_use, baud, timeout=1) as ser:
        start = time.time()
        last_status_print = 0.0
        first_fix = None

        while True:
            if time.time() - start > fix_timeout_s:
                raise TimeoutError("No GNSS fix. Muévete a cielo abierto e inténtalo de nuevo.")

            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue

            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            fix = extract_fix(msg)

            if getattr(msg, "sentence_type", "") == "GGA":
                now = time.time()
                if now - last_status_print > status_every_s:
                    last_status_print = now
                    try:
                        fixq = int(getattr(msg, "gps_qual", 0))
                    except ValueError:
                        fixq = 0
                    try:
                        sats = int(getattr(msg, "num_sats", 0))
                    except ValueError:
                        sats = 0
                    print(f"[WAIT] fixq={fixq} sats={sats}")

            if fix is not None:
                lat, lon = fix
                t = time.time()
                first_fix = (lat, lon, t)
                print(f"[READY] lat={lat:.7f}, lon={lon:.7f}")
                break

        lat, lon, t0 = first_fix
        samples.append({"lat": lat, "lon": lon, "timestamp": t0})
        last_t = t0
        capture_start = t0
        print(f"[CAPTURE] 1/{n_samples}")

        while len(samples) < n_samples:
            if time.time() - capture_start > capture_timeout_s:
                raise TimeoutError(f"Solo se obtuvieron {len(samples)}/{n_samples} muestras.")

            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue

            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            fix = extract_fix(msg)
            if fix is None:
                continue

            now = time.time()
            if now - last_t < sample_spacing_s:
                continue

            lat, lon = fix
            samples.append({"lat": lat, "lon": lon, "timestamp": now})
            last_t = now
            print(f"[CAPTURE] {len(samples)}/{n_samples}")

    lat_avg = sum(s["lat"] for s in samples) / len(samples)
    lon_avg = sum(s["lon"] for s in samples) / len(samples)

    print(f"[AVG] lat={lat_avg:.7f}, lon={lon_avg:.7f}")

    return {
        "avg": {"lat": lat_avg, "lon": lon_avg},
        "n_samples": len(samples)
    }


if __name__ == "__main__":
    try:
        result = capture_avg_fix()
        print(result)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

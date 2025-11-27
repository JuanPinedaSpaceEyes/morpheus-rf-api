# app/gps_capture.py
import time
import serial
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


def extract_fix(msg):
    """
    Dado un mensaje pynmea2, devuelve (lat, lon) en grados decimales si
    el mensaje tiene un fix válido; en caso contrario devuelve None.
    """
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
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
    fix_timeout_s: float = DEFAULT_FIX_TIMEOUT_S,
    n_samples: int = DEFAULT_N_SAMPLES,
    capture_timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
    sample_spacing_s: float = DEFAULT_SAMPLE_SPACING_S,
    status_every_s: float = DEFAULT_STATUS_EVERY_S,
) -> dict:
    """
    BLOQUEANTE.
    Abre el puerto serie, espera un fix GNSS, captura n_samples posiciones
    y devuelve un dict JSON-friendly con las muestras y el promedio.
    """
    samples = []

    with serial.Serial(port, baud, timeout=1) as ser:
        # ---- 1) ESPERAR FIX ----
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

            # Indicador de progreso (fix quality y satélites)
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

        # ---- 2) CAPTURAR N MUESTRAS + PROMEDIO ----
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
        "n_samples": len(samples),
        "port": port,
        "baud": baud,
        "samples": samples,
    }


if __name__ == "__main__":
    try:
        result = capture_avg_fix()
        print(result)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

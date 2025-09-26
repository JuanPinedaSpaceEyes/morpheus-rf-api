#!/usr/bin/env python3
"""
realtime_peaks_final.py
Versión final optimizada para tiempo real usando método de archivo temporal
Basado en el método que funciona en tu sistema.
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
import signal
from typing import List, Optional
import numpy as np

# =========================
#     CONFIGURACIÓN
# =========================
RX_FREQ = 5.500e9    # Centro de todo el espectro
SAMPLE_RATE = 20e6  # Máxima cobertura posible
FFT_SIZE = 4096
BLOCK_SAMPLES = 8192  # Tamaño probado que funciona

# Detección de picos
THRESH_ABOVE_MED_DB = 6.0  # dB sobre mediana (más sensible)
MIN_SEP_HZ = 300e3  # 200 kHz separación mínima
TOP_N = 10  # máximo picos por bloque

# Control de tiempo
UPDATE_INTERVAL_SEC = 1.0  # actualizar cada segundo
CLI_TIMEOUT_SEC = 4.0

# Configuración de visualización
SHOW_NOISE_FLOOR = True  # mostrar información del ruido de fondo
SHOW_BLOCK_STATS = True  # estadísticas por bloque
COMPACT_OUTPUT = False  # salida compacta o detallada


# =========================
#  UTILIDADES CLI
# =========================
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


def find_bladerf_cli() -> str:
    env_cli = (os.environ.get("BLADERF_CLI") or "").strip()
    if env_cli:
        return os.path.basename(env_cli) if _in_flatpak() else env_cli

    search_path = "/usr/local/bin:/usr/bin:/snap/bin:" + os.environ.get("PATH", "")
    for name in ("bladeRF-cli", "bladerf-cli"):
        p = shutil.which(name, path=search_path)
        if p:
            return os.path.basename(p) if _in_flatpak() else p

    if _in_flatpak():
        return "bladeRF-cli"

    home = os.path.expanduser("~")
    candidates = [
        "/usr/local/bin/bladeRF-cli", "/usr/bin/bladeRF-cli", "/snap/bin/bladeRF-cli",
        "/usr/local/bin/bladerf-cli", "/usr/bin/bladerf-cli", "/snap/bin/bladerf-cli",
        f"{home}/bladeRF/host/build/cli/src/bladeRF-cli",
        f"{home}/bladeRF/host/build/cli/src/bladerf-cli",
    ]
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError("No se encontró el CLI de bladeRF.")


def _cli_base_argv() -> list[str]:
    env_cli = (os.environ.get("BLADERF_CLI") or "").strip()
    cli = os.path.basename(env_cli) if env_cli else find_bladerf_cli()
    return _spawn_host_prefix() + [cli]


def _run_cli_once(expr: str, silent: bool = False) -> tuple[int, str]:
    env = _child_env()
    argv = _cli_base_argv() + ["-e", expr]
    if not silent:
        print(f"[cli] {expr}")
    try:
        p = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=CLI_TIMEOUT_SEC,
        )
        out = (p.stdout or "").strip()
        if not silent and p.returncode != 0:
            print(f"[cli][error={p.returncode}] {out}")
        return p.returncode, out
    except Exception as e:
        if not silent:
            print(f"[cli][exception] {e}")
        return 1, str(e)


# =========================
#   CAPTURA DE DATOS
# =========================
class BladeRFCapture:
    def __init__(self):
        self.configured = False
        self.temp_dir = tempfile.mkdtemp(prefix='bladerf_peaks_')
        print(f"Usando directorio temporal: {self.temp_dir}")

    def __del__(self):
        # Limpiar directorio temporal
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

    def configure(self, freq: float = RX_FREQ, samplerate: float = SAMPLE_RATE) -> bool:
        """Configura el bladeRF una sola vez"""
        # Limita razonablemente (evita 300e6)
        sr = int(min(max(samplerate, 1e6), 61_440_000))  # 1 MS/s .. 61.44 MS/s aprox
        bw = int(min(max(sr, 200_000), 56_000_000))  # BW RF ≈ SR clamped a ≤56 MHz

        # Intenta rx1; si falla, cae a rx
        expr_rx1 = (
            f"set frequency rx1 {int(freq)}; "
            f"set samplerate rx1 {sr}; "
            f"set bandwidth rx1 {bw}; "
            f"set agc rx1 on"
        )
        code, out = _run_cli_once(expr_rx1, silent=True)
        if code == 0 and "Invalid" not in out and "invalid" not in out:
            self.configured = True
            print(f"[OK] BladeRF configurado: {freq / 1e6:.1f} MHz, {sr / 1e6:.1f} MS/s, BW {bw / 1e6:.1f} MHz (rx1)")
            return True

        expr_rx = (
            f"set frequency rx {int(freq)}; "
            f"set samplerate rx {sr}; "
            f"set bandwidth rx {bw}; "
            f"set agc rx on"
        )
        code2, out2 = _run_cli_once(expr_rx, silent=True)
        if code2 == 0 and "Invalid" not in out2 and "invalid" not in out2:
            self.configured = True
            print(f"[OK] BladeRF configurado: {freq / 1e6:.1f} MHz, {sr / 1e6:.1f} MS/s, BW {bw / 1e6:.1f} MHz (rx)")
            return True

        print(f"[ERROR] Configuración falló:\n  rx1: {out}\n  rx:  {out2}")
        return False

    def capture_block(self, n_samples: int = BLOCK_SAMPLES) -> Optional[np.ndarray]:
        """Captura un bloque usando el método de archivo temporal que funciona"""
        if not self.configured:
            if not self.configure():
                return None

        # Crear archivo temporal único
        temp_file = os.path.join(self.temp_dir, f'capture_{int(time.time() * 1000)}.bin')

        try:
            # Comando de captura
            expr = f"rx config n={n_samples} file={temp_file}; rx start; rx wait"
            code, out = _run_cli_once(expr, silent=True)

            if code != 0:
                return None

            # Verificar que el archivo fue creado
            if not os.path.exists(temp_file):
                return None

            file_size = os.path.getsize(temp_file)
            expected_bytes = n_samples * 2 * 2  # SC16: 2 bytes I + 2 bytes Q

            if file_size < expected_bytes:
                return None

            # Leer datos SC16
            with open(temp_file, 'rb') as f:
                data = f.read(expected_bytes)

            # Convertir SC16 a complex64
            iq = np.frombuffer(data, dtype=np.int16).astype(np.float32).reshape(-1, 2)
            samples = (iq[:, 0] + 1j * iq[:, 1]) / 2048.0
            return samples.astype(np.complex64)

        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass


# =========================
#   PROCESADO DE SEÑAL
# =========================
def compute_psd(samples: np.ndarray, fft_size: int = FFT_SIZE) -> np.ndarray:
    """PSD con ventana Hanning, potencia y notch de DC."""
    s = samples[:fft_size] if len(samples) >= fft_size else np.pad(samples, (0, fft_size - len(samples)))
    windowed = s * np.hanning(len(s))
    fft = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    psd = 10 * np.log10(np.abs(fft)**2 + 1e-12)  # potencia en vez de módulo
    # notch DC (bin central)
    mid = len(psd)//2
    med = np.median(psd)
    psd[mid-1:mid+2] = med
    return psd


def find_peaks(psd: np.ndarray, fs: float, center_hz: float,
               thresh_above_med_db: float = THRESH_ABOVE_MED_DB,
               min_sep_hz: float = MIN_SEP_HZ, top_n: int = TOP_N):
    """Encuentra picos en el espectro"""
    med = np.median(psd)
    threshold = med + thresh_above_med_db

    # Máximos locales que superan el umbral
    locmax = np.zeros_like(psd, dtype=bool)
    locmax[1:-1] = (psd[1:-1] > psd[:-2]) & (psd[1:-1] > psd[2:]) & (psd[1:-1] >= threshold)
    idxs = np.flatnonzero(locmax)

    if idxs.size == 0:
        return [], med, np.max(psd)

    # Ordenar por potencia descendente
    idxs = idxs[np.argsort(psd[idxs])[::-1]]

    # Suprimir picos cercanos (NMS)
    bins_per_hz = len(psd) / fs
    min_sep_bins = max(1, int(min_sep_hz * bins_per_hz))
    selected = []
    taken = np.zeros_like(psd, dtype=bool)

    for i in idxs:
        if taken[i]:
            continue
        selected.append(i)
        lo = max(0, i - min_sep_bins)
        hi = min(len(psd), i + min_sep_bins + 1)
        taken[lo:hi] = True
        if len(selected) >= top_n:
            break

    # Convertir índices a frecuencias
    freqs_offset = np.linspace(-fs / 2, fs / 2, len(psd), endpoint=False)
    peaks = []
    for i in selected:
        f_hz = center_hz + freqs_offset[i]
        peaks.append((float(f_hz), float(psd[i])))

    # Ordenar por potencia
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks, med, np.max(psd)


# =========================
#   VISUALIZACIÓN
# =========================
def format_frequency(freq_hz: float) -> str:
    """Formatea frecuencia de manera legible"""
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.3f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.3f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.1f} kHz"
    else:
        return f"{freq_hz:.0f} Hz"


def print_peaks(peaks, noise_floor, max_power, block_num, capture_time):
    """Imprime los picos detectados"""
    timestamp = time.strftime("[%H:%M:%S]")

    if COMPACT_OUTPUT:
        # Formato compacto
        if peaks:
            freqs_str = ", ".join([format_frequency(f) for f, _ in peaks[:3]])
            print(f"{timestamp} #{block_num:4d} | {len(peaks)} picos | Top: {freqs_str}")
        else:
            print(f"{timestamp} #{block_num:4d} | Sin picos (ruido: {noise_floor:.1f} dB)")
    else:
        # Formato detallado
        print(f"\n{timestamp} ═══ Bloque #{block_num:4d} ═══")
        if SHOW_BLOCK_STATS:
            print(f"Tiempo captura: {capture_time:.2f}s | Ruido: {noise_floor:.1f} dB | Máx: {max_power:.1f} dB")

        if peaks:
            print(f"🎯 {len(peaks)} picos detectados:")
            for i, (f, p) in enumerate(peaks, 1):
                offset = (f - RX_FREQ) / 1e6
                snr = p - noise_floor
                print(f"   {i:2d}. {format_frequency(f):>12} │ {p:6.1f} dB │ SNR: {snr:5.1f} dB │ Δ{offset:+7.3f} MHz")
        else:
            print(f"💤 Sin picos sobre {THRESH_ABOVE_MED_DB} dB del ruido de fondo")


# =========================
#   BUCLE PRINCIPAL
# =========================
def main():
    # Banner
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🎯 BladeRF Real-Time Peak Detection 🎯" + " " * 19 + "║")
    print("╚" + "═" * 78 + "╝")

    # Configuración
    print(f"\n📡 Configuración:")
    print(f"   • Frecuencia central: {format_frequency(RX_FREQ)}")
    print(f"   • Ancho de banda: {format_frequency(SAMPLE_RATE)}")
    print(f"   • Resolución FFT: {FFT_SIZE} bins")
    print(f"   • Umbral detección: {THRESH_ABOVE_MED_DB} dB sobre mediana")
    print(f"   • Máximo picos: {TOP_N}")
    print(f"   • Actualización: cada {UPDATE_INTERVAL_SEC:.1f}s")

    # Inicializar capturador
    capture = BladeRFCapture()
    if not capture.configure():
        print("❌ Error: No se pudo configurar bladeRF")
        return 1

    print(f"\n🚀 Iniciando captura... (Ctrl+C para salir)")
    print("─" * 80)

    block_count = 0
    start_time = time.time()
    total_samples = 0

    try:
        while True:
            capture_start = time.time()

            # Capturar bloque
            samples = capture.capture_block(BLOCK_SAMPLES)

            if samples is None:
                print("⚠️  Error en captura, reintentando...")
                time.sleep(0.5)
                continue

            capture_time = time.time() - capture_start
            block_count += 1
            total_samples += len(samples)

            # Procesar
            psd = compute_psd(samples, FFT_SIZE)
            peaks, noise_floor, max_power = find_peaks(
                psd, SAMPLE_RATE, RX_FREQ,
                THRESH_ABOVE_MED_DB, MIN_SEP_HZ, TOP_N
            )

            # Mostrar resultados
            print_peaks(peaks, noise_floor, max_power, block_count, capture_time)

            # Estadísticas cada cierto tiempo
            if block_count % 20 == 0:
                elapsed = time.time() - start_time
                rate = total_samples / elapsed / 1e6
                print(
                    f"\n📊 Estadísticas: {block_count} bloques, {rate:.1f} MS/s promedio, {elapsed:.0f}s transcurridos")

            # Control de velocidad
            time.sleep(max(0, UPDATE_INTERVAL_SEC - capture_time))

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n🏁 Sesión terminada:")
        print(f"   • Bloques procesados: {block_count}")
        print(f"   • Muestras totales: {total_samples:,}")
        print(f"   • Tiempo total: {elapsed:.1f}s")
        print(f"   • Tasa promedio: {total_samples / elapsed / 1e6:.1f} MS/s")

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error fatal: {e}")
        sys.exit(1)
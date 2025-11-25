import os
import threading
import time

import numpy as np
from bladerf import _bladerf

# --- matplotlib para guardar imágenes (backend sin ventana) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUM_SAMPLES_BLOCK = 4096
BYTES_PER_SAMPLE = 4  # SC16_Q11 -> I(int16) + Q(int16)

# Carpeta donde se van a guardar las gráficas
PLOTS_DIR = "plots_dual_bladerf"


def _devinfo_serial_str(info) -> str:
    """
    Devuelve el serial de un DevInfo como string (decodificando bytes si hace falta).
    """
    s = info.serial
    if isinstance(s, (bytes, bytearray)):
        return s.decode("ascii", errors="ignore")
    return str(s)


class DualBladeRFService:
    def __init__(
        self,
        serial_24: str,
        serial_58: str,
        center_freq_24: float,
        center_freq_58: float,
        sample_rate: float,
        gain_db: int,
    ):
        """
        Servicio para controlar dos bladeRF:
        - Uno centrado en center_freq_24 (por ejemplo 2.4 GHz)
        - Otro centrado en center_freq_58 (por ejemplo 5.8 GHz)
        """

        # --- Configuración base ---
        self.serial_24 = serial_24.strip()
        self.serial_58 = serial_58.strip()
        self.center_freq_24 = center_freq_24  # Hz
        self.center_freq_58 = center_freq_58  # Hz
        self.sample_rate = sample_rate        # Hz
        self.gain_db = gain_db

        # --- Handles de dispositivos y canales (se llenan en init_devices) ---
        self.dev_24 = None
        self.dev_58 = None
        self.rx_24 = None
        self.rx_58 = None

        # --- Datos compartidos entre hilos ---
        self.shared_data = {"24": None, "58": None}
        self.last_power_db = {"24": None, "58": None}
        self.last_update_ts = {"24": None, "58": None}

        # Lock para proteger shared_data / last_power_db / last_update_ts
        self.lock = threading.Lock()

        # Evento para indicar a los hilos que deben detenerse
        self.stop_event = threading.Event()

        # Referencias a los hilos de recepción
        self.thread_24 = None
        self.thread_58 = None

        # Hilo para crear gráficas periódicas
        self.plot_thread = None
        self.plot_interval = 1.0  # segundos entre gráficas
        self.plot_counter = 0

        # Estado del servicio
        self.running = False

    # ------------------------------------------------------------------
    # Inicialización de dispositivos
    # ------------------------------------------------------------------

    def init_devices(self) -> None:
        """
        Abre los dos bladeRF por serial/prefijo y configura sus canales RX.
        NO arranca la captura todavía.
        """
        if self.dev_24 is not None and self.dev_58 is not None:
            print("[DualBladeRFService] Devices already initialized.")
            return

        devinfos = _bladerf.get_device_list()
        print("[DualBladeRFService] Available bladeRF devices:")
        for i, info in enumerate(devinfos):
            serial_str = _devinfo_serial_str(info)
            print(
                f"  {i}: backend={info.backend}, serial={serial_str}, "
                f"usb_bus={info.usb_bus}, usb_addr={info.usb_addr}, instance={info.instance}"
            )

        if len(devinfos) < 2:
            raise RuntimeError(
                "Less than 2 bladeRF devices detected. "
                "Connect both boards and restart the server."
            )

        # Abrir cada dispositivo por serial (o prefijo)
        self.dev_24 = self._open_device_by_serial_prefix(devinfos, self.serial_24)
        self.dev_58 = self._open_device_by_serial_prefix(devinfos, self.serial_58)

        # Configurar canales RX (sin habilitarlos todavía)
        self.rx_24 = self._configure_rx_channel(
            self.dev_24,
            freq_hz=self.center_freq_24,
            sample_rate_hz=self.sample_rate,
            gain_db=self.gain_db,
            channel_index=0,
        )
        self.rx_58 = self._configure_rx_channel(
            self.dev_58,
            freq_hz=self.center_freq_58,
            sample_rate_hz=self.sample_rate,
            gain_db=self.gain_db,
            channel_index=0,
        )

        print("[DualBladeRFService] Devices initialized and RX channels configured.")

    def _open_device_by_serial_prefix(self, devinfos, serial_prefix: str):
        """
        Busca en devinfos un DevInfo cuyo serial empiece por serial_prefix
        y lo abre con _bladerf.BladeRF(devinfo=...).
        """
        serial_prefix = serial_prefix.strip()
        for info in devinfos:
            serial_str = _devinfo_serial_str(info)
            if serial_str.startswith(serial_prefix):
                print(
                    f"[DualBladeRFService] Opening bladeRF with serial prefix "
                    f"'{serial_prefix}' ({serial_str}) ..."
                )
                try:
                    dev = _bladerf.BladeRF(devinfo=info)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not open bladeRF with serial '{serial_str}': {e}"
                    ) from e

                print(f"[DualBladeRFService] Device opened: {dev}")
                return dev

        raise RuntimeError(
            f"No bladeRF found with serial starting with '{serial_prefix}'"
        )

    def _configure_rx_channel(
        self,
        dev,
        freq_hz: float,
        sample_rate_hz: float,
        gain_db: int,
        channel_index: int = 0,
    ):
        """
        Configura parámetros del canal RX (frecuencia, Fs, BW, ganancia).
        NO configura el streaming síncrono ni habilita el canal.
        """
        rx_ch = dev.Channel(_bladerf.CHANNEL_RX(channel_index))

        print(
            f"[DualBladeRFService] Configuring RX{channel_index} at "
            f"{freq_hz / 1e6:.3f} MHz, Fs={sample_rate_hz / 1e6:.3f} MHz, "
            f"gain={gain_db} dB"
        )

        rx_ch.frequency = freq_hz
        rx_ch.sample_rate = sample_rate_hz
        rx_ch.bandwidth = sample_rate_hz / 2
        rx_ch.gain_mode = _bladerf.GainMode.Manual
        rx_ch.gain = gain_db

        rx_ch.enable = False
        return rx_ch

    def _configure_sync_stream(self, dev):
        """
        Configura sync_config para un dispositivo (RX_X1, SC16_Q11, etc.).
        """
        dev.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=NUM_SAMPLES_BLOCK,
            num_transfers=8,
            stream_timeout=3500,
        )

    # ------------------------------------------------------------------
    # Control de captura (start / stop)
    # ------------------------------------------------------------------

    def start_capture(self) -> None:
        """
        Configura el streaming síncrono en ambos dispositivos, habilita
        los canales RX y lanza los hilos de recepción y de guardado de gráficas.
        """
        if self.running:
            print("[DualBladeRFService] Capture already running.")
            return

        if self.dev_24 is None or self.dev_58 is None:
            raise RuntimeError("Devices are not initialized. Call init_devices() first.")
        if self.rx_24 is None or self.rx_58 is None:
            raise RuntimeError("RX channels are not configured.")

        # Reset del evento de parada y de las estructuras compartidas
        self.stop_event = threading.Event()
        with self.lock:
            self.shared_data = {"24": None, "58": None}
            self.last_power_db = {"24": None, "58": None}
            self.last_update_ts = {"24": None, "58": None}
        self.plot_counter = 0

        # Crear carpeta de plots si no existe
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Configurar streaming síncrono para ambos dispositivos
        self._configure_sync_stream(self.dev_24)
        self._configure_sync_stream(self.dev_58)

        # Habilitar canales RX
        self.rx_24.enable = True
        self.rx_58.enable = True

        # Crear y lanzar hilos de recepción
        self.thread_24 = threading.Thread(
            target=self._rx_worker, args=("24", self.dev_24), daemon=True
        )
        self.thread_58 = threading.Thread(
            target=self._rx_worker, args=("58", self.dev_58), daemon=True
        )

        self.thread_24.start()
        self.thread_58.start()

        # Crear y lanzar hilo de generación de gráficas
        self.plot_thread = threading.Thread(
            target=self._plot_worker, daemon=True
        )
        self.plot_thread.start()

        self.running = True
        print("[DualBladeRFService] Capture started (with plotting).")

    def stop_capture(self) -> None:
        """
        Señala a los hilos que se detengan, espera a que terminen y deshabilita
        los canales RX. NO cierra los dispositivos (se dejan abiertos).
        """
        if not self.running:
            print("[DualBladeRFService] Capture is not running.")
            return

        print("[DualBladeRFService] Stopping capture ...")
        self.stop_event.set()

        # Esperar a que terminen los hilos de RX
        for t in (self.thread_24, self.thread_58):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)

        # Esperar a que termine el hilo de plots
        if self.plot_thread is not None and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=2.0)

        # Deshabilitar canales RX
        if self.rx_24 is not None:
            try:
                self.rx_24.enable = False
            except Exception as e:
                print(f"[DualBladeRFService] Error disabling RX24: {e}")

        if self.rx_58 is not None:
            try:
                self.rx_58.enable = False
            except Exception as e:
                print(f"[DualBladeRFService] Error disabling RX58: {e}")

        self.running = False
        print("[DualBladeRFService] Capture stopped.")

    # ------------------------------------------------------------------
    # Hilos: recepción y plotting
    # ------------------------------------------------------------------

    def _rx_worker(self, key: str, dev) -> None:
        """
        Hilo de recepción para una de las bandas.
        key = "24" o "58".
        - Lee bloques IQ con sync_rx
        - Convierte a complejo normalizado
        - Calcula potencia media (dBFS aprox)
        - Actualiza shared_data, last_power_db y last_update_ts
        """
        buf = bytearray(NUM_SAMPLES_BLOCK * BYTES_PER_SAMPLE)

        while not self.stop_event.is_set():
            try:
                dev.sync_rx(buf, NUM_SAMPLES_BLOCK)
            except Exception as e:
                print(f"[DualBladeRFService] sync_rx error on {key}: {e}")
                self.stop_event.set()
                break

            # Convertir buffer -> int16 -> I/Q -> complejo
            samples_iq = np.frombuffer(buf, dtype=np.int16)
            i = samples_iq[0::2]
            q = samples_iq[1::2]
            x = i.astype(np.float32) + 1j * q.astype(np.float32)
            # Normalizar a ~[-1, 1] para SC16_Q11
            x /= 2048.0

            # Potencia media (lin y en dBFS aprox)
            power_lin = float(np.mean(np.abs(x) ** 2))
            power_db = 10.0 * np.log10(power_lin + 1e-12)
            now = time.time()

            # Actualizar estructuras compartidas
            with self.lock:
                self.shared_data[key] = x.copy()
                self.last_power_db[key] = power_db
                self.last_update_ts[key] = now

    def _plot_worker(self) -> None:

        print("[DualBladeRFService] Plot worker started.")
        os.makedirs(PLOTS_DIR, exist_ok=True)

        while not self.stop_event.is_set():
            if not self.running:
                break

            try:
                freqs, psd_24, psd_58 = self.compute_psd(nfft=1024)
            except Exception as e:

                time.sleep(0.5)
                continue


            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig.suptitle("Espectro dual bladeRF (2.4 GHz y 5.8 GHz)")

            ax1.plot(freqs, psd_24)
            ax1.set_ylabel("PSD 2.4 GHz (dB)")
            ax1.grid(True)
            ax1.set_ylim(-120, 0)

            ax2.plot(freqs, psd_58)
            ax2.set_ylabel("PSD 5.8 GHz (dB)")
            ax2.set_xlabel("Frecuencia relativa (MHz)")
            ax2.grid(True)
            ax2.set_ylim(-120, 0)

            # Nombre de archivo: timestamp + contador
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"dual_bladerf_{ts_str}_{self.plot_counter:04d}.png"
            filepath = os.path.join(PLOTS_DIR, filename)

            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"[DualBladeRFService] Saved plot: {filepath}")
            self.plot_counter += 1

            # Esperar antes de la próxima gráfica
            time.sleep(self.plot_interval)

        print("[DualBladeRFService] Plot worker stopped.")

    # ------------------------------------------------------------------
    # Utilidades para PSD (pueden usarse también desde otros endpoints)
    # ------------------------------------------------------------------

    def compute_psd(self, nfft: int = 1024):
        """
        Calcula el espectro (PSD) de las dos bandas usando las últimas
        muestras guardadas en shared_data.

        Devuelve:
            freqs_mhz: np.ndarray de frecuencias relativas en MHz
            psd_24: np.ndarray con PSD en dB para 2.4 GHz
            psd_58: np.ndarray con PSD en dB para 5.8 GHz
        """
        if not self.running:
            raise RuntimeError("Capture is not running.")

        # Copiamos fuera del lock para minimizar tiempo dentro de sección crítica
        with self.lock:
            x24 = None if self.shared_data["24"] is None else self.shared_data["24"].copy()
            x58 = None if self.shared_data["58"] is None else self.shared_data["58"].copy()

        if x24 is None or x58 is None:
            raise RuntimeError("No data available yet. Wait a moment and try again.")

        # Aseguramos longitud suficiente
        nfft = min(nfft, len(x24), len(x58))

        x24 = x24[:nfft]
        x58 = x58[:nfft]

        # Ventana de Hann
        w = np.hanning(nfft)

        # Eje de frecuencias relativo, centrado en 0, en MHz
        freqs = np.linspace(
            -self.sample_rate / 2, self.sample_rate / 2, nfft, endpoint=False
        ) / 1e6

        def _psd(x):
            X = np.fft.fftshift(np.fft.fft(x * w, n=nfft))
            return 20.0 * np.log10(np.abs(X) + 1e-12)

        psd_24 = _psd(x24)
        psd_58 = _psd(x58)

        return freqs, psd_24, psd_58

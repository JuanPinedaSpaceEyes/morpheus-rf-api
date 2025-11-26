import os
import sys
import ctypes

# Asegurar que libbladeRF esté visible para CFFI
lib_path = "/opt/homebrew/lib/libbladeRF.dylib"
if os.path.exists(lib_path):
    ctypes.cdll.LoadLibrary(lib_path)
    os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(lib_path) + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
else:
    print(f"⚠️ No se encontró la librería en {lib_path}")
    sys.exit(1)

from bladerf import _bladerf
import numpy as np
import time
import requests
import matplotlib.pyplot as plt
import torch
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Spectrogram
from scipy.signal import correlate
import shutil

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.routers.pipeline_router import psd_hub, make_psd_frame, pred_hub, doa_hub, spec_hub

# ================================================================================
# HELPERS PARA DEVINFO / SERIAL
# ================================================================================

def _devinfo_serial_str(info) -> str:
    """
    Devuelve el serial de un DevInfo como string, decodificando bytes si hace falta.
    """
    s = getattr(info, "serial", "")
    if isinstance(s, (bytes, bytearray)):
        return s.decode("ascii", errors="ignore")
    return str(s)

# ================================================================================
# CONFIGURATION
# ================================================================================

# ------- Two-Stage Prediction Configuration ---------------------------------
TWO_STAGE_PREDICTION = os.getenv("TWO_STAGE_PREDICTION", "1") == "1"
BINARY_CONFIDENCE_THRESHOLD = float(os.getenv("BINARY_CONFIDENCE_THRESHOLD", "0.4"))

# ------- Class Dictionaries --------------------------------------------------
CLASS_DICTS = {
    'binary': {0: 'Noise', 1: 'Drone'},
    'multiclass': {
        0: 'DJI Inspire 2',
        1: 'DJI Mini 4K',
        2: 'DJI Mavic 2 Air S',
        3: 'DJI Mavic Mini',
        4: 'DJI Mavic Pro',
        5: 'DJI Mavic Pro 2',
        6: 'DJI Phantom 4',
        7: 'Jammer',
        8: 'Parrot Disco'
    }
}

# ------- Configuración del Modelo -------------------------------------------------
MODEL_CONFIGS = {
    'binary': {
        'path': os.getenv("BINARY_MODEL_PATH",
                          "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced_BIN-UNF1-(IQSignal_DroneDetectSNR).pt")
    },
    'multiclass': {
        'path': os.getenv("MULTICLASS_MODEL_PATH",
                          "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced_MC-UNF5-(IQSig_DroneDetectSNR).pt")
    }
}

# ------- SDR Configuration ---------------------------------------------------
SDR_CONFIG = {
    'sample_rate': 40e6,
    'center_freq': 2440000000,
    'step_freq': 20000000,
    'gain': 30,
    'num_samples': int(3e6),
    'min_freq': 2400000000,
    'max_freq': 2480000000,
    'bandwidth_divisor': 2,
    'num_buffers': 16,
    'buffer_size': 8192,
    'num_transfers': 8,
    'stream_timeout': 3500,
    'bytes_per_sample': 4
}

# ------- Spectrogram Configuration -------------------------------------------
SPEC_CONFIG = {
    'n_fft': 512,
    'win_length': 512,
    'hop_length': 5860,
    'window_fn': torch.hann_window,
    'power': None,
    'normalized': False,
    'center': False,
    'onesided': False
}

# ------- Visualization Configuration -----------------------------------------
VIS_CONFIG = {
    'sample_freq': SDR_CONFIG['sample_rate'],
    'n_fft': SPEC_CONFIG['n_fft'],
    'win_length': SPEC_CONFIG['win_length'],
    'hop_length': SPEC_CONFIG['hop_length'],
    'pmin' : -50,
    'pmax' : 30,
    'figsize': (16, 4),
    'show_stats': True
}

# ------- Plot Saving Configuration -------------------------------------------
SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(os.getenv("PIPELINE_PLOT_DIR","/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/plots"))

# Rutas para "última" imagen (las que va a servir FastAPI)
LAST_SPEC_PATH = Path(os.getenv("PIPELINE_LAST_SPEC", "/tmp/morpheus_pipeline/last_spectrogram.png"))
LAST_DOA_PATH = Path(os.getenv("PIPELINE_LAST_DOA", "/tmp/morpheus_pipeline/last_doa.png"))

if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LAST_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_DOA_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------- API Endpoints -------------------------------------------------------
INGEST_URL = os.getenv("PSD_INGEST_URL", "http://127.0.0.1:8000/pipeline/psd/ingest")
PRED_INGEST_URL = os.getenv("PRED_INGEST_URL", "http://127.0.0.1:8000/pipeline/pred/ingest")
DOA_INGEST_URL = os.getenv("DOA_INGEST_URL", "http://127.0.0.1:8000/pipeline/doa/ingest")
SPEC_INGEST_URL = os.getenv("SPEC_INGEST_URL","http://127.0.0.1:8000/pipeline/spec/ingest")


# ------- DOA Configuration ---------------------------------------------------
DOA_CONFIG = {
    'speed_of_light': 299_792_458.0,
    'antenna_spacing_cm': 6.14,
    'block_size': 4096,
    'angles': np.linspace(-90, 90, 721),
    'num_samples': 4096,
    'num_buffers': 32,
    'buffer_size': 8192,
    'num_transfers': 16,
    'dynamic_range_db': 15,
    'military_green': "#4B9920"
}

# ================================================================================
# BLADERF DEVICE INITIALIZATION
# ================================================================================

os.environ.setdefault("LIBUSB_DEBUG", "3")
os.environ.pop("LIBUSB_DEBUG", None)
DEV_ID = os.getenv("BLADERF_DEVICE")
DEVICE_NAME = os.getenv("PIPELINE_DEVICE_NAME", DEV_ID or "default_blade")


def _open_bladerf_from_env(dev_id: str | None):
    """
    Abre un bladeRF usando BLADERF_DEVICE como PREFIJO de serial.
    Si dev_id es None → abre el primer dispositivo disponible.
    """
    try:
        devinfos = _bladerf.get_device_list()
    except Exception as e:
        print(f"[SDR] Error llamando a get_device_list(): {e}")
        sys.exit(2)

    if not devinfos:
        print("[SDR] No se encontraron dispositivos bladeRF en get_device_list().")
        try:
            out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
            print("[SDR] bladeRF-cli -p stdout:\n", out.stdout)
            print("[SDR] bladeRF-cli -p stderr:\n", out.stderr)
        except Exception as e:
            print("[SDR] No pude ejecutar bladeRF-cli -p:", e)
        sys.exit(2)

    if not dev_id:
        sys.exit(1)

    serial_prefix = dev_id.strip()
    print(f"[SDR] BLADERF_DEVICE='{serial_prefix}', buscando por prefijo de serial...")

    for info in devinfos:
        serial_str = _devinfo_serial_str(info)
        print(f"   - encontrado dispositivo con serial={serial_str}")
        if serial_str.startswith(serial_prefix):
            print(f"[SDR] Abriendo bladeRF con serial que empieza por '{serial_prefix}'")
            try:
                dev = _bladerf.BladeRF(devinfo=info)
                print(f"[SDR] Dispositivo abierto: {dev}")
                return dev
            except Exception as e:
                print(f"[SDR] Error abriendo dispositivo con ese serial: {e}")
                sys.exit(1)

    print(f"[SDR] No se encontró ningún dispositivo cuyo serial empiece por '{serial_prefix}'")
    try:
        out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
        print("[SDR] bladeRF-cli -p stdout:\n", out.stdout)
        print("[SDR] bladeRF-cli -p stderr:\n", out.stderr)
    except Exception as e:
        print("[SDR] No pude ejecutar bladeRF-cli -p:", e)
    sys.exit(2)

try:
    sdr = _open_bladerf_from_env(DEV_ID)
    print("[SDR] BladeRF device initialized successfully")
except SystemExit:
    raise
except Exception as e:
    print(f"[SDR] Error inesperado inicializando bladeRF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================================================================================
# MODEL LOADING
# ================================================================================

try:
    if TWO_STAGE_PREDICTION:
        binary_model = torch.jit.load(MODEL_CONFIGS['binary']['path'], map_location="cpu")
        binary_model.eval()
        print(f"[Model] Modelo binario cargado: {MODEL_CONFIGS['binary']['path']}")

        multiclass_model = torch.jit.load(MODEL_CONFIGS['multiclass']['path'], map_location="cpu")
        multiclass_model.eval()
        print(f"[Model] Modelo multiclase cargado: {MODEL_CONFIGS['multiclass']['path']}")
    else:
        # Legacy single model mode
        model = torch.jit.load(MODEL_CONFIGS['multiclass']['path'], map_location="cpu")
        model.eval()
        print(f"[Model] Modelo único cargado: {MODEL_CONFIGS['multiclass']['path']}")
except Exception as e:
    print(f"[Model] Error cargando modelos: {e}")
    sys.exit(1)


# ================================================================================
# FUNCTION DEFINITIONS
# ================================================================================

# ------- PSD Computation -----------------------------------------------------
def _compute_psd_db(x: np.ndarray, nfft: int = 4096) -> np.ndarray:
    """Compute Power Spectral Density in dB."""
    if x.ndim != 1:
        raise ValueError("x debe ser 1D complejo")
    seg = x[:nfft]
    if seg.shape[0] < nfft:
        seg = np.pad(seg, (0, nfft - seg.shape[0]))
    w = np.hanning(nfft)
    X = np.fft.fftshift(np.fft.fft(seg * w, n=nfft))
    return 10.0 * np.log10(np.abs(X) + 1e-12)


def publish_psd(x_complex: np.ndarray, center_hz: float, sample_rate: float,
                nfft: int = 4096, drone_id: str | None = None):
    """Publish PSD data to backend."""
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

def publish_doa(angle_deg: float):
    body = {
        "angle_deg": float(angle_deg),
    }
    try:
        requests.post(DOA_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_doa HTTP error:", e)

def build_spectrogram_frame(spectrogram: torch.Tensor, class_name: str) -> dict:
    spec_np = spectrogram.detach().cpu().numpy()  # (C, F, T)
    C, F, T = spec_np.shape

    pmin = float(np.nanmin(spec_np))
    pmax = float(np.nanmax(spec_np))

    frame = {
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
    return frame



# ------- Spectrogram Transform -----------------------------------------------
class transform_spectrogram(torch.nn.Module):
    """Transform IQ signal to spectrogram."""

    def __init__(self, device, n_fft=512, win_length=512, hop_length=5860,
                 window_fn=torch.hann_window, power=None, normalized=False,
                 center=False, onesided=False):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            window_fn=window_fn, power=power, normalized=normalized,
            center=center, onesided=onesided
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


# ------- Two-Stage Prediction ------------------------------------------------
def two_stage_predict(binary_model, multiclass_model, spectrogram, binary_threshold=0.5):
    """
    Perform two-stage prediction:
    1. Binary classification (Drone vs Noise)
    2. If Drone detected, perform multiclass classification

    Returns:
        final_class_name: str - The final predicted class
        binary_result: dict - Binary classification results
        multiclass_result: dict or None - Multiclass classification results
    """
    spec_input = spectrogram.type(torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # Stage 1: Binary prediction
        binary_outputs = binary_model(spec_input).view(-1)
        binary_probs = torch.sigmoid(binary_outputs)
        binary_pred = (binary_probs > binary_threshold).float()
        binary_class_idx = int(binary_pred.item())
        binary_class_name = CLASS_DICTS['binary'][binary_class_idx]
        binary_confidence = binary_probs.item() if binary_class_idx == 1 else 1 - binary_probs.item()

    binary_result = {
        'class_name': binary_class_name,
        'confidence': binary_confidence,
        'raw_prob': binary_probs.item()
    }

    # Stage 2: Multiclass classification (only if Drone detected)
    multiclass_result = None
    final_class_name = binary_class_name

    if binary_class_name == 'Drone':
        with torch.no_grad():
            multiclass_outputs = multiclass_model(spec_input)
            multiclass_probs = torch.softmax(multiclass_outputs, dim=1)
            multiclass_pred = multiclass_outputs.argmax(dim=1)
            multiclass_class_idx = int(multiclass_pred.item())
            multiclass_class_name = CLASS_DICTS['multiclass'][multiclass_class_idx]
            multiclass_confidence = multiclass_probs[0, multiclass_class_idx].item()

        multiclass_result = {
            'class_name': multiclass_class_name,
            'confidence': multiclass_confidence,
            'class_idx': multiclass_class_idx,
            'all_probs': multiclass_probs[0].cpu().numpy().tolist()
        }
        final_class_name = multiclass_class_name

    return final_class_name, binary_result, multiclass_result


# ------- Visualization -------------------------------------------------------
def visualize_spectrogram(spectrogram: torch.Tensor, class_name: str,
                          blade_id: str,n_fft=512, win_length=512, hop_length=5860,
                          sample_freq=40e6, pmin: float = None, pmax: float = None,
                          show_stats: bool = True, figsize: tuple = (16, 4)) -> None:
    """Visualize a spectrogram with I and Q channels or a Power spectrogram."""
    # Determine layout based on spectrogram shape
    if blade_id is None:
        blade_id = DEVICE_NAME


    if spectrogram.ndim == 3:
        if spectrogram.shape[0] == 2:
            ch_names = {0: "I Spectrogram", 1: "Q Spectrogram"}
            fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor = "none")
            axes.patch.set_alpha(0.0)
            axes.spines['bottom'].set_color('white')
            axes.spines['left'].set_color('white')
            fig.patch.set_alpha(0)
            fig.suptitle(f'{class_name} | {blade_id} ' )
        elif spectrogram.shape[0] == 1:
            ch_names = {0: f"Power Spectrogram"}
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes.patch.set_alpha(0.0)
            axes.spines['bottom'].set_color('white')
            axes.spines['left'].set_color('white')
            axes = [axes]  # Make it iterable
            fig.patch.set_alpha(0)
            fig.suptitle(f'{blade_id}')
        else:
            raise ValueError(f"Spectrogram shape {spectrogram.shape} doesn't match expected dimensions")
    else:
        raise ValueError(f"Expected 3D spectrogram, got shape {spectrogram.shape}")

    print(f"[Visualización] Forma del espectrograma: {spectrogram.shape}")
    print(f"[Visualización] Clase: {class_name} (device={blade_id})")

    # Convert to numpy for matplotlib
    spectrogram_np = spectrogram.detach().cpu().numpy()
    n_freq_bins = spectrogram_np.shape[1]
    n_time_bins = spectrogram_np.shape[2]

    # Create time axis in milliseconds
    time_bin_duration = hop_length / sample_freq  # in seconds
    time_duration_ms = np.arange(n_time_bins) * time_bin_duration * 1e3  # convert to ms

    # Create frequency axis
    freqs = np.fft.fftfreq(n_fft, 1 / sample_freq)
    sorted_indices = np.argsort(freqs)
    freqs = freqs[sorted_indices]
    freqs_in_mhz = freqs / 1e6

    # Set color scale limits
    if pmin is None:
        pmin = np.nanmin(spectrogram_np)
    if pmax is None:
        pmax = np.nanmax(spectrogram_np)

    # Ensure frequency axis matches spectrogram dimensions
    if len(freqs_in_mhz) != n_freq_bins:
        print(f"[Visualización] Advertencia: Desajuste en bins de frecuencia. Esperados {n_freq_bins}, obtenidos {len(freqs_in_mhz)}")
        freqs_in_mhz = np.linspace(freqs_in_mhz[0], freqs_in_mhz[-1], n_freq_bins)

    t = time_duration_ms  # Time in milliseconds
    f = freqs_in_mhz  # Frequency in MHz

    # Create visualization for each channel
    for ch in range(spectrogram_np.shape[0]):

        # Use imshow for faster rendering


        im = axes[ch].imshow(
            spectrogram_np[ch],
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            interpolation='nearest',
            vmin=pmin,
            vmax=pmax
        )
        # Set x-axis ticks in milliseconds
        max_time = time_duration_ms[-1] if len(time_duration_ms) > 0 else 75
        axes[ch].set_xticks(np.arange(0, max_time, step=10), )
        axes[ch].tick_params(axis='x', colors='white')
        axes[ch].tick_params(axis='y', colors='white')

    plt.tight_layout()

    # Print statistics if requested
    if show_stats:
        print(f"[Visualización] Estadísticas del espectrograma:")
        print(f"   Min: {np.nanmin(spectrogram_np):.2f} dB")
        print(f"   Max: {np.nanmax(spectrogram_np):.2f} dB")
        print(f"   Mean: {np.nanmean(spectrogram_np):.2f} dB")
        print(f"   Std: {np.nanstd(spectrogram_np):.2f} dB")

    if SAVE_PLOTS:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = PLOT_DIR / f"spectrogram_{ts}.png"
            fig.savefig(out_path, dpi=120)
            print(f"[Visualización] Espectrograma guardado: {out_path}")

            # Copiamos a una ruta fija que servirá el endpoint
            shutil.copy2(out_path, LAST_SPEC_PATH)
            print(f"[Visualización] Último espectrograma actualizado: {LAST_SPEC_PATH}")
        except Exception as e:
            print("[Visualización] savefig error:", e)

    plt.close(fig)


# ------- DOA Functions -------------------------------------------------------
def steering_vector(theta_deg, M, d_lambda):
    """Compute steering vector for given angle."""
    m = np.arange(M)[:, None]
    return np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(np.deg2rad(theta_deg)))


def music_block(Xb, angles, d_lambda, num_expected_signals=1, diag_load=1e-6):
    """MUSIC algorithm for DOA estimation."""
    Xb = np.asarray(Xb, dtype=np.complex128)
    K, M = Xb.shape

    R = (Xb.conj().T @ Xb) / max(1, K)
    R += (diag_load * (np.trace(R).real / M)) * np.eye(M)

    J = np.fliplr(np.eye(R.shape[0]))
    R = 0.5 * (R + J @ R.conj() @ J)

    w, v = np.linalg.eigh(R)
    ratio = (w[-1] / max(w[0], 1e-12)).real
    if ratio < 2:
        return np.ones(len(angles))

    w, v = np.linalg.eigh(R)
    d = int(np.clip(num_expected_signals, 0, M - 1))
    Vn = v[:, :M - d]
    Pn = Vn @ Vn.conj().T

    ang = np.asarray(angles, dtype=float)
    P = np.empty(ang.size, dtype=float)
    idx = np.arange(M).reshape(-1, 1)

    for i, th_deg in enumerate(ang):
        th = np.deg2rad(th_deg)
        a = np.exp(-2j * np.pi * d_lambda * idx * np.sin(th))
        denom = np.real((a.conj().T @ Pn @ a)[0, 0])
        P[i] = 1.0 / max(denom, 1e-12)

    return P


# ------- Prediction Publishing -----------------------------------------------
def publish_pred(label_name: str, confidence: float = None,
                 binary_info: dict = None, multiclass_info: dict = None):
    """Publish prediction with enhanced metadata for two-stage prediction."""
    body = {
        "label": label_name,
        "timestamp": time.time(),
    }

    if confidence is not None:
        body["confidence"] = confidence

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

def publish_spec(frame: dict):
    try:
        requests.post(SPEC_INGEST_URL, json=frame, timeout=0.7)
    except Exception as e:
        print("[pipeline] publish_spec HTTP error:", e)


# ================================================================================
# BLADERF CHANNEL CONFIGURATION
# ================================================================================

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))
rx1 = sdr.Channel(_bladerf.CHANNEL_RX(0))

# Apply SDR configuration
for ch in (rx1, rx_ch):
    ch.frequency = SDR_CONFIG['center_freq']
    ch.sample_rate = SDR_CONFIG['sample_rate']
    ch.bandwidth = SDR_CONFIG['sample_rate'] / SDR_CONFIG['bandwidth_divisor']
    ch.gain_mode = _bladerf.GainMode.Manual
    ch.gain = SDR_CONFIG['gain']

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=SDR_CONFIG['num_buffers'],
    buffer_size=SDR_CONFIG['buffer_size'],
    num_transfers=SDR_CONFIG['num_transfers'],
    stream_timeout=SDR_CONFIG['stream_timeout']
)

rx1.enable = False
rx_ch.enable = True

buf = bytearray(1024 * SDR_CONFIG['bytes_per_sample'])

# ================================================================================
# TRANSFORM AND DOA INITIALIZATION
# ================================================================================

transform = transform_spectrogram(
    device="cpu",
    **SPEC_CONFIG
)

# DOA variables
c = DOA_CONFIG['speed_of_light']
fc_hz = SDR_CONFIG['center_freq']
d_cm = DOA_CONFIG['antenna_spacing_cm']
lam = c / fc_hz
d_lambda = (d_cm / 100.0) / lam
block_size = DOA_CONFIG['block_size']
angles = DOA_CONFIG['angles']

# Frequency hopping variables
center_freq = SDR_CONFIG['center_freq']
direction = 1

# ================================================================================
# MAIN PROCESSING LOOP
# ================================================================================

print(f"[Pipeline] Iniciando recepción (Two-stage: {TWO_STAGE_PREDICTION})")

try:
    while True:
        start_time = time.time()
        print(f"\n[FRECUENCIA]: {center_freq} Hz")

        # Reset channel configuration for single RX
        rx1.enable = False
        rx_ch.enable = False
        sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=SDR_CONFIG['num_buffers'],
            buffer_size=SDR_CONFIG['buffer_size'],
            num_transfers=SDR_CONFIG['num_transfers'],
            stream_timeout=SDR_CONFIG['stream_timeout']
        )
        rx1.enable = False
        rx_ch.enable = True
        print("corono ---------->")
        # Receive samples
        x = np.zeros(SDR_CONFIG['num_samples'], dtype=np.complex64)
        num_samples_read = 0

        while num_samples_read < SDR_CONFIG['num_samples']:
            num = min(len(buf) // SDR_CONFIG['bytes_per_sample'],
                      SDR_CONFIG['num_samples'] - num_samples_read)
            sdr.sync_rx(buf, num)
            samples = np.frombuffer(buf, dtype=np.int16)
            samples = samples[0::2] + 1j * samples[1::2]
            samples /= 2048.0
            x[num_samples_read:num_samples_read + num] = samples[0:num]
            num_samples_read += num

        # Generate spectrogram
        I = np.real(x)
        Q = np.imag(x)
        sample = torch.tensor(np.stack([I, Q], axis=0))
        spec = transform(sample)

        # ============ PREDICTION ============
        if TWO_STAGE_PREDICTION:
            drone_predict, binary_result, multiclass_result = two_stage_predict(
                binary_model, multiclass_model, spec, BINARY_CONFIDENCE_THRESHOLD
            )

            print(f"\n[Stage 1 - Binary] {binary_result['class_name']} (conf: {binary_result['confidence']:.4f})")
            if multiclass_result:
                print(f"[Stage 2 - Multiclass] {multiclass_result['class_name']} (conf: {multiclass_result['confidence']:.4f})")
            print(f"[Final] {drone_predict}")

            # Publish with detailed info
            publish_pred(
                label_name=drone_predict,
                confidence=multiclass_result['confidence'] if multiclass_result else binary_result['confidence'],
                binary_info=binary_result,
                multiclass_info=multiclass_result
            )
        else:
            # Legacy single model prediction
            with torch.no_grad():
                spec_input = spec.unsqueeze(0).unsqueeze(0)
                outputs = model(spec_input)
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)

            drone_predict = CLASS_DICTS['multiclass'][preds.item()]
            confidence = probs[0, preds.item()].item()
            print(f"[Prediction] {drone_predict} (conf: {confidence:.4f})")

            publish_pred(label_name=drone_predict, confidence=confidence)

        frame = make_psd_frame(
            x=x,
            center_hz=center_freq,
            sample_rate=SDR_CONFIG['sample_rate'],
            nfft=1024,
            drone_id=drone_predict,
            schema_version="1.0",
        )
        publish_psd(x, center_freq, SDR_CONFIG['sample_rate'], nfft=1024, drone_id=drone_predict)

        try:
            spec_frame = build_spectrogram_frame(
                spectrogram=spec.unsqueeze(0),
                class_name=drone_predict,
            )
            publish_spec(spec_frame)
        except Exception as e:
            print("[pipeline] spec_hub.set_last error:", e)

        visualize_spectrogram(
            spectrogram=spec.unsqueeze(0),
            class_name=f"{drone_predict}",
            blade_id = DEVICE_NAME,
            **VIS_CONFIG
        )

        # ============ DOA PROCESSING ============
        # Only perform DOA if drone detected (not noise/jammer)
        if drone_predict not in ['Noise', 'Jammer']:
            print("entro a doa ----------->")
            fc_hz = center_freq
            lam = c / fc_hz
            d_lambda = (d_cm / 100.0) / lam

            rx1.enable = False
            rx_ch.enable = False
            sdr.sync_config(
                _bladerf.ChannelLayout.RX_X2,
                _bladerf.Format.SC16_Q11,
                num_buffers=DOA_CONFIG['num_buffers'],
                buffer_size=DOA_CONFIG['buffer_size'],
                num_transfers=DOA_CONFIG['num_transfers'],
                stream_timeout=SDR_CONFIG['stream_timeout']
            )
            rx1.enable = True
            rx_ch.enable = True

            num_samples_doa = DOA_CONFIG['num_samples']
            buf_doa = bytearray(num_samples_doa * 8)
            sdr.sync_rx(buf_doa, num_samples_doa)
            raw = np.frombuffer(buf_doa, dtype=np.int16).reshape(-1, 4)
            x1 = (raw[:, 0] + 1j * raw[:, 1]) / 2048.0
            x2 = (raw[:, 2] + 1j * raw[:, 3]) / 2048.0

            x1 /= np.sqrt(np.mean(np.abs(x1) ** 2))
            x2 /= np.sqrt(np.mean(np.abs(x2) ** 2))

            xc = correlate(x1, x2, mode='full')
            lag = np.argmax(np.abs(xc)) - (len(x1) - 1)
            phi = np.angle(np.vdot(x1, x2))
            phi_deg = np.degrees(phi)
            print(f"[DOA] lag_muestras={lag}, fase_promedio={phi_deg:.3f} grados")

            publish_doa(
                angle_deg=float(phi_deg),
            )

            X_full = np.stack([x1, x2], axis=1)
            num_blocks = X_full.shape[0] // block_size
            acc = np.zeros(len(angles), dtype=float)

            num_signals = 1
            for b in range(num_blocks):
                sl = slice(b * block_size, (b + 1) * block_size)
                Xb = X_full[sl, :]
                acc += music_block(Xb, angles, d_lambda, num_expected_signals=num_signals, diag_load=1e-6)

            P_music = acc / max(1, num_blocks)
            psr_db = 10 * np.log10(P_music.max() / (np.median(P_music) + 1e-12))
            if psr_db < 8:
                print(f"[DOA] Sin DOA confiable (PSR={psr_db:.1f} dB).")

            P_lin = np.asarray(P_music, float)
            ang = np.asarray(angles, float)
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
            print(f"[DOA] Ángulo estimado: {theta_hat:.2f}°")

            dr = DOA_CONFIG['dynamic_range_db']
            P_db = 10 * np.log10(np.maximum(P_music, 1e-12))
            P_db = np.clip(P_db - P_db.max(), -dr, 0)

            theta = np.deg2rad(angles)

            fig = plt.figure(figsize=(16, 7), facecolor="none")
            ax = fig.add_subplot(111, projection="polar", facecolor="none")
            ax.patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            ax.plot(theta, P_db, linewidth=2, color=DOA_CONFIG['military_green'])

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetamin(-90)
            ax.set_thetamax(90)

            ax.set_rlim(-dr, 0)
            ax.set_rlabel_position(180)
            ax.grid(True)
            plt.tight_layout()

        if center_freq >= SDR_CONFIG['max_freq']:
            center_freq = SDR_CONFIG['max_freq']
            direction = -1
        elif center_freq <= SDR_CONFIG['min_freq']:
            center_freq = SDR_CONFIG['min_freq']
            direction = 1

        center_freq += direction * SDR_CONFIG['step_freq']

        # Update SDR frequency
        for ch in (rx1, rx_ch):
            ch.frequency = center_freq

        # Print processing time
        elapsed = time.time() - start_time
        print(f"[Pipeline] Tiempo de procesamiento: {elapsed:.3f} segundos\n")



except KeyboardInterrupt:
    print("\n[Pipeline] Interrumpido por el usuario (Ctrl+C)")
except Exception as e:
    print(f"\n[Pipeline] Error inesperado: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("[Pipeline] Cerrando SDR...")
    try:
        rx1.enable = False
        rx_ch.enable = False
        sdr.close()
        print("[Pipeline] SDR cerrado correctamente")
    except Exception as e:
        print(f"[Pipeline] Error al cerrar SDR: {e}")
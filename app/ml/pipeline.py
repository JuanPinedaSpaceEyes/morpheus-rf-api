import os
import sys
import ctypes
from pathlib import Path
from datetime import datetime
import time
import threading
import shutil
import subprocess
from typing import Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
import torch
from torchaudio.transforms import Spectrogram
from scipy.signal import correlate
from bladerf import _bladerf
from collections import Counter, defaultdict

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
            "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced_BIN-UNF1-(IQSignal_DroneDetectSNR).pt",
        )
    },
    "multiclass": {
        "path": os.getenv(
            "MULTICLASS_MODEL_PATH",
            "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced_MC-UNF5-(IQSig_DroneDetectSNR).pt",
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

C0 = 299_792_458.0          # velocidad de la luz (m/s)

SAMPLE_RATE = 40e6          # Hz
CENTER_FREQ = 2_440_000_000 # Hz
GAIN_DB     = 30            # dB  (-15 a 60)

N_FFT   = 512               # tamaño FFT para espectrograma
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
D_CM  = 6.25           # separación entre antenas en cm
D_M   = D_CM / 100.0   # en metros
FC_HZ = 2.44e9         # frecuencia central para DOA

# DOA / MVDR params
BLOCK_SIZE_MVDR = 4096
ANGLES_DEG      = np.linspace(-90, 90, 721)   # malla fina
DIAG_LOAD       = 1e-3


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
# Helpers PSD / SPEC / PRED / DOA
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


def publish_doa(angle_deg: float) -> None:
    body = {"angle_deg": float(angle_deg)}
    try:
        requests.post(DOA_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_doa HTTP error:", e)


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
                return dev
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

) -> None:
    """
    Bucle principal del pipeline. Bloqueante. Pensado para correrse en un hilo.

    Nuevo comportamiento:
    - Modo SCAN: el bladeRF barre un conjunto de frecuencias centrales
      (definidas por SDR_CONFIG[min_freq, max_freq, step_freq]).
    - Si en alguna frecuencia se detecta un dron (clase != "Noise"/"Jammer"),
      el pipeline entra en modo TRACK y se queda "anclado" en esa frecuencia.
    - Si en modo TRACK se deja de ver el dron durante varios ciclos seguidos,
      se vuelve a modo SCAN y se reanuda el barrido hasta encontrarlo de nuevo.

    """

    NUM_READS = 5  # nº de lecturas MVDR por ciclo
    PSR_MIN_DB = 8.0  # umbral de calidad (peak / median)
    WINDOW_WIDTH = 10.0
    effective_dev_id = dev_id or os.getenv("BLADERF_DEVICE")
    blade_name = device_name or os.getenv("PIPELINE_DEVICE_NAME", effective_dev_id or "default_blade")

    print(f"[Pipeline] run_pipeline(dev_id={effective_dev_id}, name={blade_name})")



    # Parámetros del comportamiento de escaneo / seguimiento
    LOCK_CONSECUTIVE = int(os.getenv("DRONE_LOCK_CONSECUTIVE", "2"))   # nº de detecciones seguidas para fijar frecuencia
    LOST_CONSECUTIVE = int(os.getenv("DRONE_LOST_CONSECUTIVE", "4"))   # nº de pérdidas seguidas para soltar frecuencia
    SCAN_WINDOW = int(os.getenv("SCAN_WINDOW_SIZE", "1"))              # ventana de suavizado en modo SCAN
    TRACK_WINDOW = int(os.getenv("TRACK_WINDOW_SIZE", "5"))            # ventana de suavizado en modo TRACK

    # Definimos lista discreta de frecuencias de escaneo a partir de la config
    min_freq = int(SDR_CONFIG["min_freq"])
    max_freq = int(SDR_CONFIG["max_freq"])
    step_freq = int(SDR_CONFIG["step_freq"])

    if step_freq <= 0:
        raise RuntimeError("SDR_CONFIG['step_freq'] debe ser > 0 para el modo SCAN/TRACK")

    scan_freqs = list(range(min_freq, max_freq + 1, step_freq))
    if not scan_freqs:
        scan_freqs = [int(SDR_CONFIG["center_freq"])]

    print(f"[Pipeline] Frecuencias de escaneo: {', '.join(str(f) for f in scan_freqs)}")

    # Estado del modo de operación
    mode = "scan"           # 'scan' o 'track'
    locked_freq = None      # frecuencia central bloqueada en modo TRACK
    last_detection_freq = None
    scan_index = 0
    consecutive_hits = 0
    consecutive_misses = 0


    try:
        sdr = _open_bladerf_from_env(effective_dev_id)
    except Exception as e:
        print(f"[Pipeline] No se pudo abrir bladeRF ({blade_name}): {e}")
        return

    # Configurar el bladeRF para usar referencia externa de 10 MHz
    sdr.set_pll_refclk(int(10e6))
    sdr.set_pll_enable(True)
    # (Opcional pero recomendado) Esperar a que el PLL bloquee
    for _ in range(50):
        if sdr.get_pll_lock_state():  # True cuando el PLL está bloqueado
            break
        time.sleep(0.1)

    rx0, rx1 = configure_common_channels(sdr)

    # Config canales
    rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))
    rx1 = sdr.Channel(_bladerf.CHANNEL_RX(0))

    for ch in (rx1, rx_ch):
        ch.sample_rate = SDR_CONFIG["sample_rate"]
        ch.bandwidth = SDR_CONFIG["sample_rate"] / SDR_CONFIG["bandwidth_divisor"]
        ch.gain_mode = _bladerf.GainMode.Manual
        ch.gain = SDR_CONFIG["gain"]

    rx1.enable = False
    rx_ch.enable = True

    buf = bytearray(1024 * SDR_CONFIG["bytes_per_sample"])
    transform = transform_spectrogram(device="cpu", **SPEC_CONFIG)

    c = DOA_CONFIG["speed_of_light"]
    d_cm = DOA_CONFIG["antenna_spacing_cm"]
    block_size = DOA_CONFIG["block_size"]
    angles = DOA_CONFIG["angles"]

    print(f"[Pipeline] Iniciando recepción ({blade_name}) (Two-stage: {TWO_STAGE_PREDICTION})")

    # Historial para suavizado de predicciones
    binary_results_list = []
    multiclass_results_list = []
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print(f"[Pipeline] stop_event recibido, saliendo ({blade_name})")
                break

            # Seleccionar frecuencia y ventana de suavizado según modo
            if mode == "scan":
                window_size = SCAN_WINDOW
                center_freq = scan_freqs[scan_index]
                scan_index = (scan_index + 1) % len(scan_freqs)
            else:  # mode == "track"
                window_size = TRACK_WINDOW
                if locked_freq is None:
                    # Fallback: si por alguna razón no hay frecuencia bloqueada, volvemos a scan
                    mode = "scan"
                    window_size = SCAN_WINDOW
                    center_freq = scan_freqs[scan_index]
                    scan_index = (scan_index + 1) % len(scan_freqs)
                else:
                    center_freq = locked_freq

            start_time = time.time()
            print(f"\n[{blade_name}] MODO={mode.upper()} FRECUENCIA: {center_freq} Hz")

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

            # ----------- Captura de muestras -----------
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

            I = np.real(x)
            Q = np.imag(x)
            sample = torch.tensor(np.stack([I, Q], axis=0))
            spec = transform(sample)

            # === Predicción ===
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
                    window_size=window_size,
                )
                print(
                    f"\n[{blade_name}] [Stage 1] {binary_result['class_name']} "
                    f"(conf={binary_result['confidence']:.4f})"
                )


                if multiclass_result:
                    drone_predict_raw = multiclass_result['class_name']
                    drone_confidence_raw = multiclass_result['confidence']
                    drone_predict = multiclass_result_smoothed['confidence']
                    drone_confidence = multiclass_result_smoothed['class_name']  # Update final class name if smoothed
                    print(
                        f"[{blade_name}] [Stage 2] {multiclass_result['class_name']} "
                        f"(conf={multiclass_result['confidence']:.4f})"
                    )

                drone_predict = final_class_name
                # Usamos la confianza del binario suavizado como valor principal
                drone_confidence = (
                    binary_result_smoothed["confidence"]
                    if binary_result_smoothed is not None
                    else binary_result["confidence"]
                )

                print(f"[{blade_name}] [Final] {drone_predict} (conf={drone_confidence:.4f})")

                publish_pred(
                    label_name=drone_predict,
                    confidence=drone_confidence,
                    binary_info=binary_result_smoothed,
                    multiclass_info=multiclass_result_smoothed,
                    center_freq=float(center_freq),
                    mode=mode,
                )

                # Etiqueta para la imagen del espectrograma (solo debug)
                label_for_plot = f"Final: {drone_predict} ({drone_confidence:.2f})"
                visualize_spectrogram(
                    spectrogram=spec.unsqueeze(0),
                    class_name=label_for_plot,
                    blade_id=blade_name,
                    **VIS_CONFIG,
                )
            else:
                with torch.no_grad():
                    spec_input = spec.unsqueeze(0).unsqueeze(0)
                    outputs = model(spec_input)
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)

                drone_predict = CLASS_DICTS["multiclass"][preds.item()]
                drone_confidence = probs[0, preds.item()].item()
                print(f"[{blade_name}] [Prediction] {drone_predict} (conf={drone_confidence:.4f})")
                publish_pred(
                    label_name=drone_predict,
                    confidence=drone_confidence,
                    center_freq=float(center_freq),
                    mode=mode,
                )

            # ----------- PSD → backend -----------
            publish_psd(x, center_freq, SDR_CONFIG["sample_rate"], nfft=1024, drone_id=drone_predict)

            # ----------- Spectrogram → backend -----------
            try:
                spec_frame = build_spectrogram_frame(
                    spectrogram=spec.unsqueeze(0),
                    class_name=drone_predict,
                )
                publish_spec(spec_frame)
            except Exception as e:
                print("[pipeline] publish_spec error:", e)

            # ----------- DOA (solo si no es Noise/Jammer) -----------
            if drone_predict not in ["Noise", "Jammer"]:
                spectra_acc = None  # acumular espectros MVDR
                angles_list = []  # ángulos válidos
                valid_reads = 0

                for i in range(NUM_READS):

                    rx0.enable = False
                    rx1.enable = False
                    sdr.sync_config(
                        layout=_bladerf.ChannelLayout.RX_X2,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=32,
                        buffer_size=16384,
                        num_transfers=16,
                        stream_timeout=3500,
                    )
                    rx0.enable = True
                    rx1.enable = True

                    x1, x2 = capture_dual_channel(sdr, NUM_SAMPLES_DOA)

                    # Normalizar potencia
                    x1 /= np.sqrt(np.mean(np.abs(x1) ** 2) + 1e-12)
                    x2 /= np.sqrt(np.mean(np.abs(x2) ** 2) + 1e-12)

                    # Matriz de datos para MVDR: (K, M) = (N, 2)
                    X_full = np.stack([x1, x2], axis=1)  # (K, 2)
                    X_full = preprocess_iq(X_full, demean=True, normalize=True)

                    P_mvdr, theta_hat, psr_db, d_lambda = estimate_doa_mvdr(
                        X_full,
                        fc_hz=FC_HZ,
                        d_m=D_M,
                        block_size=BLOCK_SIZE_MVDR,
                        angles_deg=ANGLES_DEG,
                        diag_load=DIAG_LOAD,
                    )

                    # print(f"[MVDR] PSR = {psr_db:.1f} dB, theta_hat = {theta_hat:.2f}°")

                    # Filtro de calidad
                    if psr_db < PSR_MIN_DB:
                        # print("   -> Lectura descartada (PSR demasiado bajo)")
                        continue

                    # Acumular espectro y ángulo
                    if spectra_acc is None:
                        spectra_acc = P_mvdr.copy()
                    else:
                        spectra_acc += P_mvdr

                    angles_list.append(theta_hat)
                    valid_reads += 1

                if valid_reads == 0:
                    print("\n[DOA MVDR] No se obtuvo ninguna lectura confiable.")
                    continue

                    # Espectro promedio
                P_mvdr_mean = spectra_acc / valid_reads

                # Array de ángulos
                angles_arr = np.array(angles_list)

                # ==========================================
                # Buscar el "ángulo más común" por clusters
                # ==========================================
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
                theta_cluster_mean = float(cluster_angles.mean())

                print("\n[DOA MVDR] Lecturas válidas:", valid_reads)
                print("   Ángulos individuales (deg):", angles_arr)
                print(
                    f"   Grupo más denso dentro de ±{WINDOW_WIDTH / 2:.1f}° → {cluster_angles}"
                )
                print(
                    f"   Ángulo 'más común' (media del grupo): {theta_cluster_mean:.2f}°"
                )




            # ----------- Actualizar estado SCAN / TRACK -----------
            # Consideramos que hay "dron presente" si la clase final NO es Noise/Jammer
            drone_present = drone_predict not in ["Noise", "Jammer"]

            if mode == "scan":
                if drone_present:
                    consecutive_hits += 1
                    last_detection_freq = center_freq
                    print(
                        f"[{blade_name}] SCAN: detección en {center_freq} Hz "
                        f"({consecutive_hits}/{LOCK_CONSECUTIVE})"
                    )
                    if consecutive_hits >= LOCK_CONSECUTIVE:
                        mode = "track"
                        locked_freq = center_freq
                        consecutive_hits = 0
                        consecutive_misses = 0
                        print(f"[{blade_name}] >>> Cambio a modo TRACK en {locked_freq} Hz")
                else:
                    if consecutive_hits > 0:
                        print(f"[{blade_name}] SCAN: detección interrumpida, reseteando contador.")
                    consecutive_hits = 0
            else:
                if drone_present:
                    consecutive_misses = 0
                    last_detection_freq = center_freq
                else:
                    consecutive_misses += 1
                    print(
                        f"[{blade_name}] TRACK: pérdida {consecutive_misses}/{LOST_CONSECUTIVE} "
                        f"en {center_freq} Hz"
                    )
                    if consecutive_misses >= LOST_CONSECUTIVE:
                        print(
                            f"[{blade_name}] >>> Dron perdido en {center_freq} Hz, "
                            f"volviendo a modo SCAN."
                        )
                        mode = "scan"
                        locked_freq = None
                        consecutive_hits = 0
                        consecutive_misses = 0
                        # Reanudar scan desde la frecuencia siguiente a la última detección
                        if last_detection_freq is not None and scan_freqs:
                            try:
                                idx = scan_freqs.index(int(last_detection_freq))
                                scan_index = (idx + 1) % len(scan_freqs)
                            except ValueError:
                                # Si por alguna razón no está exacta en la lista (redondeos), no pasa nada
                                pass

            elapsed = time.time() - start_time
            print(f"[{blade_name}] Tiempo de procesamiento: {elapsed:.3f} s\n")

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

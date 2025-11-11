from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F  # interpolate
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Spectrogram
from scipy.signal import correlate
import json  # <--- agregado

# -----------------------------
# Guardado de plots por entorno
# -----------------------------
SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(
    os.getenv("PIPELINE_PLOT_DIR", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/plots"))
if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Directorio/rutas para “último resultado” (JSON e imagen)
LAST_DIR = Path(os.getenv("PIPELINE_LAST_DIR", str(PLOT_DIR)))
LAST_DIR.mkdir(parents=True, exist_ok=True)
LAST_JSON = LAST_DIR / "last_doa.json"
LAST_POLAR = LAST_DIR / "mvdr_polar_latest.png"

def _atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def _save_latest_figure(fig_path: Path):
    # Guarda con timestamp y además sobreescribe "latest"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ts_path = fig_path.with_name(fig_path.stem + f"_{ts}" + fig_path.suffix)
    plt.savefig(ts_path, dpi=120, bbox_inches="tight")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")  # copia "latest"
    return str(fig_path), str(ts_path)

# --- Apertura robusta del bladeRF ---
os.environ.setdefault("LIBUSB_DEBUG", "3")
DEV_ID = os.getenv("BLADERF_DEVICE")
try:
    if DEV_ID:
        # Ejemplos válidos:
        #   "*:serial=1f76c0...."
        #   "libusb:bus=20,addr=3"
        sdr = _bladerf.BladeRF(DEV_ID)
    else:
        sdr = _bladerf.BladeRF()
except _bladerf.NoDevError:
    print("[pipeline] No se encontraron dispositivos bladeRF desde este proceso.")
    # Volcamos lo que ve la CLI para comparar (quedará en /pipeline/logs)
    try:
        out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
        print("[pipeline] bladeRF-cli -p stdout:\n", out.stdout)
        print("[pipeline] bladeRF-cli -p stderr:\n", out.stderr)
    except Exception as e:
        print("[pipeline] No pude ejecutar bladeRF-cli -p:", e)
    sys.exit(2)


# ------------------------------------------------------------------

class transform_spectrogram(torch.nn.Module):
    def __init__(
            self,
            device,
            n_fft=1024,
            win_length=1024,
            hop_length=2930,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            center=False,
            onesided=False
    ):
        super().__init__()
        self.spec = Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=window_fn,
                                power=power, normalized=normalized, center=center, onesided=onesided).to(device=device)
        self.win_length = win_length  # Fixed: was self.win_lengt
        self.epsilon = 1e-12

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        iq_signal = iq_signal[0, :] + (1j * iq_signal[1, :])
        spec = self.spec(iq_signal)
        spec = torch.view_as_real(spec)
        spec = torch.moveaxis(spec, 2, 0)
        spec = torch.sqrt(spec[0, :, :] ** 2 + spec[1, :, :] ** 2)
        spec = 10 * torch.log10(spec + self.epsilon)
        return spec


def visualize_spectrogram(spectrogram: torch.Tensor, class_name: torch.Tensor,
                          n_fft=1024, win_length=1024, hop_length=2930, sample_freq=40e6,
                          show_stats: bool = True, figsize: tuple = (16, 4)) -> None:
    if spectrogram.ndim != 3 or spectrogram.shape[0] < 2:
        ch_names = {0: f"Power Spectrogram - {class_name}"}
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]  # Make it iterable
    elif spectrogram.shape[0] >= 2:
        ch_names = {0: "I Spectrogram", 1: "Q Spectrogram"}
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'{class_name}')
    else:
        raise ValueError("Spectrogram shape {spectrogram.shape} doesnt match expected dimensions")

    spectrogram_np = spectrogram.detach().cpu().numpy()

    n_freq_bins = spectrogram_np.shape[1]  # Frequency bins
    n_time_bins = spectrogram_np.shape[2]  # Time bins

    time_per_fft = hop_length / sample_freq  # Time per FFT in seconds
    time_axis = np.arange(n_time_bins) * time_per_fft  # Match actual time bins
    time_duration_ms = time_axis * 1000  # Convert to ms

    freqs = sample_freq / win_length * np.arange(-n_fft / 2, n_fft / 2)
    freqs_in_mhz = freqs / 1e6

    t = time_duration_ms
    f = freqs_in_mhz

    for ch in range(spectrogram_np.shape[0]):
        im = axes[ch].imshow(
            spectrogram_np[ch],
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            interpolation='nearest'
        )
        axes[ch].set_ylabel('Frequency [MHz]')
        axes[ch].set_xlabel('Time [ms]')
        max_time = time_duration_ms[-1] if len(time_duration_ms) > 0 else 75
        axes[ch].set_xticks(np.arange(0, max_time, step=10))
        axes[ch].set_title(f'{ch_names[ch]}')
        plt.colorbar(im, ax=axes[ch], label='Power [dB]')

    plt.tight_layout()
    plt.show()


def steering_vector(theta_deg, M, d_lambda):
    m = np.arange(M)[:, None]  # (M,1)
    return np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(np.deg2rad(theta_deg)))  # (M,1)


def mvdr_block(X_block, angles, d_lambda):
    K, M = X_block.shape
    R = (X_block.conj().T @ X_block) / max(1, K)
    delta = 1e-3 * (np.trace(R).real / M)
    R = R + delta * np.eye(M)
    R_inv = np.linalg.pinv(R)

    P = np.empty(len(angles), dtype=float)
    for i, ang in enumerate(angles):
        a = steering_vector(ang, M, d_lambda)  # (M,1)
        denom = (a.conj().T @ R_inv @ a).item()
        P[i] = 1.0 / np.real(denom)
    return P


# --- Crear ambos canales RX ---
rx1 = sdr.Channel(_bladerf.CHANNEL_RX(0))  # RX1 (antena 1)
rx2 = sdr.Channel(_bladerf.CHANNEL_RX(1))  # RX2 (antena 2)

# Configs
sample_rate = 40e6
center_freq = 2455500000  # 2445.5 - 2455.5
step_freq = 40000000  # 20MHz
gain = 30  # -15 a 60 dB
num_samples = int(3e6)  # hop_lenght

for ch in (rx1, rx2):
    ch.frequency = center_freq
    ch.sample_rate = sample_rate
    ch.bandwidth = sample_rate / 2
    ch.gain_mode = _bladerf.GainMode.Manual
    ch.gain = gain

# --- Sync: 2 Rx (MIMO) ---
sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500)

# Habilitar ambos front-ends
rx1.enable = True
rx2.enable = True

# --- Recepción y desentrelazado ---
ints_per_frame = 4  # I0,Q0,I1,Q1
bytes_per_frame = 2 * ints_per_frame  # 8 bytes
buf = bytearray(1024 * bytes_per_frame)

transform = transform_spectrogram(
    device="cpu",
    n_fft=1024,
    win_length=1024,
    hop_length=2930,  # 73.6 us
    window_fn=torch.hann_window,
    power=None,
    normalized=False,
    center=False,
    onesided=False
)

print("Starting dual-RX receive")

c = 299_792_458.0
fc_hz = center_freq
d_cm = 6.14  # <-- AJUSTA A TU SEPARACIÓN REAL (cm)
lam = c / fc_hz
d_lambda = (d_cm / 100.0) / lam
print("d_lambda: ", d_lambda)
block_size = 4096
angles = np.linspace(-90, 90, 721)  # malla más fina

while True:
    x1 = np.zeros(num_samples, dtype=np.complex64)
    x2 = np.zeros(num_samples, dtype=np.complex64)
    frames_read = 0
    while frames_read < num_samples:
        ask = min(len(buf) // bytes_per_frame, num_samples - frames_read)
        sdr.sync_rx(buf, ask)

        raw = np.frombuffer(buf, dtype=np.int16, count=ask * ints_per_frame)
        i0, q0 = raw[0::4].astype(np.float32), raw[1::4].astype(np.float32)
        i1, q1 = raw[2::4].astype(np.float32), raw[3::4].astype(np.float32)

        x1[frames_read:frames_read + ask] = (i0 + 1j * q0) / 2048.0
        x2[frames_read:frames_read + ask] = (i1 + 1j * q1) / 2048.0
        frames_read += ask

    # =========================
    # [DOA-INSERCIÓN A]  (TDOA/FASE INMEDIATA TRAS ADQUISICIÓN)
    # =========================
    xc = correlate(x1, x2, mode='full')
    lag = np.argmax(np.abs(xc)) - (len(x1) - 1)
    phi = np.angle(np.vdot(x1, x2))  # fase media radiantes
    phi_deg = np.degrees(phi)  # ángulo de fase (relativo)

    # TDOA -> ángulo (cuida el dominio de asin)
    d_m = d_cm / 100.0
    t_delay = lag / sample_rate
    arg = np.clip((c * t_delay) / d_m, -1.0, 1.0)
    theta_tdoa = np.degrees(np.arcsin(arg))

    print(f"[TDOA] lag_muestras={lag}, fase_promedio={phi_deg:.3f}°, theta_tdoa={theta_tdoa:.2f}°")

    # Escribe/actualiza JSON con datos parciales (sin MVDR aún)
    try:
        partial = {
            "ts": datetime.now().isoformat(),
            "fc_hz": fc_hz,
            "sample_rate": sample_rate,
            "d_cm": d_cm,
            "phase_deg": float(phi_deg),
            "lag_samples": int(lag),
            "theta_tdoa_deg": float(theta_tdoa)
        }
        _atomic_write_json(LAST_JSON, partial)
    except Exception as e:
        print("[warn] No pude escribir last_doa.json (parcial):", e)

    # =========================
    # (tu espectrograma: sin cambios)
    # =========================
    I1, Q1 = np.real(x1), np.imag(x1)
    I2, Q2 = np.real(x2), np.imag(x2)

    sample1 = torch.tensor(np.stack([I1, Q1], axis=0))  # RX1
    sample2 = torch.tensor(np.stack([I2, Q2], axis=0))  # RX2

    spec1 = transform(sample1)
    spec2 = transform(sample2)

    visualize_spectrogram(
        spectrogram=spec1.view(1, 1024, 1024),
        class_name=f"test"
    )
    visualize_spectrogram(
        spectrogram=spec2.view(1, 1024, 1024),
        class_name=f"test"
    )

    # normaliza potencia
    x1 /= np.sqrt(np.mean(np.abs(x1) ** 2))
    x2 /= np.sqrt(np.mean(np.abs(x2) ** 2))
    X_full = np.stack([x1, x2], axis=1)  # shape (N, 2)
    num_blocks = X_full.shape[0] // block_size
    acc = np.zeros(len(angles), dtype=float)

    for b in range(num_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        Xb = X_full[sl, :]  # (K, 2)
        acc += mvdr_block(Xb, angles, d_lambda)

    P_mvdr = acc / max(1, num_blocks)
    # ===== Estimar ángulo a partir de MVDR =====
    P_lin = np.asarray(P_mvdr, float)
    ang = np.asarray(angles, float)
    step = ang[1] - ang[0]

    edge = 2
    i_search = np.arange(edge, len(P_lin) - edge)
    i0 = i_search[np.argmax(P_lin[i_search])]  # índice del pico

    if 0 < i0 < len(P_lin) - 1:
        y1, y2, y3 = P_lin[i0 - 1], P_lin[i0], P_lin[i0 + 1]
        denom = (y1 - 2 * y2 + y3)
        delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
    else:
        delta = 0.0

    theta_hat = ang[i0] + delta * step  # grados
    print("El angulo en que se encuentra el dron es: ", theta_hat)

    # ---------------------------
    # Gráfica polar semicircular en dB
    # ---------------------------
    dr = 35  # rango dinámico mostrado en dB
    P_db = 10 * np.log10(np.maximum(P_mvdr, 1e-12))
    P_db = np.clip(P_db - P_db.max(), -dr, 0)  # normaliza pico a 0 dB y recorta a [-dr, 0]
    theta = np.deg2rad(angles)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, P_db, linewidth=2)

    # Semicírculo superior: 0° arriba, -90° izq, +90° der
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_rlim(-dr, 0)
    ax.set_rlabel_position(180)
    ax.grid(True)
    plt.title(f"MVDR (2 canales) — θ̂ ≈ {theta_hat:.2f}°")
    plt.tight_layout()

    # =========================
    # [DOA-INSERCIÓN B]  (GUARDADO DE IMAGEN + JSON COMPLETO)
    # =========================
    latest_path = None
    try:
        if SAVE_PLOTS:
            latest_path, _ = _save_latest_figure(LAST_POLAR)
        else:
            # Si no guardas plots, igual cerramos la figura para no consumir memoria
            pass
    except Exception as e:
        print("[warn] No pude guardar la figura polar:", e)
    finally:
        plt.show()

    # Actualiza JSON con el MVDR y la imagen (si existe)
    try:
        payload = {
            "ts": datetime.now().isoformat(),
            "fc_hz": fc_hz,
            "sample_rate": sample_rate,
            "d_cm": d_cm,
            "phase_deg": float(phi_deg),
            "lag_samples": int(lag),
            "theta_tdoa_deg": float(theta_tdoa),
            "theta_mvdr_deg": float(theta_hat),
            "latest_polar_path": latest_path
        }
        _atomic_write_json(LAST_JSON, payload)
    except Exception as e:
        print("[warn] No pude escribir last_doa.json (final):", e)

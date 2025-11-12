from bladerf import _bladerf
import numpy as np, time, math, requests
import matplotlib.pyplot as plt
import torch
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Spectrogram
from scipy.signal import correlate

# pipeline.py está en app/ml/, la raíz del proyecto es parents[2]
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.routers.pipeline_router import psd_hub, make_psd_frame, pred_hub, doa_hub

INGEST_URL = os.getenv("PSD_INGEST_URL", "http://127.0.0.1:8000/pipeline/psd/ingest")
PRED_INGEST_URL = os.getenv("PRED_INGEST_URL", "http://127.0.0.1:8000/pipeline/pred/ingest")
DOA_INGEST_URL = os.getenv("DOA_INGEST_URL", "http://127.0.0.1:8000/pipeline/doa/ingest")
# ------- Apertura robusta del bladeRF ------------------------------------------------------------------------------------------
os.environ.setdefault("LIBUSB_DEBUG", "3")
os.environ.pop("LIBUSB_DEBUG", None)
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

# --------- IMAGE SAVE CONFIG -------------------------------------------------------------------------------------------------------
SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(os.getenv("PIPELINE_PLOT_DIR", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/plots"))
if SAVE_PLOTS:
     PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Model configs------------------------------------------------------------------------------------------------------------
dictionaryDrones = {0: 'DJI Mini 4K', 1: 'Jammer', 2: 'Noise'}
# dictionaryDrones = {0: 'DJI Inspire 2', 1: 'DJI MINI 4K', 2: 'DJI Mavic 2 Air S', 3: 'DJI Mavic Mini', 4: 'DJI Mavic Pro', 5: 'DJI Mavic Pro 2', 6: 'DJI Phantom 4', 7: 'Noise', 8: 'Parrot Disco'}
# dictionaryDrones = {0: 'Dron', 1: 'Noise'}
# --------------------------------- LOAD MODEL -----------------------------------------------------------------------------------------
IN_CHANNELS = 1
NUM_CLASSES = 3
shape = 512  # 512 - 1024
hop = 5860  # 5860 - 2930
MODEL_TRACE = "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced_mc.pt"

try:
    model = torch.jit.load(MODEL_TRACE, map_location="cpu")
    model.eval()
    print(f"[pipeline] Modelo trazado cargado correctamente desde: {MODEL_TRACE}")
except Exception as e:
    print(f"[pipeline] Error cargando modelo trazado: {e}")
    sys.exit(1)


# ---------------- FUNCIONES ------------------------------------------------------------------------------------------------------------

def _compute_psd_db(x: np.ndarray, nfft: int = 4096) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("x debe ser 1D complejo")
    seg = x[:nfft]
    if seg.shape[0] < nfft:
        seg = np.pad(seg, (0, nfft - seg.shape[0]))
    w = np.hanning(nfft)
    X = np.fft.fftshift(np.fft.fft(seg * w, n=nfft))
    return 10.0 * np.log10(np.abs(X) + 1e-12)


def publish_psd(x_complex: np.ndarray, center_hz: float, sample_rate: float, nfft: int = 4096,
                drone_id: str | None = None):
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


class transform_spectrogram(torch.nn.Module):
    def __init__(
            self,
            device,
            n_fft=shape,
            win_length=shape,
            hop_length=hop,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            center=False,
            onesided=False
    ):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            window_fn=window_fn, power=power, normalized=normalized,
            center=center, onesided=onesided
        ).to(device=device)
        self.win_length = win_length
        self.epsilon = 1e-12

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        # Combinar I/Q en señal compleja
        iq_signal = iq_signal[0, :] + 1j * iq_signal[1, :]
        iq_signal = iq_signal - iq_signal.mean()
        # --- Espectrograma ---
        spec_complex = self.spec(iq_signal)
        # spec_complex = torch.fft.fftshift(spec_complex, dim=0)
        # `spec_complex` es complejo: separar parte real e imaginaria
        spec_real = spec_complex.real
        spec_imag = spec_complex.imag
        # Calcular magnitud
        spec_magnitude = torch.sqrt(spec_real ** 2 + spec_imag ** 2 + self.epsilon)
        # Convertir a escala logarítmica (dB)
        spec_db = 10 * torch.log10(spec_magnitude + self.epsilon)
        return spec_db


def visualize_spectrogram(spectrogram: torch.Tensor, class_name: torch.Tensor,
                          n_fft=shape, win_length=shape, hop_length=hop, sample_freq=40e6,
                          show_stats: bool = True, figsize: tuple = (16, 4)) -> None:
    """
    Visualize a spectrogram with I and Q channels.

    Args:
        spectrogram: Input spectrogram tensor of shape (channels, freq_bins, time_bins)
        label: Label tensor for the spectrogram
        sampling_rate: Sampling rate in Hz
        show_stats: Whether to print statistics
        figsize: Figure size tuple (width, height)
    """
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

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Class: {class_name}")

    # Convert to numpy for matplotlib
    spectrogram_np = spectrogram.detach().cpu().numpy()

    # Get actual dimensions from the spectrogram
    n_freq_bins = spectrogram_np.shape[1]  # Frequency bins
    n_time_bins = spectrogram_np.shape[2]  # Time bins

    # Create time axis based on actual number of time bins
    time_per_fft = hop_length / sample_freq  # Time per FFT in seconds
    time_axis = np.arange(n_time_bins) * time_per_fft  # Match actual time bins
    time_duration_ms = time_axis * 1000  # Convert to ms

    # Create frequency axis
    freqs = sample_freq / win_length * np.arange(-n_fft / 2, n_fft / 2)
    freqs_in_mhz = freqs / 1e6

    # # Ensure frequency axis matches spectrogram dimensions
    # if len(freqs_in_mhz) > n_freq_bins:
    #     freqs_in_mhz = freqs_in_mhz[:n_freq_bins]

    t = time_duration_ms  # Time in milliseconds
    f = freqs_in_mhz  # Frequency in MHz

    # Create visualization
    for ch in range(spectrogram_np.shape[0]):
        # Use imshow instead of pcolormesh (much faster!)
        im = axes[ch].imshow(
            spectrogram_np[ch],
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            interpolation='nearest'  # or 'bilinear' for smoother look
        )
        axes[ch].set_ylabel('Frequency [MHz]')
        axes[ch].set_xlabel('Time [ms]')
        # Set x-axis ticks in milliseconds
        max_time = time_duration_ms[-1] if len(time_duration_ms) > 0 else 75
        axes[ch].set_xticks(np.arange(0, max_time, step=10))  # every 10 ms
        axes[ch].set_title(f'{ch_names[ch]}')
        # axes[ch].set_ylim([0, f.max()])  # Only positive frequencies
        plt.colorbar(im, ax=axes[ch], label='Power [dB]')

    plt.tight_layout()

    if SAVE_PLOTS:
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = PLOT_DIR / f"spectrogram_{ts}.png"
            plt.savefig(out_path, dpi=120)
            print(f"[pipeline] saved_plot={out_path}")
        except Exception as e:
            print("[pipeline] savefig error:", e)
    else:
        plt.close()


    # # Print statistics if requested (ignoring NaN values)
    # # Check for NaN values
    # nan_count = np.isnan(spectrogram_np).sum()
    # total_elements = spectrogram_np.size
    # print(f"NaN values in spectrogram: {nan_count} out of {total_elements} elements")
    # if show_stats:
    #     print(f"\nSpectrogram Statistics (NaN-ignored):")
    #     print(f"  Min: {np.nanmin(spectrogram_np):.2f} dB")
    #     print(f"  Max: {np.nanmax(spectrogram_np):.2f} dB")
    #     print(f"  Mean: {np.nanmean(spectrogram_np):.2f} dB")
    #     print(f"  Std: {np.nanstd(spectrogram_np):.2f} dB")


def steering_vector(theta_deg, M, d_lambda):
    m = np.arange(M)[:, None]  # (M,1)
    return np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(np.deg2rad(theta_deg)))  # (M,1)


def music_block(Xb, angles, d_lambda, num_expected_signals=1, diag_load=1e-6):
    """
    Xb: (K, M) snapshots complejos del bloque (K muestras, M sensores)
    angles: array de ángulos en grados (p.ej. -90..90)
    d_lambda: espaciamiento normalizado (d/λ)
    num_expected_signals: nº de fuentes esperadas (>=1). Debe cumplirse M > num_expected_signals
    diag_load: carga diagonal (se escala por la energía de R)
    """
    Xb = np.asarray(Xb, dtype=np.complex128)
    K, M = Xb.shape

    # --- Matriz de covarianza (Nr x Nr en tu base). Aquí Nr=M ---
    # En tu base: R = r @ r^H, con r de tamaño (Nr x K).
    # Como Xb es (K x M), r = Xb.T ⇒ R = Xb^H Xb / K (Hermítica, estable numéricamente).
    R = (Xb.conj().T @ Xb) / max(1, K)
    # Carga diagonal (escalada por energía) para estabilidad
    R += (diag_load * (np.trace(R).real / M)) * np.eye(M)

    J = np.fliplr(np.eye(R.shape[0]))  # Forward-Backward Averaging
    R = 0.5 * (R + J @ R.conj() @ J)

    # test simple: ¿hay fuente?
    w, v = np.linalg.eigh(R)
    ratio = (w[-1] / max(w[0], 1e-12)).real  # lambda_max / lambda_min
    if ratio < 2:  # umbral 1.5–2 (ajústalo)
        return np.ones(len(angles))  # espectro plano → “sin DOA”

    # --- Descomposición espectral (usar 'eigh' para Hermítica) ---
    w, v = np.linalg.eigh(R)  # autovalores ya en orden ascendente
    d = int(np.clip(num_expected_signals, 0, M - 1))  # nº fuentes; garantizar M-d >= 1
    # Subespacio de ruido: los M-d eigenvectores de autovalores más pequeños
    Vn = v[:, :M - d]  # (M, M-d)

    # Proyector de ruido (constante en el barrido)
    Pn = Vn @ Vn.conj().T  # (M, M)

    # --- Barrido angular (usamos tus 'angles' en º, no -π..π directamente) ---
    ang = np.asarray(angles, dtype=float)
    P = np.empty(ang.size, dtype=float)
    idx = np.arange(M).reshape(-1, 1)  # (M,1)

    for i, th_deg in enumerate(ang):
        th = np.deg2rad(th_deg)
        # Vector director ULA, elementos en posiciones 0, d, 2d, ... (referencia broadside)
        # Si ves el pico espejado, invierte el signo del exponente.
        a = np.exp(-2j * np.pi * d_lambda * idx * np.sin(th))  # (M,1)
        denom = np.real((a.conj().T @ Pn @ a)[0, 0])
        P[i] = 1.0 / max(denom, 1e-12)  # métrica MUSIC (lineal)

    return P


def publish_pred(label_id: int):
    body = {
        "label": classes[int(label_id)] if 0 <= int(label_id) < len(classes) else str(label_id),
    }
    # 1) directo al hub (instantáneo para el WS)
    try:
        pred_hub.set_last(body)
    except Exception as e:
        print("[pipeline] pred_hub.set_last error:", e)

    # 2) opcional: POST al endpoint (desacoplar procesos / logs)
    try:
        requests.post(PRED_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_pred HTTP error:", e)

def publish_doa(angle_deg: float):

    body = {
        "angle_deg": float(angle_deg),
    }
    try:
        doa_hub.set_last(body)
    except Exception as e:
        print("[pipeline] doa_hub.set_last error:", e)
    try:
        requests.post(DOA_INGEST_URL, json=body, timeout=0.5)
    except Exception as e:
        print("[pipeline] publish_doa HTTP error:", e)


# ---------------- BladeRF Configs ------------------------------------------------------------------------------------------------------------
# --- Crear ambos canales RX ---
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))  # RX 2
rx1 = sdr.Channel(_bladerf.CHANNEL_RX(0))  # RX1 (antena 1)

# Configs
sample_rate = 40e6
center_freq = 2440000000  # 2445.5 - 2455.5 #250 de 2455500000 # 250 de 2460000000
gain = 30  # -15 a 60 dB
num_samples = int(3e6)

step_freq = 20000000  # 20MHz
min_freq = 2400000000
max_freq = 2480000000
direction = 1

# Mismos parámetros en ambos canales
for ch in (rx1, rx_ch):
    ch.frequency = center_freq
    ch.sample_rate = sample_rate
    ch.bandwidth = sample_rate / 2
    ch.gain_mode = _bladerf.GainMode.Manual
    ch.gain = gain

# --- Sync: 2 Rx (MIMO) ---
sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                fmt=_bladerf.Format.SC16_Q11,  # int16s
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500)

# Habilitar ambos front-ends
rx1.enable = False
rx_ch.enable = True

# --- Recepción y desentrelazado ---
bytes_per_sample = 4  # int16 I + int16 Q
buf = bytearray(1024 * bytes_per_sample)

# ---------------- Variables matematicas y inicializacion de la clase transformadas de espectrogramas ------------------------------------------------------------------------------------------------------------

transform = transform_spectrogram(
    device="cpu",
    n_fft=shape,
    win_length=shape,
    hop_length=hop,
    window_fn=torch.hann_window,
    power=None,
    normalized=False,
    center=False,
    onesided=False
)

c = 299_792_458.0
fc_hz = center_freq  # <-- Frecuencia central
d_cm = 6.14  # <-- Separacion de antenas
lam = c / fc_hz
d_lambda = (d_cm / 100.0) / lam
# print("d_lambda: ", d_lambda)
block_size = 4096
angles = np.linspace(-90, 90, 721)  # malla más fina

# ---------------- LOOP PRINCIPAL ------------------------------------------------------------------------------------------------------------

print("Starting dual-RX receive")
while True:
    print("[FRECUENCIA]: ", center_freq)
    rx1.enable = False
    rx_ch.enable = False
    sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                    fmt=_bladerf.Format.SC16_Q11,  # int16s
                    num_buffers=16,
                    buffer_size=8192,
                    num_transfers=8,
                    stream_timeout=3500)
    rx1.enable = False
    rx_ch.enable = True
    x = np.zeros(num_samples, dtype=np.complex64)  # storage for IQ samples
    num_samples_read = 0
    while True:
        if num_samples > 0 and num_samples_read == num_samples:
            break
        elif num_samples > 0:
            num = min(len(buf) // bytes_per_sample, num_samples - num_samples_read)
        else:
            num = len(buf) // bytes_per_sample
        sdr.sync_rx(buf, num)  # Read into buffer
        samples = np.frombuffer(buf, dtype=np.int16)
        samples = samples[0::2] + 1j * samples[1::2]  # Convert to complex type
        samples /= 2048.0  # Scale to -1 to 1 (12-bit ADC)
        x[num_samples_read:num_samples_read + num] = samples[0:num]  # Store buf in samples array
        num_samples_read += num

    # Separar I y Q
    I = np.real(x)
    Q = np.imag(x)

    sample = torch.tensor(np.stack([I, Q], axis=0))

    spec = transform(sample)

    if center_freq >= max_freq:
        center_freq = max_freq
        direction = -1
    elif center_freq <= min_freq:
        center_freq = min_freq
        direction = 1

    spec = torch.tensor(spec, dtype=torch.float32).view(1, 1, 512, 512)
    with torch.no_grad():
        # Binario -----------------------
        # outputs = model(spec).view(-1)
        # preds = (torch.sigmoid(outputs) > 0.5).float()
        # probs = torch.sigmoid(outputs)
        # Multi-Class ----------------------------
        # outputs = model(spec)
        # probs = torch.softmax(outputs, dim=1)  # Convert to probabilities
        # print(probs)
        # pred = probs.argmax(dim=1)
        # 3 clases--------------------
        outputs = model(spec)  # (N, C)
        preds = outputs.argmax(dim=1)
        probs = torch.softmax(outputs, dim=1)

    drone_predict = dictionaryDrones[preds.item()]
    print(drone_predict)
    frame = make_psd_frame(x=x, center_hz=center_freq, sample_rate=sample_rate, nfft=4096,
                           drone_id=dictionaryDrones[preds.item()], schema_version="1.0")
    psd_hub.set_last(frame)

    publish_psd(x, center_freq, sample_rate, nfft=4096,
                drone_id=dictionaryDrones[preds.item()])

    preds = outputs.argmax(dim=1)
    probs = torch.softmax(outputs, dim=1)

    # mapea a tu diccionario de clases
    classes = [dictionaryDrones[i] for i in range(NUM_CLASSES)]
    label_id = int(preds.item())
    probs_np = probs[0].detach().cpu().numpy()

    # publica predicción
    # publish_pred(label_id=label_id)

    visualize_spectrogram(
        spectrogram=spec.view(1, shape, shape),
        class_name=f"{drone_predict}")

    if (drone_predict == 'Noise' or drone_predict == "Jammer"):
        center_freq += direction * step_freq
        for ch in (rx1, rx_ch):
            ch.frequency = center_freq
        continue

    if (drone_predict != 'Noise' and drone_predict != 'Jammer'):
        # Nuevas variables si cambio de frecuencia:
        fc_hz = center_freq
        lam = c / fc_hz
        d_lambda = (d_cm / 100.0) / lam
        rx1.enable = False
        rx_ch.enable = False
        sdr.sync_config(_bladerf.ChannelLayout.RX_X2,
                        _bladerf.Format.SC16_Q11,
                        num_buffers=32,
                        buffer_size=8192,
                        num_transfers=16,
                        stream_timeout=3500)
        rx1.enable = True
        rx_ch.enable = True

        num_samples_doa = 4096
        buf = bytearray(num_samples_doa * 8)
        sdr.sync_rx(buf, num_samples_doa)
        raw = np.frombuffer(buf, dtype=np.int16).reshape(-1, 4)
        x1 = (raw[:, 0] + 1j * raw[:, 1]) / 2048.0
        x2 = (raw[:, 2] + 1j * raw[:, 3]) / 2048.0

        # normaliza potencia
        x1 /= np.sqrt(np.mean(np.abs(x1) ** 2))
        x2 /= np.sqrt(np.mean(np.abs(x2) ** 2))

        xc = correlate(x1, x2, mode='full')
        lag = np.argmax(np.abs(xc)) - (len(x1) - 1)
        phi = np.angle(np.vdot(x1, x2))  # fase media radiantes
        phi_deg = np.degrees(phi)  # angulo
        print(f"[LECTURA 2 Antennas] lag_muestras={lag}, fase_promedio={phi_deg:.3f} grados")

        X_full = np.stack([x1, x2], axis=1)  # shape (N, 2)
        num_blocks = X_full.shape[0] // block_size
        acc = np.zeros(len(angles), dtype=float)

        num_signals = 1
        for b in range(num_blocks):
            sl = slice(b * block_size, (b + 1) * block_size)
            Xb = X_full[sl, :]  # (K, 2)
            acc += music_block(Xb, angles, d_lambda, num_expected_signals=num_signals, diag_load=1e-6)

        P_music = acc / max(1, num_blocks)
        psr_db = 10 * np.log10(P_music.max() / (np.median(P_music) + 1e-12))  # peak/median
        if psr_db < 8:  # 6–8 dB típico
            print(f"Sin DOA confiable (PSR={psr_db:.1f} dB).")
        # ===== Estimar ángulo a partir de music =====
        P_lin = np.asarray(P_music, float)
        ang = np.asarray(angles, float)
        step = ang[1] - ang[0]

        # Evitar picos falsos en los bordes (+/-90°)
        edge = 2
        i_search = np.arange(edge, len(P_lin) - edge)
        i0 = i_search[np.argmax(P_lin[i_search])]  # índice del pico

        # Refinamiento sub-bin (parabólico) en POTENCIA lineal
        if 0 < i0 < len(P_lin) - 1:
            y1, y2, y3 = P_lin[i0 - 1], P_lin[i0], P_lin[i0 + 1]
            denom = (y1 - 2 * y2 + y3)
            delta = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
        else:
            delta = 0.0

        theta_hat = ang[i0] + delta * step  # grados
        print("El angulo en que se encuentra el dron es: ", theta_hat)

        publish_doa(
            angle_deg=float(theta_hat),
        )

        # ---------------------------
        # Gráfica polar semicircular en dB
        # ---------------------------
        dr = 15  # rango dinámico mostrado en dB (ajústalo a gusto)
        P_db = 10 * np.log10(np.maximum(P_music, 1e-12))
        P_db = np.clip(P_db - P_db.max(), -dr, 0)  # normaliza pico a 0 dB y recorta a [-dr, 0]

        theta = np.deg2rad(angles)
        MILITARY_GREEN = "#4B9920"  # verde militar (Army Green)

        fig = plt.figure(figsize=(16, 7), facecolor="none")
        ax = fig.add_subplot(111, projection="polar", facecolor="none")
        ax.patch.set_alpha(0.0)  # fondo del eje transparente
        fig.patch.set_alpha(0.0)  # fondo de la figura transparente
        ax.plot(theta, P_db, linewidth=2, color=MILITARY_GREEN)

        # Semicírculo superior: 0° arriba, -90° izq, +90° der (sentido horario)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        # Escala radial en dB (0 dB afuera, -dr dB hacia adentro)
        ax.set_rlim(-dr, 0)
        ax.set_rlabel_position(180)  # etiquetas a la izquierda, como en tu ejemplo
        ax.grid(True)
        plt.tight_layout()

        if SAVE_PLOTS:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = PLOT_DIR / f"direction_{ts}.png"
            plt.savefig(out_path, dpi=120)
            print(f"[pipeline] saved_plot={out_path}")
        plt.show()
sdr.close()

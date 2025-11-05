from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Spectrogram
import time



SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(os.getenv("PIPELINE_PLOT_DIR", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/plots"))
if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 1
NUM_CLASSES = 11

# --- Apertura robusta del bladeRF ---
os.environ.setdefault("LIBUSB_DEBUG", "3")
DEV_ID = os.getenv("BLADERF_DEVICE")

try:
    if DEV_ID:
        sdr = _bladerf.BladeRF(DEV_ID)
    else:
        sdr = _bladerf.BladeRF()
except _bladerf.NoDevError:
    print("[pipeline] No se encontraron dispositivos bladeRF desde este proceso.")
    try:
        out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
        print("[pipeline] bladeRF-cli -p stdout:\n", out.stdout)
        print("[pipeline] bladeRF-cli -p stderr:\n", out.stderr)
    except Exception as e:
        print("[pipeline] No pude ejecutar bladeRF-cli -p:", e)
    sys.exit(2)


# -----------------------------
# Carga del modelo trazado TorchScript
# -----------------------------
MODEL_TRACE = "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/weights/ConvNeXtTiny_traced.pt"

try:
    convnext_tiny_model = torch.jit.load(MODEL_TRACE, map_location="cpu")
    convnext_tiny_model.eval()
    print(f"[pipeline] Modelo trazado cargado correctamente desde: {MODEL_TRACE}")
except Exception as e:
    print(f"[pipeline] Error cargando modelo trazado: {e}")
    sys.exit(1)


# -----------------------------
# Transformador de espectrograma
# -----------------------------
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
        self.spec = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            window_fn=window_fn, power=power, normalized=normalized,
            center=center, onesided=onesided
        ).to(device=device)
        self.win_length = win_length
        self.epsilon = 1e-12

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        iq_signal = iq_signal[0, :] + (1j * iq_signal[1, :])
        spec = self.spec(iq_signal)
        spec = torch.view_as_real(spec)
        spec = torch.moveaxis(spec, 2, 0)
        spec = 10 * torch.log10(spec ** 2 + self.epsilon)
        spec = np.sqrt(spec[0]**2 + spec[1]**2)
        return spec


# -----------------------------
# Visualización
# -----------------------------
def visualize_spectrogram(
    spectrogram: torch.Tensor,
    label: float,
    time_duration: float = 75e-3,
    n_fft=1024,
    win_length=1024,
    hop_length=2930,
    sample_freq=40e6,
    show_stats: bool = True,
    figsize: tuple = (16, 4)
) -> None:
    ch_names = {0: "Signal"}
    spectrogram_np = spectrogram.detach().cpu().numpy()
    n_freq_bins = spectrogram_np.shape[1]
    n_time_bins = spectrogram_np.shape[2]

    time_per_fft = hop_length / sample_freq
    time_axis = np.arange(n_time_bins) * time_per_fft
    time_duration_ms = time_axis * 1000
    freqs = sample_freq / win_length * np.arange(-n_fft / 2, n_fft / 2)
    freqs_in_mhz = freqs / 1e6

    if len(freqs_in_mhz) > n_freq_bins:
        freqs_in_mhz = freqs_in_mhz[:n_freq_bins]

    t = time_duration_ms
    f = freqs_in_mhz

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ch in range(min(2, spectrogram_np.shape[0])):
        im = axes[ch].imshow(
            spectrogram_np[ch],
            aspect='auto',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]],
            interpolation='nearest'
        )
        axes[ch].set_ylabel('Frequency [MHz]')
        axes[ch].set_xlabel('Time [ms]')
        axes[ch].set_xticks(np.arange(0, time_duration_ms[-1] if len(time_duration_ms) > 0 else 75, step=10))
        axes[ch].set_title(f'{ch_names[ch]} Spectrogram, label:{label}')
        plt.colorbar(im, ax=axes[ch], label='Power [dB]')

    fig.suptitle(f'Spectrogram')
    plt.tight_layout()

    if show_stats:
        print(f"\nSpectrogram Statistics (NaN-ignored):")
        print(f"  Min: {np.nanmin(spectrogram_np):.2f} dB")
        print(f"  Max: {np.nanmax(spectrogram_np):.2f} dB")
        print(f"  Mean: {np.nanmean(spectrogram_np):.2f} dB")
        print(f"  Std: {np.nanstd(spectrogram_np):.2f} dB")

    plt.show()
    plt.close(fig)


# -----------------------------
# Configuración del receptor BladeRF
# -----------------------------
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))  # RX 2

sample_rate = 40e6
center_freq = 2440000000
step_freq = 40000000
gain = 30
num_samples = int(3e6)

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate / 2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=8192,
    num_transfers=8,
    stream_timeout=3500
)

buf = bytearray(1024 * 4)
print("Starting receive")
rx_ch.enable = True
direction = 1

min_freq = 2400000000
max_freq = 2480000000

start_time2 = time.time()
b = 1

# -----------------------------
# Loop principal
# -----------------------------
while True:
    start_time = time.time()
    print("Frecuencia central: ", center_freq)
    x = np.zeros(num_samples, dtype=np.complex64)
    num_samples_read = 0

    while True:
        if num_samples > 0 and num_samples_read == num_samples:
            break
        elif num_samples > 0:
            num = min(len(buf) // 4, num_samples - num_samples_read)
        else:
            num = len(buf) // 4
        sdr.sync_rx(buf, num)
        samples = np.frombuffer(buf, dtype=np.int16)
        samples = samples[0::2] + 1j * samples[1::2]
        samples /= 2048.0
        x[num_samples_read:num_samples_read + num] = samples[0:num]
        num_samples_read += num

    I = np.real(x)
    Q = np.imag(x)
    sample = torch.tensor(np.stack([I, Q], axis=0))

    transform = transform_spectrogram(
        device="cpu",
        n_fft=1024,
        win_length=1024,
        hop_length=2930,
        window_fn=torch.hann_window,
        power=None,
        normalized=False,
        center=False,
        onesided=False
    )
    spec = transform(sample)
    spectrogram_np = spec.detach().cpu().numpy()
    std = np.nanstd(spectrogram_np)

    if center_freq >= max_freq:
        center_freq = max_freq
        direction = -1
    elif center_freq <= min_freq:
        center_freq = min_freq
        direction = 1
        if b != 1:
            end_time2 = time.time()
            print(f"Tiempo de realizar un ciclo: {end_time2 - start_time2:.6f} segundos")
        b = 0

    original_shape = spec.shape
    print(spec.shape)

    spec = torch.tensor(spec, dtype=torch.float32).view(1, 1024, 1024)

    # --- Inferencia usando el modelo trazado ---
    with torch.no_grad():
        logit = convnext_tiny_model(spec)
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float()

    print(f"{pred=}, {prob=}")
    end_time = time.time()
    print(f"Tiempo de leer y realizar el espectrograma: {end_time - start_time:.6f} segundos")

    visualize_spectrogram(
        spectrogram=spec.view(original_shape),
        label=f"{pred} {prob}"
    )

from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F  # interpolate
import joblib
import sklearn
import sys
import subprocess
import os
from pathlib import Path
from datetime import datetime
from torchaudio.transforms import Spectrogram

# -----------------------------
# Guardado de plots por entorno
# -----------------------------
SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(
    os.getenv("PIPELINE_PLOT_DIR", ""))
if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 2
NUM_CLASSES = 2

# --- RUTA DEL SCALER (robusta por env, con fallback al lado del script) ---
# DEFAULT_SCALER = Path(__file__).resolve().parent / "scaler.save"
# SCALER_PATH = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER))).resolve()

# --- Apertura robusta del bladeRF ---
os.environ.setdefault("LIBUSB_DEBUG", "3")
DEV_ID = os.getenv("BLADERF_DEVICE", "*:serial=7b1b9ce290114baaab24bb18a37507c2").strip()
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

# ConvNeXt Tiny defined for finetuning
class ConvNeXtTinyFineTuner(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 2, num_classes: int = 10, init_stem: bool = False):
        super(ConvNeXtTinyFineTuner, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        # Replace classifier head
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes - 1)

        # Adapt stem to in_channels input channels
        old_stem = self.backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None
        )
        if init_stem:
            # Initialize new-stem with RGB weights averaged over the input channels
            with torch.no_grad():
                W = old_stem.weight
                W_mean = W.mean(dim=1, keepdim=True)
                new_stem.weight.copy_(W_mean.repeat(1, in_channels, 1, 1))
                if old_stem.bias is not None:
                    new_stem.bias.copy_(old_stem.bias)

        self.backbone.features[0][0] = new_stem

        # Freeze backbone except stem and classifier head
        self._freeze_backbone()

    def _freeze_backbone(self):
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze stem convolutional layer parameters
        for param in self.backbone.features[0][0].parameters():
            param.requires_grad = True
        # Unfreeze classifier head parameters
        for param in self.backbone.classifier[2].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x).squeeze()


convnext_tiny_model = ConvNeXtTinyFineTuner(pretrained=True, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                                            init_stem=False)
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/models/best_state.pth")
convnext_tiny_model.load_state_dict(torch.load(MODEL_WEIGHTS, weights_only=True, map_location=torch.device('cpu')))
convnext_tiny_model.eval()


# print(convnext_tiny_model)


class transform_spectrogram(torch.nn.Module):
    def __init__(
            self,
            device,
            n_fft=1024,
            win_length=1024,
            hop_length=976,
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

        # Parámetros
        order = 8
        ripple = 0.5  # dB de rizado en la banda pasante
        cutoff = 0.3  # frecuencia normalizada (0 a 1, donde 1 es Nyquist)
        # Diseño del filtro
        b, a = signal.cheby1(order, ripple, cutoff, btype='low', analog=False)
        iq_signal = signal.lfilter(b, a, iq_signal)

        # magnitudes = torch.abs(iq_signal)
        # print("Máximo antes de normalizar:", torch.max(magnitudes))
        # print("Mínimo antes de normalizar:", torch.min(magnitudes))
        # power_mean = torch.mean(torch.abs(iq_signal)**2)
        # iq_signal = iq_signal / torch.sqrt(power_mean)
        # magnitudes = torch.abs(iq_signal)
        # print("Máximo despues de normalizar:", torch.max(magnitudes))
        # print("Mínimo despues de normalizar:", torch.min(magnitudes))
        # Factor de decimación
        decimation_factor = 3
        iq_signal_decimated = torch.tensor(signal.decimate(iq_signal, decimation_factor, ftype='fir', zero_phase=True))
        spec = self.spec(iq_signal_decimated)
        spec = torch.view_as_real(spec)
        spec = torch.moveaxis(spec, 2, 0)
        spec = torch.log10(spec ** 2 + self.epsilon)  # Convert to dB scale
        return spec


def visualize_spectrogram(spectrogram: torch.Tensor, label: float, time_duration: float = 75e-3,
                          n_fft=1024, win_length=1024, hop_length=976, sample_freq=13.33e6,
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
    ch_names = {0: "I", 1: "Q"}

    # print(f"Spectrogram shape: {spectrogram.shape}")
    # rint(f"Label: {label}")

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

    # Ensure frequency axis matches spectrogram dimensions
    if len(freqs_in_mhz) > n_freq_bins:
        freqs_in_mhz = freqs_in_mhz[:n_freq_bins]

    t = time_duration_ms  # Time in milliseconds
    f = freqs_in_mhz  # Frequency in MHz

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ch in range(min(2, spectrogram_np.shape[0])):
        im = axes[ch].pcolormesh(t, f, spectrogram_np[ch], shading='gouraud')
        axes[ch].set_ylabel('Frequency [MHz]')
        axes[ch].set_xlabel('Time [ms]')
        # Set x-axis ticks in milliseconds
        max_time = time_duration_ms[-1] if len(time_duration_ms) > 0 else 75
        axes[ch].set_xticks(np.arange(0, max_time, step=10))  # every 10 ms
        axes[ch].set_title(f'{ch_names[ch]} Spectrogram, label:{label}')
        axes[ch].set_ylim([0, f.max()])  # Only positive frequencies
        plt.colorbar(im, ax=axes[ch], label='Power [dB]')

    fig.suptitle(f'Spectrogram')
    plt.tight_layout()

    # Print statistics if requested (ignoring NaN values)
    # Check for NaN values
    nan_count = np.isnan(spectrogram_np).sum()
    total_elements = spectrogram_np.size
    # print(f"NaN values in spectrogram: {nan_count} out of {total_elements} elements")
    if show_stats:
        print(f"\nSpectrogram Statistics (NaN-ignored):")
        print(f"  Min: {np.nanmin(spectrogram_np):.2f} dB")
        print(f"  Max: {np.nanmax(spectrogram_np):.2f} dB")
        print(f"  Mean: {np.nanmean(spectrogram_np):.2f} dB")
        print(f"  Std: {np.nanstd(spectrogram_np):.2f} dB")

    if SAVE_PLOTS:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = PLOT_DIR / f"spectrogram_{ts}.png"
        plt.savefig(out_path, dpi=120)
        # print(f"[pipeline] saved_plot={out_path}")
    plt.show()
    # En modo headless cerramos siempre (evita warning de FigureCanvasAgg)
    plt.close(fig)


# --- Usar la instancia 'sdr' ya creada arriba (no re-abrir de nuevo) ---
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))  # RX 2

# Configs
sample_rate = 40e6
center_freq = 2400000000
step_freq = 20000000  # 20MHz
gain = 30  # -15 a 60 dB
num_samples = int(3e6)  # hop_lenght

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate / 2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

# Setup synchronous stream
sdr.sync_config(layout=_bladerf.ChannelLayout.RX_X2,  # o RX_X2
                fmt=_bladerf.Format.SC16_Q11,  # int16s
                num_buffers=16,
                buffer_size=8192,
                num_transfers=8,
                stream_timeout=3500)

# Create receive buffer
bytes_per_sample = 4  # int16 I + int16 Q
buf = bytearray(1024 * bytes_per_sample)

print("Starting receive")
rx_ch.enable = True
direction = 1

min_freq = 2400000000
max_freq = 2480000000

while True:
    print("Frecuencia central: ", center_freq)
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

    # Espectrogramas

    # if not SCALER_PATH.exists():
    #    print(f"[pipeline] No encontré scaler en: {SCALER_PATH}")
    #    sys.exit(3)
    # scaler = joblib.load(str(SCALER_PATH))

    sample_rate_down = sample_rate / 1
    transform = transform_spectrogram(
        device="cpu",
        n_fft=1024,
        win_length=1024,
        hop_length=976,  # 73.6 us
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

    if (std < 1.2):
        center_freq += direction * step_freq
        rx_ch.frequency = center_freq
        continue

    # --- PREPROCESAMIENTO: redimensionar ANTES del scaler ---
    original_shape = spec.shape
    # print(spec.shape)

    # sample_reshaped = spec.reshape(1, -1).numpy()
    # print(sample_reshaped.shape)
    # sample_scaled = scaler.transform(sample_reshaped)
    spec = torch.tensor(spec, dtype=torch.float32).view(1, 2, 1024, 1024)

    # print("sample_shape: ", spec.shape)

    logit = convnext_tiny_model(spec)
    prob = torch.sigmoid(logit)
    pred = (prob > 0.5).float()
    print(f"{pred=}, {prob=}")
    if pred == 0:
        center_freq += direction * step_freq
        rx_ch.frequency = center_freq
    visualize_spectrogram(
        spectrogram=spec.view(original_shape),
        label=f"{pred} {prob}"
    )
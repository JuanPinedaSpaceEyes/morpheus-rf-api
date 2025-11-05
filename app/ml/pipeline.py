#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
from torchaudio.transforms import Spectrogram
from bladerf import _bladerf


SAVE_PLOTS = os.getenv("PIPELINE_SAVE_PLOTS", "1") == "1"
PLOT_DIR = Path(os.getenv("PIPELINE_PLOT_DIR", "./plots"))
if SAVE_PLOTS:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

IN_CHANNELS = 1
NUM_CLASSES = 11

MODEL_TRACE_PATH = os.getenv(
    "MODEL_TRACE_PATH",
    "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/models/convnext_traced.pt"
)
device = torch.device("cpu")


print(f"[pipeline] Loading traced model from: {MODEL_TRACE_PATH}")
model = torch.jit.load(MODEL_TRACE_PATH, map_location=device)
model.eval()


class TransformSpectrogram(torch.nn.Module):
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
        self.spec = Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided
        ).to(device=device)
        self.epsilon = 1e-12

    def forward(self, iq_signal: torch.Tensor) -> torch.Tensor:
        # Si la señal ya es compleja, no tocarla
        if not torch.is_complex(iq_signal):
            if iq_signal.ndim == 2 and iq_signal.shape[0] == 2:
                iq_signal = iq_signal[0, :] + 1j * iq_signal[1, :]
            else:
                iq_signal = iq_signal.to(torch.complex64)

        # Filtro Chebyshev
        order = 8
        ripple = 0.5
        cutoff = 0.3
        b, a = signal.cheby1(order, ripple, cutoff, btype='low', analog=False)
        iq_signal = signal.lfilter(b, a, iq_signal)

        # Decimación
        decimation_factor = 3
        iq_signal_decimated = torch.tensor(
            signal.decimate(iq_signal, decimation_factor, ftype='fir', zero_phase=True)
        )

        # Espectrograma complejo → magnitud logarítmica
        spec = self.spec(iq_signal_decimated)
        spec = torch.view_as_real(spec)
        spec = torch.moveaxis(spec, 2, 0)
        spec = torch.log10(spec ** 2 + self.epsilon)

        # Fusionar canales complejos → magnitud (1 canal)
        spec = spec.pow(2).sum(dim=0, keepdim=True).sqrt()
        return spec



def visualize_spectrogram(spectrogram: torch.Tensor, label: str, sample_freq=13.33e6,
                          n_fft=1024, hop_length=976, figsize=(12, 4)) -> None:
    spectrogram_np = spectrogram.detach().cpu().numpy()
    n_freq_bins = spectrogram_np.shape[1]
    n_time_bins = spectrogram_np.shape[2]

    time_per_fft = hop_length / sample_freq
    time_axis = np.arange(n_time_bins) * time_per_fft * 1000  # ms
    freqs = sample_freq / n_fft * np.arange(-n_fft / 2, n_fft / 2)
    freqs_mhz = freqs[:n_freq_bins] / 1e6

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(time_axis, freqs_mhz, spectrogram_np[0], shading='gouraud')
    ax.set_ylabel('Frequency [MHz]')
    ax.set_xlabel('Time [ms]')
    plt.colorbar(im, ax=ax, label='Power [dB]')
    ax.set_title(f'Spectrogram | {label}')

    if SAVE_PLOTS:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = PLOT_DIR / f"spectrogram_{ts}.png"
        plt.savefig(out_path, dpi=120)
        print(f"[pipeline] Saved plot: {out_path}")

    plt.show()
    plt.close(fig)


os.environ.setdefault("LIBUSB_DEBUG", "3")
DEV_ID = os.getenv("BLADERF_DEVICE", "*:serial=7b1b9ce290114baaab24bb18a37507c2").strip()

try:
    sdr = _bladerf.BladeRF(DEV_ID) if DEV_ID else _bladerf.BladeRF()
except _bladerf.NoDevError:
    print("[pipeline] No se encontró ningún dispositivo BladeRF.")
    try:
        out = subprocess.run(["bladeRF-cli", "-p"], capture_output=True, text=True)
        print(out.stdout)
        print(out.stderr)
    except Exception as e:
        print("[pipeline] No se pudo ejecutar bladeRF-cli -p:", e)
    sys.exit(2)

rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(1))
sample_rate = 40e6
center_freq = 2400000000
step_freq = 20000000
gain = 30
num_samples = int(3e6)

rx_ch.frequency = center_freq
rx_ch.sample_rate = sample_rate
rx_ch.bandwidth = sample_rate / 2
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = gain

sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X2,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=8192,
    num_transfers=8,
    stream_timeout=3500
)

buf = bytearray(1024 * 4)
print("[pipeline] Starting receive loop...")

rx_ch.enable = True
direction = 1
min_freq = 2400000000
max_freq = 2480000000


transform = TransformSpectrogram(device="cpu")

while True:
    print(f"[pipeline] Frecuencia central: {center_freq}")
    x = np.zeros(num_samples, dtype=np.complex64)
    num_samples_read = 0

    while num_samples_read < num_samples:
        num = min(len(buf) // 4, num_samples - num_samples_read)
        sdr.sync_rx(buf, num)
        samples = np.frombuffer(buf, dtype=np.int16)
        samples = samples[0::2] + 1j * samples[1::2]
        samples /= 2048.0
        x[num_samples_read:num_samples_read + num] = samples[:num]
        num_samples_read += num

    sample = torch.tensor(x, dtype=torch.complex64).unsqueeze(0)

    # Espectrograma (1 canal)
    spec = transform(sample)
    spectrogram_np = spec.detach().cpu().numpy()
    std = np.nanstd(spectrogram_np)

    if center_freq >= max_freq:
        center_freq = max_freq
        direction = -1
    elif center_freq <= min_freq:
        center_freq = min_freq
        direction = 1

    if std < 1.2:
        center_freq += direction * step_freq
        rx_ch.frequency = center_freq
        continue

    spec = spec.clone().detach().float()

    if spec.ndim == 3:
        spec = spec.unsqueeze(0)  
    elif spec.ndim == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)  

    spec = torch.nn.functional.interpolate(
        spec, size=(1024, 1024), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        output = model(spec)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()

    label_str = f"pred={pred.item():.0f}, prob={prob.item():.3f}"
    print(f"[pipeline] {label_str}")

    # Visualización
    visualize_spectrogram(spec, label_str)

    # Actualizar frecuencia
    if pred == 0:
        center_freq += direction * step_freq
        rx_ch.frequency = center_freq

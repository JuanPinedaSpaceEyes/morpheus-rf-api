#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import signal
from dataclasses import dataclass
from threading import Thread, Event
from typing import Optional, Tuple, List

import numpy as np

try:
    import bladerf
except Exception as e:
    print("ERROR: No se pudo importar 'bladerf'. Instala:  pip install pybladeRF")
    print("Detalle:", e)
    sys.exit(1)


@dataclass
class DevCfg:
    name: str
    serial: Optional[str]
    freq_hz: float
    sr_hz: float
    bw_hz: Optional[float]  # puede venir None -> se calcula segura
    gain_db: int
    nfft: int
    topk: int
    threshold_db: float
    buffer_size: int = 8192
    num_buffers: int = 16
    num_transfers: int = 8
    stream_timeout_ms: int = 3500


class BladeRFWorker:
    def __init__(self, cfg: DevCfg):
        self.cfg = cfg
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

        self.dev: Optional[bladerf.BladeRF] = None
        self.rx = None

        self.window = np.hanning(self.cfg.nfft).astype(np.float32)

    def _safe_bw(self, sr_hz: float, bw_hz_req: Optional[float]) -> int:
        # Límite superior seguro ≈ 0.8 * SR (y nunca > 56 MHz)
        max_rf_bw = int(min(sr_hz * 0.8, 56e6))
        if bw_hz_req is None:
            return max_rf_bw
        return int(min(bw_hz_req, max_rf_bw))

    def _open_and_configure(self):
        self.dev = bladerf.BladeRF(f"serial={self.cfg.serial}") if self.cfg.serial else bladerf.BladeRF()
        self.rx = self.dev.Channel(bladerf.CHANNEL_RX(0))

        # Frecuencia primero
        self.rx.frequency = self.cfg.freq_hz

        # Sample rate
        self.rx.sample_rate = self.cfg.sr_hz
        # Leer SR efectiva (puede ajustar a valores discretos internos)
        sr_eff = float(self.rx.sample_rate)

        # Bandwidth “segura”
        bw_target = self._safe_bw(sr_eff, self.cfg.bw_hz)

        # Modo de ganancia manual + valor
        self.rx.gain_mode = bladerf.GainMode.Manual
        self.rx.gain = int(self.cfg.gain_db)

        # Intento 1: BW objetivo seguro
        try:
            self.rx.bandwidth = bw_target
        except Exception as e1:
            # Intento 2: 0.6 * SR
            try:
                self.rx.bandwidth = int(sr_eff * 0.6)
            except Exception as e2:
                # Intento 3: SR/2 o 10 MHz, lo que sea menor
                try:
                    self.rx.bandwidth = int(min(sr_eff * 0.5, 10e6))
                except Exception as e3:
                    # Si también falla, re-lanza último error para ver el mensaje
                    raise e3

        # Config de streaming
        self.dev.sync_config(
            layout=bladerf.ChannelLayout.RX_X1,
            fmt=bladerf.Format.SC16_Q11,
            num_buffers=self.cfg.num_buffers,
            buffer_size=self.cfg.buffer_size,
            num_transfers=self.cfg.num_transfers,
            stream_timeout=self.cfg.stream_timeout_ms
        )

        # Habilitar RX
        self.rx.enable = True

        # Imprimir efectivos
        print(f"[{self.cfg.name}] Efectivos -> SR: {float(self.rx.sample_rate)/1e6:.3f} MS/s | "
              f"BW: {float(self.rx.bandwidth)/1e6:.3f} MHz | Freq: {float(self.rx.frequency)/1e9:.3f} GHz")

    def _close(self):
        try:
            if self.rx:
                self.rx.enable = False
        except Exception:
            pass
        try:
            if self.dev:
                self.dev.close()
        except Exception:
            pass
        self.dev = None
        self.rx = None

    def _iq_block(self) -> np.ndarray:
        nsamps = self.cfg.buffer_size
        buf = np.empty(nsamps * 2, dtype=np.int16)
        self.dev.sync_rx(buf, nsamps)
        i = buf[0::2].astype(np.float32) / 2048.0
        q = buf[1::2].astype(np.float32) / 2048.0
        return i + 1j * q

    def _freq_axis_hz(self, n: int) -> np.ndarray:
        sr = float(self.rx.sample_rate)
        fc = float(self.rx.frequency)
        freqs = np.fft.fftfreq(n, d=1.0 / sr)
        return np.fft.fftshift(freqs) + fc

    def _detect_peaks(self, power_db: np.ndarray, freqs_hz: np.ndarray) -> List[Tuple[float, float]]:
        noise_floor = np.median(power_db)
        mask = power_db > (noise_floor + self.cfg.threshold_db)
        if not np.any(mask):
            return []
        cand_idx = np.where(mask)[0]
        k = min(self.cfg.topk, cand_idx.size)
        top_idx = cand_idx[np.argpartition(power_db[cand_idx], -k)[-k:]]
        top_idx = top_idx[np.argsort(power_db[top_idx])[::-1]]
        return [(float(freqs_hz[i]), float(power_db[i])) for i in top_idx]

    def _loop(self):
        print(f"[{self.cfg.name}] Iniciando captura en {self.cfg.freq_hz/1e9:.3f} GHz | "
              f"sr={self.cfg.sr_hz/1e6:.1f} MS/s | bw_req={(self.cfg.bw_hz or 0)/1e6:.1f} MHz | "
              f"gain={self.cfg.gain_db} dB | nfft={self.cfg.nfft}")
        try:
            self._open_and_configure()
            freq_axis = self._freq_axis_hz(self.cfg.nfft)
            while not self.stop_event.is_set():
                iq = self._iq_block()
                if iq.size < self.cfg.nfft:
                    iq = np.pad(iq, (0, self.cfg.nfft - iq.size), mode='constant')
                else:
                    iq = iq[:self.cfg.nfft]
                xw = iq * self.window
                spec = np.fft.fftshift(np.fft.fft(xw, n=self.cfg.nfft))
                power_db = 20.0 * np.log10(np.abs(spec) + 1e-12)
                peaks = self._detect_peaks(power_db, freq_axis)
                if peaks:
                    ts = time.strftime("%H:%M:%S")
                    lines = [f"{f/1e6:10.3f} MHz  {p:6.1f} dB" for f, p in peaks]
                    print(f"[{ts}] [{self.cfg.name}] Picos ({len(peaks)}): " + " | ".join(lines))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[{self.cfg.name}] ERROR en loop:", repr(e))
        finally:
            self._close()
            print(f"[{self.cfg.name}] Captura detenida.")

    def start(self):
        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self._close()


def parse_args():
    p = argparse.ArgumentParser(description="Captura simultánea con 2 bladeRF y detección de picos.")
    p.add_argument("--serial1", type=str, default=None)
    p.add_argument("--serial2", type=str, default=None)
    p.add_argument("--freq1", type=float, default=2.43e9)
    p.add_argument("--freq2", type=float, default=5.80e9)
    p.add_argument("--sr1", type=float, default=20e6)
    p.add_argument("--sr2", type=float, default=20e6)
    p.add_argument("--bw1", type=float, default=None, help="Si None, usa 0.8×SR1 (cap a 56 MHz)")
    p.add_argument("--bw2", type=float, default=None, help="Si None, usa 0.8×SR2 (cap a 56 MHz)")
    p.add_argument("--gain1", type=int, default=30)
    p.add_argument("--gain2", type=int, default=30)
    p.add_argument("--nfft", type=int, default=8192)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--threshold_db", type=float, default=12.0)
    p.add_argument("--buffer_size", type=int, default=8192)
    p.add_argument("--num_buffers", type=int, default=16)
    p.add_argument("--num_transfers", type=int, default=8)
    p.add_argument("--timeout_ms", type=int, default=3500)
    p.add_argument("--seconds", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cfg1 = DevCfg(
        name="2.4GHz", serial=args.serial1, freq_hz=args.freq1,
        sr_hz=args.sr1, bw_hz=args.bw1, gain_db=args.gain1,
        nfft=args.nfft, topk=args.topk, threshold_db=args.threshold_db,
        buffer_size=args.buffer_size, num_buffers=args.num_buffers,
        num_transfers=args.num_transfers, stream_timeout_ms=args.timeout_ms
    )
    cfg2 = DevCfg(
        name="5.8GHz", serial=args.serial2, freq_hz=args.freq2,
        sr_hz=args.sr2, bw_hz=args.bw2, gain_db=args.gain2,
        nfft=args.nfft, topk=args.topk, threshold_db=args.threshold_db,
        buffer_size=args.buffer_size, num_buffers=args.num_buffers,
        num_transfers=args.num_transfers, stream_timeout_ms=args.timeout_ms
    )

    w1 = BladeRFWorker(cfg1)
    w2 = BladeRFWorker(cfg2)

    stop_all = Event()
    def _sigint(sig, frame): stop_all.set()
    signal.signal(signal.SIGINT, _sigint)

    w1.start(); time.sleep(0.2); w2.start()

    t0 = time.time()
    try:
        while not stop_all.is_set():
            if args.seconds is not None and (time.time() - t0) >= args.seconds:
                break
            time.sleep(0.1)
    finally:
        w1.stop(); w2.stop()


if __name__ == "__main__":
    main()

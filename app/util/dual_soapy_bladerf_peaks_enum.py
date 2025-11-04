#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, signal
from threading import Thread, Event
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

@dataclass
class DevCfg:
    name: str
    serial: str
    freq_hz: float
    sr_hz: float
    bw_hz: Optional[float]
    gain_db: Optional[float]
    nfft: int
    topk: int
    threshold_db: float
    buf_len: int = 8192
    timeout_us: int = 500000

# --- Utilidades de enumeración ---

def kwargs_to_dict(dkw) -> Dict[str, str]:
    """Convierte SoapySDRKwargs a dict[str,str]."""
    try:
        return {str(k): str(v) for k, v in dkw.items()}
    except Exception:
        # Fallback muy defensivo
        out = {}
        for k in dir(dkw):
            if not k or k.startswith('_'): continue
            try:
                v = dkw[k]  # soporta indexado como mapping
                out[str(k)] = str(v)
            except Exception:
                pass
        return out

def match_serial(d: Dict[str, str]) -> Optional[str]:
    """Intenta extraer el serial con distintas claves."""
    for key in ("serial", "hardwareSerial", "serialNumber", "bladerf_serial"):
        if key in d and d[key]:
            return d[key]
    return None

def enumerate_bladerf_full() -> List[Tuple[Any, Dict[str, str]]]:
    """
    Devuelve lista de tuplas: (kwargs_original, dict_convertido)
    filtrando solo driver=bladerf.
    """
    devs = SoapySDR.Device.enumerate()
    out: List[Tuple[Any, Dict[str, str]]] = []
    for dkw in devs:
        d = kwargs_to_dict(dkw)
        if d.get("driver") == "bladerf":
            out.append((dkw, d))
    return out

# --- Worker ---

class SoapyWorker:
    def __init__(self, cfg: DevCfg, devargs):
        self.cfg = cfg
        self.devargs = devargs  # SoapySDRKwargs ORIGINAL (no dict)
        self.sdr = None
        self.stream = None
        self.stop = Event()
        self.t: Optional[Thread] = None
        self.win = np.hanning(self.cfg.nfft).astype(np.float32)

    def _open(self):
        self.sdr = SoapySDR.Device(self.devargs)
        ch = 0
        self.sdr.setFrequency(SOAPY_SDR_RX, ch, self.cfg.freq_hz)
        self.sdr.setSampleRate(SOAPY_SDR_RX, ch, self.cfg.sr_hz)
        if self.cfg.bw_hz:
            try: self.sdr.setBandwidth(SOAPY_SDR_RX, ch, self.cfg.bw_hz)
            except Exception as e: print(f"[{self.cfg.name}] BW rechazado: {e!r}")
        if self.cfg.gain_db is not None:
            try: self.sdr.setGain(SOAPY_SDR_RX, ch, float(self.cfg.gain_db))
            except Exception as e: print(f"[{self.cfg.name}] Gain rechazado: {e!r}")
        self.stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
        self.sdr.activateStream(self.stream)
        sr = self.sdr.getSampleRate(SOAPY_SDR_RX, ch)
        fc = self.sdr.getFrequency(SOAPY_SDR_RX, ch)
        try: bw = self.sdr.getBandwidth(SOAPY_SDR_RX, ch)
        except: bw = 0.0
        print(f"[{self.cfg.name}] Efectivos -> Freq={fc/1e9:.3f} GHz | SR={sr/1e6:.3f} MS/s | BW={bw/1e6:.3f} MHz")

    def _close(self):
        try:
            if self.stream: self.sdr.deactivateStream(self.stream)
        except: pass
        try:
            if self.stream: self.sdr.closeStream(self.stream)
        except: pass
        self.stream = None
        self.sdr = None

    def _freq_axis(self) -> np.ndarray:
        ch = 0
        sr = self.sdr.getSampleRate(SOAPY_SDR_RX, ch)
        fc = self.sdr.getFrequency(SOAPY_SDR_RX, ch)
        f = np.fft.fftfreq(self.cfg.nfft, d=1.0/sr)
        return np.fft.fftshift(f) + fc

    def _detect_peaks(self, p_db: np.ndarray, freqs: np.ndarray) -> List[Tuple[float,float]]:
        floor = np.median(p_db)
        mask = p_db > (floor + self.cfg.threshold_db)
        if not np.any(mask): return []
        idx = np.where(mask)[0]
        k = min(self.cfg.topk, idx.size)
        top = idx[np.argpartition(p_db[idx], -k)[-k:]]
        top = top[np.argsort(p_db[top])[::-1]]
        return [(float(freqs[i]), float(p_db[i])) for i in top]

    def _loop(self):
        try:
            self._open()
            freqs = self._freq_axis()
            buf = np.empty(self.cfg.buf_len, dtype=np.complex64)
            print(f"[{self.cfg.name}] Capturando: buf={self.cfg.buf_len}, nfft={self.cfg.nfft}")
            while not self.stop.is_set():
                ret = self.sdr.readStream(self.stream, [buf], self.cfg.buf_len, timeoutUs=self.cfg.timeout_us)
                n = ret.ret if hasattr(ret, "ret") else ret[0]
                if n <= 0: continue
                x = buf[:min(n, self.cfg.nfft)]
                if x.size < self.cfg.nfft:
                    x = np.pad(x, (0, self.cfg.nfft - x.size), mode='constant')
                spec = np.fft.fftshift(np.fft.fft(x * self.win, n=self.cfg.nfft))
                p_db = 20.0*np.log10(np.abs(spec)+1e-12)
                peaks = self._detect_peaks(p_db, freqs)
                if peaks:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] [{self.cfg.name}] " + " | ".join(f"{f/1e6:10.3f} MHz {p:6.1f} dB" for f,p in peaks))
        except Exception as e:
            print(f"[{self.cfg.name}] ERROR loop: {e!r}")
        finally:
            self._close()
            print(f"[{self.cfg.name}] Captura detenida.")

    def start(self):
        self.t = Thread(target=self._loop, daemon=True)
        self.t.start()

    def stop(self):
        self.stop.set()
        if self.t: self.t.join(timeout=2.0)

def parse_args():
    ap = argparse.ArgumentParser(description="Dual bladeRF (Soapy) con enumeración segura + picos")
    ap.add_argument("--serial1", required=True)
    ap.add_argument("--serial2", required=True)
    ap.add_argument("--freq1", type=float, default=2.43e9)
    ap.add_argument("--freq2", type=float, default=5.80e9)
    ap.add_argument("--sr1",   type=float, default=5e6)
    ap.add_argument("--sr2",   type=float, default=5e6)
    ap.add_argument("--bw1",   type=float, default=None)
    ap.add_argument("--bw2",   type=float, default=None)
    ap.add_argument("--gain1", type=float, default=30.0)
    ap.add_argument("--gain2", type=float, default=30.0)
    ap.add_argument("--nfft",  type=int, default=4096)
    ap.add_argument("--topk",  type=int, default=3)
    ap.add_argument("--threshold_db", type=float, default=12.0)
    ap.add_argument("--buf_len", type=int, default=8192)
    ap.add_argument("--timeout_us", type=int, default=500000)
    return ap.parse_args()

def main():
    a = parse_args()

    # 1) Enumerar y construir mapa serial -> (kwargs_original, dict_info)
    found = enumerate_bladerf_full()
    if not found:
        raise RuntimeError(
            "Soapy no ve bladeRF. Verifica el plugin:\n"
            "  - que exista /opt/homebrew/lib/SoapySDR/modules0.8/libbladerfSupport.dylib\n"
            "  - SoapySDRUtil --find \"driver=bladerf\" lista dispositivos\n"
            "  - export SOAPY_SDR_PLUGIN_PATH=/opt/homebrew/lib/SoapySDR/modules0.8"
        )

    print("=== Dispositivos enumerados (bladeRF) ===")
    sermap_kwargs: Dict[str, Any] = {}
    for dkw, d in found:
        s = match_serial(d)
        print("->", d)  # debug útil: muestra exactamente las claves que devuelve Soapy
        if s: sermap_kwargs[s] = dkw

    # 2) Validar que están los dos seriales pedidos
    missing = [s for s in (a.serial1, a.serial2) if s not in sermap_kwargs]
    if missing:
        print("\nNo encontré estos seriales vía Soapy:", missing)
        print("Disponibles:", list(sermap_kwargs.keys()))
        raise SystemExit(1)

    cfg1 = DevCfg("2.4GHz", a.serial1, a.freq1, a.sr1, a.bw1, a.gain1, a.nfft, a.topk, a.threshold_db, a.buf_len, a.timeout_us)
    cfg2 = DevCfg("5.8GHz", a.serial2, a.freq2, a.sr2, a.bw2, a.gain2, a.nfft, a.topk, a.threshold_db, a.buf_len, a.timeout_us)

    w1 = SoapyWorker(cfg1, sermap_kwargs[a.serial1])
    w2 = SoapyWorker(cfg2, sermap_kwargs[a.serial2])

    stop_all = Event()
    signal.signal(signal.SIGINT, lambda s,f: stop_all.set())

    w1.start(); time.sleep(0.5); w2.start()
    try:
        while not stop_all.is_set():
            time.sleep(0.2)
    finally:
        w1.stop(); w2.stop()

if __name__ == "__main__":
    main()

# app/util/mat_spectrum.py
from pathlib import Path
import os, time, glob, json
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.io import loadmat
import h5py

UTIL_DIR  = Path(__file__).resolve().parent
APP_DIR   = UTIL_DIR.parent
ROOT_DIR  = APP_DIR.parent

# ===== Preferencias de rutas =====
_env_mat = os.getenv("MAT_DIR")
if _env_mat:
    MAT_DIR = str(Path(_env_mat).expanduser().resolve())
elif (ROOT_DIR / "mat_files").exists():
    MAT_DIR = str((ROOT_DIR / "mat_files").resolve())
else:
    MAT_DIR = str((APP_DIR / "mat_files").resolve())

# ===== Parámetros espectrales =====
FFT_SIZE    = int(os.getenv("FFT_SIZE", "2048"))
HOP_SAMPLES = int(os.getenv("HOP_SAMPLES", str(FFT_SIZE)))  # sin solape por defecto
DEFAULT_FS  = float(os.getenv("DEFAULT_FS", "20000000"))
DEFAULT_FC  = float(os.getenv("DEFAULT_FC", "2412000000"))
WINDOW_NAME = os.getenv("WINDOW", "hann").lower()  # hann|hamming|blackman

# ===== Parámetros para .dat =====
DAT_DTYPE        = os.getenv("DAT_DTYPE", "int16").lower()         # int16|float32|complex64
DAT_LAYOUT       = os.getenv("DAT_LAYOUT", "interleaved").lower()  # interleaved|planar
DAT_ENDIAN       = os.getenv("DAT_ENDIAN", "little").lower()       # little|big
DAT_SCALE_INT16  = os.getenv("DAT_SCALE_INT16", "1")               # 1|0
DAT_DEFAULT_FS   = float(os.getenv("DAT_DEFAULT_FS", str(DEFAULT_FS)))
DAT_DEFAULT_FC   = float(os.getenv("DAT_DEFAULT_FC", str(DEFAULT_FC)))

def _win(n: int):
    if WINDOW_NAME == "hamming":
        return np.hamming(n)
    if WINDOW_NAME == "blackman":
        return np.blackman(n)
    return np.hanning(n)

# ====== Lectura .mat (v7 y v7.3) ======
def _read_mat_any(path: str):
    try:
        return loadmat(path)
    except Exception:
        return h5py.File(path, "r")

def _get_field(obj, *names, default=None):
    keys = list(obj.keys()) if hasattr(obj, "keys") else []
    lower = {k.lower(): k for k in keys}
    for n in names:
        k = lower.get(n.lower())
        if k is not None:
            v = obj[k]
            if hasattr(v, "shape") and hasattr(v, "dtype"):  # h5py dataset
                return np.array(v)
            return v
    return default

# ====== Sidecars/metadatos para .dat ======
def _try_read_json_sidecar(basepath: Path) -> dict:
    """
    Lee <archivo>.json si existe. Estructura sugerida:
    { "fs": 20e6, "fc": 2.412e9, "dtype": "int16|float32|complex64",
      "layout": "interleaved|planar", "endian": "little|big", "scale_int16": true }
    """
    json_path = basepath.with_suffix(basepath.suffix + ".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _try_read_sigmf_meta(basepath: Path) -> dict:
    """
    Lee <archivo>.sigmf-meta si existe (JSON SigMF).
    Campos típicos:
      "global": { "core:sample_rate": <Hz>, "core:datatype": "cf32_le"|"ci16_le"|... }
      "captures": [{ "core:frequency": <Hz>, ... }]
    """
    meta_path = basepath.with_suffix(basepath.suffix + ".sigmf-meta")
    if not meta_path.exists():
        # También probar mismo nombre pero sin sufijo extra (por si ya termina en .data)
        alt = basepath.with_suffix(".sigmf-meta")
        if alt.exists():
            meta_path = alt
        else:
            return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        out = {}
        g = meta.get("global", {})
        caps = meta.get("captures", []) or []
        # fs
        if "core:sample_rate" in g:
            out["fs"] = float(g["core:sample_rate"])
        # fc: a veces está en captures[0].core:frequency
        if caps and isinstance(caps, list):
            fc0 = caps[0].get("core:frequency")
            if fc0 is not None:
                out["fc"] = float(fc0)
        # dtype/endian desde core:datatype
        cdt = g.get("core:datatype", "").lower()
        # Mapeos comunes SigMF
        # cf32_le -> complex64 little endian; ci16_le -> int16 I/Q interleaved little endian
        if cdt:
            if "cf32" in cdt:
                out["dtype"] = "complex64"
            elif "ci16" in cdt:
                out["dtype"] = "int16"
            elif "cf64" in cdt:
                out["dtype"] = "complex128"
            elif "rf32" in cdt or "f32" in cdt:
                out["dtype"] = "float32"
            # endian
            out["endian"] = "little" if cdt.endswith("_le") else ("big" if cdt.endswith("_be") else DAT_ENDIAN)
            # layout: SigMF suele ser interleaved para enteros complejos / reales de I/Q
            out["layout"] = "interleaved"
        return out
    except Exception:
        return {}

def _merge_meta(*ds: dict) -> dict:
    out = {}
    for d in ds:
        for k, v in d.items():
            if v is not None and k not in out:
                out[k] = v
    return out

# ====== Lectura .dat ======
def _np_dtype_from_tokens(dtype_tok: str, endian_tok: str) -> np.dtype:
    endian = "<" if endian_tok == "little" else ">"
    if dtype_tok == "int16":
        return np.dtype(endian + "i2")
    if dtype_tok == "float32":
        return np.dtype(endian + "f4")
    if dtype_tok == "complex64":
        # np.complex64 no codifica endianness en el tipo compuesto, pero lo respetaremos al leer pares de f4
        # Si es complex nativo empaquetado, usaremos np.complex64 (que es dos f4 en el orden de la arquitectura)
        return np.complex64
    if dtype_tok == "complex128":
        return np.complex128
    # fallback
    return np.dtype(endian + "i2")

def _read_dat_raw(path: Path, meta: dict) -> np.ndarray:
    """
    Lee un archivo .dat crudo y construye un vector complejo IQ.
    Meta esperada (con defaults por entorno):
      dtype: int16|float32|complex64|complex128
      layout: interleaved|planar (ignorado si dtype es complejo nativo)
      endian: little|big
      scale_int16: bool (para normalizar a [-1,1))
    """
    dtype_tok   = (meta.get("dtype") or DAT_DTYPE).lower()
    layout_tok  = (meta.get("layout") or DAT_LAYOUT).lower()
    endian_tok  = (meta.get("endian") or DAT_ENDIAN).lower()
    scale_i16   = bool(meta.get("scale_int16", DAT_SCALE_INT16 not in ("0", "false", "False")))

    # Si el archivo está en complejos nativos (complex64/complex128), leemos directo
    if dtype_tok in ("complex64", "complex128"):
        np_dtype = _np_dtype_from_tokens(dtype_tok, endian_tok)
        data = np.fromfile(path, dtype=np_dtype)
        iq = data.astype(np.complex128, copy=False)
        return iq

    # Si es real (int16 o float32), debemos combinar I y Q
    np_dtype = _np_dtype_from_tokens(dtype_tok, endian_tok)
    raw = np.fromfile(path, dtype=np_dtype)

    if raw.size % 2 != 0:
        raise ValueError(f"El .dat '{path.name}' tiene número impar de muestras reales; no puede dividirse en I/Q.")

    if layout_tok == "interleaved":
        I = raw[0::2]
        Q = raw[1::2]
    elif layout_tok == "planar":
        half = raw.size // 2
        I = raw[:half]
        Q = raw[half:]
    else:
        raise ValueError(f"layout desconocido para .dat: {layout_tok}")

    I = I.astype(np.float64)
    Q = Q.astype(np.float64)

    if dtype_tok == "int16" and scale_i16:
        # Escalar a rango ~[-1,1)
        I /= 32768.0
        Q /= 32768.0

    iq = I + 1j * Q
    return iq.astype(np.complex128, copy=False)

def load_iq_fs_fc_from_dat(path: str) -> Tuple[np.ndarray, float, float]:
    """
    Carga IQ desde .dat con heurísticas y sidecars (.sigmf-meta / .json).
    Retorna (iq, fs, fc).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    # 1) Metadatos desde sidecars
    meta_sigmf = _try_read_sigmf_meta(p)
    meta_json  = _try_read_json_sidecar(p)
    meta = _merge_meta(meta_sigmf, meta_json)

    # 2) Leer IQ
    iq = _read_dat_raw(p, meta)

    # 3) fs/fc desde meta o defaults
    fs = float(meta.get("fs", DAT_DEFAULT_FS))
    fc = float(meta.get("fc", DAT_DEFAULT_FC))

    return iq, fs, fc

# ====== Lectura unificada IQ (según extensión) ======
def load_iq_fs_fc(path: str) -> Tuple[np.ndarray, float, float]:
    """
    Lee IQ y (fs, fc) desde .mat o .dat. Para .mat usa múltiples rutas/formatos.
    Para .dat intenta leer sidecars (SigMF/.json) y aplica variables de entorno como fallback.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    ext = Path(path).suffix.lower()
    if ext in  (".dat", ".bin"):
        return load_iq_fs_fc_from_dat(path)

    # ---- Caso .mat (original) ----
    m = _read_mat_any(path)

    iq = None

    # 1) Rutas “clásicas” (MAT v5/v7 o HDF5 con datasets simples)
    iq = _get_field(m, "iq", "IQ", "samples", "data", "x", "s")
    if iq is None:
        I = _get_field(m, "i", "I", "real")
        Q = _get_field(m, "q", "Q", "imag")
        if I is not None and Q is not None:
            iq = I.astype(np.float64) + 1j * Q.astype(np.float64)
    if iq is None:
        inter = _get_field(m, "interleaved", "raw", "y")
        if inter is not None and hasattr(inter, "ndim") and inter.ndim == 1 and inter.size % 2 == 0:
            a = inter.astype(np.float64)
            iq = a[0::2] + 1j * a[1::2]

    # 2) HDF5 compuesto tipo UHD: /uhd_samps con campos 'real' y 'imag'
    if iq is None and isinstance(m, h5py.File):
        if "uhd_samps" in m:
            ds = m["uhd_samps"]
            arr = np.array(ds)
            if hasattr(arr, "dtype") and arr.dtype.fields:
                fields = set(arr.dtype.fields.keys())
                cand_real = [n for n in fields if n.lower() in ("real", "re", "r")]
                cand_imag = [n for n in fields if n.lower() in ("imag", "im", "i")]
                if cand_real and cand_imag:
                    r = arr[cand_real[0]].astype(np.float64)
                    im = arr[cand_imag[0]].astype(np.float64)
                    iq = r + 1j * im
        if iq is None:
            def _scan(hgroup):
                for k, v in hgroup.items():
                    if isinstance(v, h5py.Dataset):
                        a = np.array(v)
                        if hasattr(a, "dtype") and a.dtype.fields:
                            flds = set(a.dtype.fields.keys())
                            if any(fn.lower() in ("real", "re", "r") for fn in flds) and any(fn.lower() in ("imag","im","i") for fn in flds):
                                rr_name = next(fn for fn in flds if fn.lower() in ("real", "re", "r"))
                                ii_name = next(fn for fn in flds if fn.lower() in ("imag","im","i"))
                                rr = a[rr_name].astype(np.float64)
                                ii = a[ii_name].astype(np.float64)
                                return rr + 1j * ii
                    elif isinstance(v, h5py.Group):
                        got = _scan(v)
                        if got is not None:
                            return got
                return None
            iq = _scan(m)

    if iq is None:
        raise ValueError("No se encontró IQ en el archivo (probé .mat: iq/samples/data/x, i+q, interleaved y HDF5 compuesto; .dat: use sidecars o variables de entorno).")

    # Acomodar forma a vector 1D complejo
    iq = np.squeeze(iq)
    if not np.iscomplexobj(iq) and getattr(iq, "ndim", 1) == 2 and iq.shape[0] == 2:
        iq = iq[0, :] + 1j * iq[1, :]
    elif not np.iscomplexobj(iq) and getattr(iq, "ndim", 1) == 2 and iq.shape[1] == 2:
        iq = iq[:, 0] + 1j * iq[:, 1]
    iq = iq.reshape(-1).astype(np.complex128, copy=False)

    # fs / fc: en muchos .mat no vienen; usar defaults si no aparecen
    fs = _get_field(m, "fs", "Fs", "sample_rate", "samp_rate", "sampleRate")
    fc = _get_field(m, "fc", "Fc", "center_freq", "centerFreq", "freq_center")
    fs = float(np.squeeze(fs)) if fs is not None else DEFAULT_FS
    fc = float(np.squeeze(fc)) if fc is not None else DEFAULT_FC

    return iq, fs, fc

# ====== FFT y PSD ======
def iq_to_psd_blocks(iq: np.ndarray, fs: float, fc: float,
                     fft_size: int = FFT_SIZE, hop: int = HOP_SAMPLES) -> List[dict]:
    win = _win(fft_size)
    win_scale = (win**2).sum()
    blocks: List[dict] = []
    if len(iq) < fft_size:
        return blocks
    for start in range(0, len(iq) - fft_size + 1, hop):
        seg = iq[start:start+fft_size]
        xw = seg * win
        X = np.fft.fftshift(np.fft.fft(xw, n=fft_size))
        psd = (np.abs(X)**2) / (fft_size * win_scale)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
        bin_hz = fs / fft_size
        start_hz = fc - fs/2.0  # por fftshift
        blocks.append({
            "start_hz": float(start_hz),
            "bin_hz": float(bin_hz),
            "bins": psd_db.astype(np.float32).tolist(),
            "capture_time_sec": time.time()
        })
    return blocks

# ====== Índice de archivos y cache ======
# id -> ruta (id = nombre de archivo sin extensión)
DRONES: Dict[str, str] = {}
# id -> lista de bloques PSD cacheados
PSD_CACHE: Dict[str, List[dict]] = {}
DEFAULT_DRONE_ID: Optional[str] = None

def scan_dir() -> Dict[str, str]:
    """
    Escanea MAT_DIR en busca de .mat y .dat.
    Si existen ambos con el mismo nombre base, se prioriza .mat.
    """
    paths_mat = glob.glob(os.path.join(MAT_DIR, "*.mat"))
    paths_dat = glob.glob(os.path.join(MAT_DIR, "*.dat"))
    paths_bin = glob.glob(os.path.join(MAT_DIR, "*.bin"))
    out: Dict[str, str] = {}
    for p in paths_dat + paths_bin:
        bid = os.path.splitext(os.path.basename(p))[0]
        out[bid] = os.path.abspath(p)
    for p in paths_mat:
        bid = os.path.splitext(os.path.basename(p))[0]
        # Prioridad a .mat frente a .dat si comparten ID
        out[bid] = os.path.abspath(p)
    return out

def refresh_dir() -> None:
    """Reescanea carpeta y mantiene DEFAULT si existe."""
    global DRONES, DEFAULT_DRONE_ID
    DRONES = scan_dir()
    if DEFAULT_DRONE_ID not in DRONES:
        DEFAULT_DRONE_ID = next(iter(DRONES.keys()), None)
    # limpia cache de ids que ya no están
    for k in list(PSD_CACHE.keys()):
        if k not in DRONES:
            PSD_CACHE.pop(k, None)

def get_blocks(drone_id: str) -> List[dict]:
    """Obtiene (o genera y cachea) bloques PSD sin cargar todo el IQ a memoria."""
    if drone_id not in DRONES:
        raise KeyError(f"Dron '{drone_id}' no encontrado en {MAT_DIR}")
    if drone_id in PSD_CACHE:
        return PSD_CACHE[drone_id]

    path = DRONES[drone_id]
    # Usa el lector perezoso para .dat/.bin o HDF5/.mat v7.3; v5 cae al lector np
    reader, fs, fc = open_iq_reader(path)

    # Genera algunos bloques y cachea (p.ej., 3–5 segs)
    blocks: List[dict] = []
    max_blocks = max(1, int(os.getenv("PSD_CACHE_BLOCKS", "32")))  # configurable
    ds = int(os.getenv("PSD_BIN_DOWNSAMPLE", "1"))  # factor de suavizado horizontal opcional

    gen = psd_stream(reader, fs, fc, FFT_SIZE, HOP_SAMPLES, ds=ds)
    for i, b in enumerate(gen):
        blocks.append(b)
        if i + 1 >= max_blocks:
            break

    PSD_CACHE[drone_id] = blocks
    return blocks


class _DatIQReader:
    def __init__(self, path: Path, meta: dict, default_fs: float, default_fc: float):
        self.path = path
        dtype_tok = (meta.get("dtype") or DAT_DTYPE).lower()
        layout_tok = (meta.get("layout") or DAT_LAYOUT).lower()
        endian_tok = (meta.get("endian") or DAT_ENDIAN).lower()
        self.scale_i16 = bool(meta.get("scale_int16", DAT_SCALE_INT16 not in ("0", "false", "False")))
        self.layout = layout_tok
        self.dtype_tok = dtype_tok

        # memmap del archivo crudo
        np_dtype = _np_dtype_from_tokens(dtype_tok, endian_tok)
        self.mm = np.memmap(path, dtype=np_dtype, mode="r")

        if dtype_tok in ("complex64", "complex128"):
            # Complejo nativo: cada elemento ya es I+jQ
            self.N = int(self.mm.shape[0])
            self._mode = "complex"
        else:
            # Real: I/Q interleaved o planar
            if layout_tok == "interleaved":
                if self.mm.size % 2 != 0:
                    raise ValueError(f"{path.name}: tamaño impar, no se puede dividir en I/Q")
                self.N = self.mm.size // 2
                self._mode = "i16_interleaved" if dtype_tok == "int16" else "f32_interleaved"
            elif layout_tok == "planar":
                if self.mm.size % 2 != 0:
                    raise ValueError(f"{path.name}: tamaño impar, no se puede dividir en I/Q")
                self.N = self.mm.size // 2
                self._mode = "i16_planar" if dtype_tok == "int16" else "f32_planar"
                self._half = self.N
            else:
                raise ValueError(f"layout desconocido: {layout_tok}")

        self.fs = float(meta.get("fs", default_fs))
        self.fc = float(meta.get("fc", default_fc))

    def n_samples(self) -> int:
        return self.N

    def segment(self, start: int, length: int) -> np.ndarray:
        """Devuelve un segmento complejo de longitud 'length' SIN leer todo el archivo."""
        if start < 0 or start + length > self.N:
            # Devuelve ceros si te sales; o lanza
            raise IndexError("segment fuera de rango")
        if self._mode == "complex":
            seg = self.mm[start:start+length]
            return seg.astype(np.complex128, copy=False)

        if "interleaved" in self._mode:
            base = 2 * start
            I = self.mm[base: base + 2*length: 2]
            Q = self.mm[base+1: base + 2*length: 2]
        else:
            # planar
            I = self.mm[start: start + length]
            Q = self.mm[self._half + start: self._half + start + length]

        # Convertir por segmento (pequeño), no todo el archivo
        I = I.astype(np.float64, copy=False)
        Q = Q.astype(np.float64, copy=False)
        if self.dtype_tok == "int16" and self.scale_i16:
            I /= 32768.0
            Q /= 32768.0
        return (I + 1j * Q).astype(np.complex128, copy=False)


class _H5IQReader:
    def __init__(self, h5: h5py.File, dataset, kind: str, rr=None, ii=None, axis_mode=None, default_fs=None, default_fc=None):
        """
        kind:
          - "compound": dataset con campos reales/imag (usar rr, ii como nombres de campo)
          - "matrix": dataset 2xN o Nx2 (usar axis_mode="rows" si 2xN, "cols" si Nx2)
        """
        self.h5 = h5
        self.ds = dataset
        self.kind = kind
        self.rr = rr
        self.ii = ii
        self.axis_mode = axis_mode
        self._N = None
        # fs/fc desde atributos si existieran
        self.fs = default_fs
        self.fc = default_fc

    def n_samples(self) -> int:
        if self._N is not None:
            return self._N
        if self.kind == "compound":
            self._N = self.ds.shape[0]
        else:
            if self.axis_mode == "rows":  # 2 x N
                self._N = self.ds.shape[1]
            else:  # Nx2
                self._N = self.ds.shape[0]
        return self._N

    def segment(self, start: int, length: int) -> np.ndarray:
        if start < 0 or start + length > self.n_samples():
            raise IndexError("segment fuera de rango (HDF5)")
        if self.kind == "compound":
            chunk = self.ds[start:start+length]
            I = chunk[self.rr].astype(np.float64, copy=False)
            Q = chunk[self.ii].astype(np.float64, copy=False)
        else:
            if self.axis_mode == "rows":  # 2 x N
                I = self.ds[0, start:start+length].astype(np.float64, copy=False)
                Q = self.ds[1, start:start+length].astype(np.float64, copy=False)
            else:  # Nx2
                I = self.ds[start:start+length, 0].astype(np.float64, copy=False)
                Q = self.ds[start:start+length, 1].astype(np.float64, copy=False)
        return (I + 1j * Q).astype(np.complex128, copy=False)


def open_iq_reader(path: str) -> Tuple[object, float, float]:
    """
    Devuelve (reader, fs, fc) donde reader tiene:
      - n_samples() -> int
      - segment(start, length) -> np.ndarray complejo
    Sin cargar todo el IQ a memoria.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    ext = p.suffix.lower()
    if ext in (".dat", ".bin"):
        meta_sigmf = _try_read_sigmf_meta(p)
        meta_json  = _try_read_json_sidecar(p)
        meta = _merge_meta(meta_sigmf, meta_json)
        r = _DatIQReader(p, meta, DEFAULT_FS, DEFAULT_FC)
        return r, r.fs, r.fc

    # Intentar HDF5 (MAT v7.3 u HDF5 genérico), sin convertir a np.array
    try:
        h5 = h5py.File(p, "r")
        # 1) compuesto
        if "uhd_samps" in h5:
            ds = h5["uhd_samps"]
            if hasattr(ds, "dtype") and ds.dtype.fields:
                flds = set(ds.dtype.fields.keys())
                rr = next((fn for fn in flds if fn.lower() in ("real","re","r")), None)
                ii = next((fn for fn in flds if fn.lower() in ("imag","im","i")), None)
                if rr and ii:
                    r = _H5IQReader(h5, ds, kind="compound", rr=rr, ii=ii,
                                    default_fs=DEFAULT_FS, default_fc=DEFAULT_FC)
                    return r, r.fs, r.fc
        # 2) escanear datasets en busca de 2xN o Nx2
        def _scan_ds(hgroup):
            for k, v in hgroup.items():
                if isinstance(v, h5py.Dataset):
                    shp = v.shape
                    if len(shp) == 2 and (shp[0] == 2 or shp[1] == 2):
                        axis_mode = "rows" if shp[0] == 2 else "cols"
                        return v, axis_mode
                elif isinstance(v, h5py.Group):
                    got = _scan_ds(v)
                    if got:
                        return got
            return None
        res = _scan_ds(h5)
        if res:
            ds, axis_mode = res
            r = _H5IQReader(h5, ds, kind="matrix", axis_mode=axis_mode,
                            default_fs=DEFAULT_FS, default_fc=DEFAULT_FC)
            return r, r.fs, r.fc
    except Exception:
        pass  # caeremos a .mat v5

    # Caso .mat v5 (loadmat): no hay forma perezosa; se cargará todo.
    # Úsalo SOLO si es pequeño; para archivos grandes convierte a v7.3 o .dat.
    iq, fs, fc = load_iq_fs_fc(str(p))  # esto sí carga todo
    class _NPReader:
        def __init__(self, arr):
            self.arr = arr
        def n_samples(self): return self.arr.shape[0]
        def segment(self, start, length): return self.arr[start:start+length]
    return _NPReader(iq), fs, fc


def psd_stream(reader, fs: float, fc: float, fft_size: int, hop: int, ds: int = 1):
    """
    Generador que emite bloques PSD leyendo solo ventanas del reader.
    """
    if reader.n_samples() < fft_size:
        return
    win = _win(fft_size)
    win_scale = float((win**2).sum())
    n_blocks = 1 + (reader.n_samples() - fft_size) // hop
    for i in range(n_blocks):
        start = i * hop
        seg = reader.segment(start, fft_size)
        X = np.fft.fftshift(np.fft.fft(seg * win, n=fft_size))
        psd = (np.abs(X)**2) / (fft_size * win_scale)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20)).astype(np.float32)
        bin_hz = fs / fft_size
        if ds > 1:
            newN = (len(psd_db) // ds) * ds
            if newN > 0:
                psd_db = psd_db[:newN].reshape(-1, ds).mean(axis=1)
                bin_hz = bin_hz * ds
        yield {
            "start_hz": float(fc - fs/2.0),
            "bin_hz": float(bin_hz),
            "bins": psd_db.tolist(),
            "capture_time_sec": time.time(),
        }

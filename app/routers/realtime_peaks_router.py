# app/routers/realtime_peaks_router.py
from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
import asyncio, json, time
from collections import deque
from typing import Deque, Dict, Any, List, Optional

router = APIRouter(prefix="/realTimePeaks", tags=["realTimePeaks"])

# ============================
#   HUB (in-memory pub/sub)
# ============================
class Watcher:
    def __init__(self, ws: WebSocket, criteria: Dict[str, Any]):
        self.ws = ws
        self.criteria = criteria  # ej: {"serial": "...", "name": "...", "fmin": .., "fmax": .., "min_snr": ..}

class PeaksHub:
    def __init__(self, max_cache: int = 200):
        self._watchers: List[Watcher] = []
        self._lock = asyncio.Lock()
        self._cache: Deque[Dict[str, Any]] = deque(maxlen=max_cache)

    def _match(self, msg: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Filtrado simple por query del watcher."""
        if not criteria:
            return True

        # msg["radio"] -> {"name": "...", "serial": "..."}
        radio = (msg.get("radio") or {})
        name = (radio.get("name") or "").lower()
        serial = (radio.get("serial") or "").lower()

        # parse básicos
        fmin = criteria.get("fmin_hz")
        fmax = criteria.get("fmax_hz")
        want_name = (criteria.get("name") or "").lower() or None
        want_serial = (criteria.get("serial") or "").lower() or None
        min_snr = criteria.get("min_snr_db")

        # si pide rango de frecuencias, basta que algún pico caiga allí
        if fmin is not None or fmax is not None:
            peaks = msg.get("peaks") or []
            ok = False
            for p in peaks:
                fhz = float(p.get("f_hz", 0.0))
                if fmin is not None and fhz < fmin:
                    continue
                if fmax is not None and fhz > fmax:
                    continue
                ok = True
                break
            if not ok:
                return False

        if want_name and (want_name not in name):
            return False

        if want_serial and (want_serial not in serial):
            return False

        if min_snr is not None:
            nf = float(msg.get("noise_floor_db", 0.0))
            # si todos los picos quedan por debajo del min_snr -> filtra
            peaks = msg.get("peaks") or []
            if not any((float(p.get("p_db", -1e9)) - nf) >= float(min_snr) for p in peaks):
                return False

        return True

    async def register(self, w: Watcher):
        async with self._lock:
            self._watchers.append(w)

    async def unregister(self, w: Watcher):
        async with self._lock:
            if w in self._watchers:
                self._watchers.remove(w)

    def cache_last(self, msg: Dict[str, Any]):
        self._cache.append(msg)

    def cached(self, limit: int = 50, criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Devuelve hasta 'limit' últimos mensajes que cumplen filtro (del más reciente hacia atrás)
        out: List[Dict[str, Any]] = []
        for m in reversed(self._cache):
            if not criteria or self._match(m, criteria):
                out.append(m)
            if len(out) >= limit:
                break
        return list(reversed(out))

    async def broadcast(self, msg: Dict[str, Any]):
        # Enviar sólo a watchers cuyos criterios matcheen
        dead: List[Watcher] = []
        payload = json.dumps(msg, ensure_ascii=False)
        async with self._lock:
            for w in self._watchers:
                try:
                    if self._match(msg, w.criteria):
                        await w.ws.send_text(payload)
                except Exception:
                    dead.append(w)
            for w in dead:
                try:
                    self._watchers.remove(w)
                except Exception:
                    pass

HUB = PeaksHub(max_cache=400)

# ============================
#  WS DE INGESTA (PRODUCTOR)
# ============================
@router.websocket("/ws/ingest-peaks")
async def ws_ingest_peaks(ws: WebSocket):
    """
    El script dual_soapy_bladerf_peaks_* se conecta aquí (cliente WS) y envía JSONs (uno por bloque).
    Cada mensaje se cachea y se retransmite a /ws/peaks (viewers).
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                # sello del servidor
                data["_server_ts"] = time.time()
                # validación mínima
                if not isinstance(data.get("peaks", []), list):
                    data["peaks"] = []

                # guarda en cache y difunde
                HUB.cache_last(data)
                await HUB.broadcast(data)

            except json.JSONDecodeError:
                # opcional: devolver error al productor
                await ws.send_text(json.dumps({"error": "invalid_json"}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        return

# ============================
#   WS DE VISTA (SUSCRIPTORES)
# ============================
@router.websocket("/ws/peaks")
async def ws_peaks(ws: WebSocket,
                   serial: Optional[str] = Query(default=None, description="Filtro por serial (contiene)"),
                   name: Optional[str] = Query(default=None, description="Filtro por nombre (contiene)"),
                   fmin_hz: Optional[float] = Query(default=None),
                   fmax_hz: Optional[float] = Query(default=None),
                   min_snr_db: Optional[float] = Query(default=None),
                   replay: int = Query(default=20, description="Cuántos últimos mensajes enviar al conectar")):
    """
    Los clientes (tu dashboard) se conectan aquí para recibir picos en tiempo real.
    Puedes filtrar por serial, nombre, rango de frecuencias, o SNR mínimo.
    """
    await ws.accept()
    watcher = Watcher(ws, {
        "serial": serial,
        "name": name,
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "min_snr_db": min_snr_db,
    })
    await HUB.register(watcher)
    try:
        # Reenvía backlog inicial (si aplica)
        if replay and replay > 0:
            cached = HUB.cached(limit=int(replay), criteria=watcher.criteria)
            for m in cached:
                await ws.send_text(json.dumps(m, ensure_ascii=False))

        # En WS de solo-push, el servidor no espera nada del cliente;
        # mantenemos la conexión viva leyendo pings.
        while True:
            # Espera no bloqueante para que FastAPI detecte cierres limpios por parte del cliente
            _ = await ws.receive_text()
            # Si quieres eco, puedes re-enviar aquí
    except WebSocketDisconnect:
        await HUB.unregister(watcher)
        return
    except Exception:
        await HUB.unregister(watcher)
        return

# ============================
#   ENDPOINTS HTTP AUXILIARES
# ============================
@router.get("/peaks/last")
def get_last_peaks(limit: int = 20,
                   serial: Optional[str] = None,
                   name: Optional[str] = None,
                   fmin_hz: Optional[float] = None,
                   fmax_hz: Optional[float] = None,
                   min_snr_db: Optional[float] = None):
    """
    Devuelve los últimos 'limit' mensajes cacheados (útil para debug o polling).
    """
    criteria = {
        "serial": serial,
        "name": name,
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "min_snr_db": min_snr_db,
    }
    return {
        "count": limit,
        "items": HUB.cached(limit=limit, criteria=criteria)
    }

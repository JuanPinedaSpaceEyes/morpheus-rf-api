# app/routers/spectrum_router.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi import Depends
import asyncio, json, os

from app.util import mat_spectrum as ms

router = APIRouter(prefix="/spectrum", tags=["spectrum"])

def _ensure_scanned():
    if not ms.DRONES:
        os.makedirs(ms.MAT_DIR, exist_ok=True)
        ms.refresh_dir()

@router.websocket("/ws/psd")
async def ws_psd(ws: WebSocket):
    await ws.accept()
    try:
        _ensure_scanned()
        qp = ws.query_params
        drone_id = qp.get("drone") or ms.DEFAULT_DRONE_ID
        fps = int(qp.get("fps") or os.getenv("STREAM_FPS", "15"))
        if not drone_id or drone_id not in ms.DRONES:
            await ws.send_text(json.dumps({"error": "Dron no disponible"}))
            await ws.close()
            return

        blocks = ms.get_blocks(drone_id)
        if not blocks:
            await ws.send_text(json.dumps({"error": "El .mat no produjo bloques PSD"}))
            await ws.close()
            return

        delay = 1.0 / max(1, fps)
        i = 0
        while True:
            await ws.send_text(json.dumps(blocks[i]))
            i = (i + 1) % len(blocks)
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
            await ws.close()
        except Exception:
            pass

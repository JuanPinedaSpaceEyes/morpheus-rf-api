from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import asyncio
from app.util.rf_capture import probe_power_state

router = APIRouter(prefix="/bladerf", tags=["bladerf"])

@router.websocket("/ws/power")
async def ws_power(ws: WebSocket, interval_ms: int = Query(1000, ge=200, le=10000)):
    await ws.accept()
    try:
        last = None
        st = probe_power_state
        await ws.send_json({"type" : "power", "data": st})
        last = st

        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            cur = probe_power_state
            if not last or cur.get("powered") != last.get("powered") or cur.get("message") != last.get("message") or cur.get("method") != last.get("method"):
                await ws.send_json({"type": "power", "data": cur})
                last = cur
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass

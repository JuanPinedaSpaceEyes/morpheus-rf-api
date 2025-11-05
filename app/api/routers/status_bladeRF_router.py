from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
import asyncio, json, os
from app.util.rf_capture import CaptureService

service = CaptureService()
service.start()


router = APIRouter(prefix="/bladerf", tags=["bladerf"])

@router.websocket("/ws/power")
async def ws_power(ws: WebSocket, interval_ms: int = Query(1000, ge=200, le=10000)):
    await ws.accept()
    try:
        last = None
        st = service.get_power_state()
        await ws.send_json({"type" : "power", "data": st})
        last = st

        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            cur = service.get_power_state()
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

@router.websocket("/ws/psd")
async def ws_psd(ws: WebSocket, interval_ms: int = Query(150, ge=50, le=2000)):
    await ws.accept()
    try:
        last_sent_block = -1
        while True:
            await asyncio.sleep(interval_ms / 1000.0)
            last = service.get_last_block()
            if not last or "psd" not in last:
                continue
            if last["block_id"] == last_sent_block:
                continue
            payload = dict(last["psd"])
            payload["capture_time_sec"] = last["capture_time_sec"]
            await ws.send_json(payload)
            last_sent_block = last["block_id"]
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass


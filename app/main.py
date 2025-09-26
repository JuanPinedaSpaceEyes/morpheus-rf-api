# app/main.py
from email.policy import default

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
from fastapi import Query

from .models import Status, PeaksBlock, Peak, PowerStatus
from .rf_capture import CaptureService

app = FastAPI(title="Morpheus RF API", version="1.1")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servicio de captura
service = CaptureService()
service.start()

@app.on_event("shutdown")
def shutdown_event():
    service.stop()

# ---- API ----
@app.get("/status", response_model=Status)
def status():
    return service.get_status()

@app.get("/peaks", response_model=PeaksBlock)
def peaks():
    last = service.get_last_block()
    if not last:
        return JSONResponse(status_code=503, content={"detail": "Aún no hay datos"})
    return PeaksBlock(
        block_id=last["block_id"],
        capture_time_sec=last["capture_time_sec"],
        noise_floor_db=last["noise_floor_db"],
        max_power_db=last["max_power_db"],
        peaks=[Peak(frequency_hz=f, power_db=p) for (f, p) in last["peaks"]],
    )

@app.get("/power", response_model=PowerStatus)
def power():
    st = service.get_power_state()
    return PowerStatus(**st)

@app.websocket("/ws/peaks")
async def ws_peaks(ws: WebSocket):
    await ws.accept()
    try:
        last_sent_id = -1
        while True:
            await asyncio.sleep(0.2)  # ~5 Hz
            last = service.get_last_block()
            if not last or last["block_id"] == last_sent_id:
                continue
            payload = dict(
                block_id=last["block_id"],
                capture_time_sec=last["capture_time_sec"],
                noise_floor_db=last["noise_floor_db"],
                max_power_db=last["max_power_db"],
                peaks=[{"frequency_hz": f, "power_db": p} for (f, p) in last["peaks"]],
            )
            await ws.send_json(payload)
            last_sent_id = last["block_id"]
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/ws/power")
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


# ---- Estáticos en /ui ----
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui/")

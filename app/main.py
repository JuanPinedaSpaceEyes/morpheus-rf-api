# app/main.py

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.pipeline_router import router as pipeline_router
from app.api.routers.spectrum_router import router as spectrum_router
from app.api.routers.realtime_peaks_router import router as realtime_peaks_router
from app.api.routers.status_bladeRF_router import router as status_bladerf

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

app = FastAPI(title="Morpheus RF API", version="1.2")
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
app.include_router(spectrum_router)
app.include_router(realtime_peaks_router)
app.include_router(status_bladerf)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#
# # Servicio de captura existente
# service = CaptureService()
# service.start()
#
# # Servicio de detección de drones (hilo interno controlado por endpoints)
# drone_detector = DroneDetectorBladeRF()
#
# @app.on_event("shutdown")
# def shutdown_event():
#     service.stop()
#     drone_detector.stop()

# @app.websocket("/ws/power")
# async def ws_power(ws: WebSocket, interval_ms: int = Query(1000, ge=200, le=10000)):
#     await ws.accept()
#     try:
#         last = None
#         st = service.get_power_state()
#         await ws.send_json({"type" : "power", "data": st})
#         last = st
#
#         while True:
#             await asyncio.sleep(interval_ms / 1000.0)
#             cur = service.get_power_state()
#             if not last or cur.get("powered") != last.get("powered") or cur.get("message") != last.get("message") or cur.get("method") != last.get("method"):
#                 await ws.send_json({"type": "power", "data": cur})
#                 last = cur
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         try:
#             await ws.send_json({"type": "error", "error": str(e)})
#         except Exception:
#             pass
#
# @app.websocket("/ws/psd")
# async def ws_psd(ws: WebSocket, interval_ms: int = Query(150, ge=50, le=2000)):
#     await ws.accept()
#     try:
#         last_sent_block = -1
#         while True:
#             await asyncio.sleep(interval_ms / 1000.0)
#             last = service.get_last_block()
#             if not last or "psd" not in last:
#                 continue
#             if last["block_id"] == last_sent_block:
#                 continue
#             payload = dict(last["psd"])
#             payload["capture_time_sec"] = last["capture_time_sec"]
#             await ws.send_json(payload)
#             last_sent_block = last["block_id"]
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         try:
#             await ws.send_json({"type": "error", "error": str(e)})
#         except Exception:
#             pass
#
# # ---- Estáticos en /ui ----
# STATIC_DIR = Path(__file__).resolve().parent / "static"
# app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
#
# @app.get("/")
# def root_redirect():
#     return RedirectResponse(url="/ui/")
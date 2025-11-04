# app/main.py

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.inference_router import router as inference_router
from app.routers.pipeline_router import router as pipeline_router
from app.routers.spectrum_router import router as spectrum_router
from app.routers.realtime_peaks_router import router as realtime_peaks_router

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

app = FastAPI(title="Morpheus RF API", version="1.2")
app.include_router(inference_router, prefix="/pipelineInference", tags=["pipelineInference"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
app.include_router(spectrum_router)
app.include_router(realtime_peaks_router)

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

# # ---- API existente ----
# @app.get("/status", response_model=Status)
# def status():
#     return service.get_status()
#
# @app.get("/peaks", response_model=PeaksBlock)
# def peaks():
#     last = service.get_last_block()
#     if not last:
#         return JSONResponse(status_code=503, content={"detail": "Aún no hay datos"})
#     return PeaksBlock(
#         block_id=last["block_id"],
#         capture_time_sec=last["capture_time_sec"],
#         noise_floor_db=last["noise_floor_db"],
#         max_power_db=last["max_power_db"],
#         peaks=[Peak(frequency_hz=f, power_db=p) for (f, p) in last["peaks"]],
#     )
#
# @app.get("/power", response_model=PowerStatus)
# def power():
#     st = service.get_power_state()
#     return PowerStatus(**st)
#
# @app.get("/psd")
# def psd():
#     last = service.get_last_block()
#     if not last or "psd" not in last:
#         return JSONResponse(status_code=503, content={"detail": "Aún no hay PSD"})
#     return last["psd"]
#
#
# @app.websocket("/ws/peaks")
# async def ws_peaks(ws: WebSocket):
#     await ws.accept()
#     try:
#         last_sent_id = -1
#         while True:
#             await asyncio.sleep(0.2)  # ~5 Hz
#             last = service.get_last_block()
#             if not last or last["block_id"] == last_sent_id:
#                 continue
#             payload = dict(
#                 block_id=last["block_id"],
#                 capture_time_sec=last["capture_time_sec"],
#                 noise_floor_db=last["noise_floor_db"],
#                 max_power_db=last["max_power_db"],
#                 peaks=[{"frequency_hz": f, "power_db": p} for (f, p) in last["peaks"]],
#             )
#             await ws.send_json(payload)
#             last_sent_id = last["block_id"]
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         try:
#             await ws.send_json({"error": str(e)})
#         except Exception:
#             pass
#
#
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
#
# class BandConfig(BaseModel):
#     center_hz: float
#     sample_rate: float
#
#
# @app.post("/set_band")
# def set_band(cfg: BandConfig):
#     ok = service.set_band(cfg.center_hz, cfg.sample_rate)
#     if not ok:
#         return JSONResponse(status_code=500, content={"detail": "Error al configurar BladeRF"})
#     return {
#         "status": "ok",
#         "center_hz": cfg.center_hz,
#         "sample_rate": cfg.sample_rate,
#     }
#
# # ---- NUEVOS ENDPOINTS: Drone Detector (basado en bladeRF) ----
#
# class DroneSummary(BaseModel):
#     running: bool
#     mode: str
#     runtime_seconds: int
#     total_devices: int
#     drones_count: int
#     others_count: int
#     drones: list[dict]
#
# @app.get("/drone/start", tags=["drone"])
# def start_drone_detector(simulate: bool = Query(True, description="true = simulación (default), false = modo real")):
#     """
#     Arranca el detector en un hilo:
#       - simulate=true: usa flujo simulado (ideal para probar UI/flujo).
#       - simulate=false: punto de conexión para captura real (cuando conectes tu pipeline bladeRF).
#     """
#     return drone_detector.start(simulate=simulate)
#
# @app.get("/drone/stop", tags=["drone"])
# def stop_drone_detector():
#     """Detiene el hilo del detector."""
#     return drone_detector.stop()
#
# @app.get("/drone/summary", response_model=DroneSummary, tags=["drone"])
# def drone_summary():
#     """
#     Resumen compacto del detector: estado, tiempo, conteos y lista parcial de drones.
#     """
#     return drone_detector.summary()
#
#
#
#
#
# # ---- Estáticos en /ui ----
# STATIC_DIR = Path(__file__).resolve().parent / "static"
# app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
#
# @app.get("/")
# def root_redirect():
#     return RedirectResponse(url="/ui/")
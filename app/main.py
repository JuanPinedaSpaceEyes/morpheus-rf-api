# app/main.py

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.pipeline_router import router as pipeline_router
from app.api.routers.spectrum_router import router as spectrum_router

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

app = FastAPI(title="Morpheus RF API", version="1.2")
app.include_router(pipeline_router)
app.include_router(spectrum_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
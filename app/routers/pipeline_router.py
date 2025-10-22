# app/pipeline_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pathlib import Path
import subprocess, os, time, io, threading, signal
from fastapi.responses import PlainTextResponse


router = APIRouter()

# ---------- Config ----------
PIPELINE_PATH = Path(__file__).resolve().parent.parent / "pipeline.py"
LOG_DIR = Path(os.getenv("PIPELINE_LOG_DIR", "/tmp/morpheus_pipeline"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
CWD = Path(os.getenv("PIPELINE_CWD", str(PIPELINE_PATH.parent)))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")  # opcional: fuerza intérprete

# ---------- Estado ----------
class _State:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.started_at: Optional[float] = None
        self.log_file: Optional[Path] = None
        self.lock = threading.Lock()

state = _State()

# ---------- Helpers ----------
def _start_pipeline(args: Optional[List[str]] = None):
    if not PIPELINE_PATH.exists():
        raise HTTPException(500, f"pipeline.py no existe en {PIPELINE_PATH}")
    with state.lock:
        if state.proc and state.proc.poll() is None:
            raise HTTPException(409, "El pipeline ya está corriendo.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"pipeline_{ts}.log"
        log_f = open(log_path, "a", buffering=1)

        cmd = [PYTHON_BIN, "-u", str(PIPELINE_PATH)]
        if args:
            cmd += args

        # IMPORTANTE: hereda variables de entorno del proceso (ya incluyen .env si lo cargaste)
        env = os.environ.copy()

        proc = subprocess.Popen(
            cmd,
            cwd=str(CWD),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        state.proc = proc
        state.started_at = time.time()
        state.log_file = log_path

        return {
            "status": "started",
            "pid": proc.pid,
            "log_file": str(log_path),
            "cwd": str(CWD),
            "cmd": cmd,
        }

def _status():
    with state.lock:
        running = state.proc is not None and state.proc.poll() is None
        rc = None if not state.proc else state.proc.poll()
        return {
            "running": running,
            "pid": None if not state.proc else state.proc.pid,
            "returncode": rc,
            "uptime_sec": None if not running else int(time.time() - (state.started_at or time.time())),
            "log_file": None if not state.log_file else str(state.log_file),
            "started_at": state.started_at,
            "pipeline_path": str(PIPELINE_PATH),
        }

def _tail_log(path: Path, kb: int) -> str:
    if not path.exists():
        return ""
    n = kb * 1024
    with open(path, "rb") as f:
        try:
            f.seek(-n, io.SEEK_END)
        except OSError:
            f.seek(0)
        data = f.read().decode(errors="replace")
    return data

def _stop(timeout: float = 5.0):
    with state.lock:
        if not state.proc or state.proc.poll() is not None:
            return {"status": "not_running"}

        state.proc.terminate()
        try:
            state.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            state.proc.kill()
        rc = state.proc.poll()
        pid = state.proc.pid
        state.proc = None
        return {"status": "stopped", "pid": pid, "returncode": rc}

# ---------- Endpoints ----------
@router.get("/run", tags=["pipeline"])
def run_get():
    # Arranca el pipeline usando SOLO el entorno del proceso (incluye tu .env)
    return _start_pipeline(args=None)

@router.get("/status", tags=["pipeline"])
def status_get():
    return _status()

@router.get("/logs", tags=["pipeline"], response_class=PlainTextResponse)
def logs_get(tail_kb: int = Query(64, ge=1, le=1024)):
    if not state.log_file:
        return PlainTextResponse("")
    return PlainTextResponse(_tail_log(state.log_file, tail_kb))

@router.post("/stop", tags=["pipeline"])
def stop_post(timeout_sec: float = Query(5.0, ge=1.0, le=60.0)):
    return _stop(timeout=timeout_sec)

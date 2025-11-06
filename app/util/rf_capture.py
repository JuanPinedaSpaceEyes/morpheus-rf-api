# app/util/rf_capture.py
import os, shutil, subprocess
from typing import Optional, Tuple, List

# Timeout pequeño para llamadas CLI
CLI_TIMEOUT_SEC = 3.0

def _in_flatpak() -> bool:
    return os.path.exists("/.flatpak-info") or "FLATPAK_ID" in os.environ or "/app" in os.environ.get("PATH", "")

def _spawn_host_prefix() -> List[str]:
    if _in_flatpak() and shutil.which("flatpak-spawn"):
        return ["flatpak-spawn", "--host"]
    return []

def _child_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = "/usr/local/bin:/usr/bin:/snap/bin:" + env.get("PATH", "")
    return env

def _resolve_cli_name() -> Optional[str]:
    """
    Busca bladeRF-cli/bladerf-cli respetando BLADERF_CLI si está seteado.
    Devuelve ruta absoluta si es posible; si estás en flatpak, puede devolver solo el nombre.
    """
    env_cli = (os.environ.get("BLADERF_CLI") or "").strip()
    if env_cli and os.path.exists(env_cli):
        return os.path.basename(env_cli) if _in_flatpak() else env_cli

    search_path = "/usr/local/bin:/usr/bin:/snap/bin:" + os.environ.get("PATH", "")
    for name in ("bladeRF-cli", "bladerf-cli"):
        p = shutil.which(name, path=search_path)
        if p:
            return os.path.basename(p) if _in_flatpak() else p

    # Como último recurso en flatpak, devuelve el nombre
    if _in_flatpak():
        return "bladeRF-cli"
    return None

def _probe_with_cli() -> Tuple[bool, str, str]:
    cli = _resolve_cli_name()
    if not cli:
        return False, "bladerf-cli", "CLI no encontrado en PATH ni BLADERF_CLI"
    argv = _spawn_host_prefix() + [cli, "-p"]
    try:
        p = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=_child_env(), timeout=CLI_TIMEOUT_SEC
        )
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out:
            low = out.lower()
            if any(s in low for s in ("bladerf", "nuand", "serial", "product")) and "no devices available" not in low:
                # Hay dispositivo(s) visibles para el CLI
                # Devuelve la primera línea a modo de resumen
                return True, "bladerf-cli -p", out.splitlines()[0]
            if "no devices available" in low:
                return False, "bladerf-cli -p", "No hay dispositivos disponibles"
        return False, "bladerf-cli -p", out or f"returncode={p.returncode}"
    except subprocess.TimeoutExpired:
        return False, "bladerf-cli -p", f"timeout>{CLI_TIMEOUT_SEC}s"
    except Exception as e:
        return False, "bladerf-cli -p", f"error: {e}"

def _probe_with_lsusb() -> Tuple[bool, str, str]:
    lsusb = shutil.which("lsusb") or "/usr/bin/lsusb"
    if not lsusb or not os.path.exists(lsusb):
        return False, "lsusb", "lsusb no disponible"
    argv = _spawn_host_prefix() + [lsusb]
    try:
        p = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=_child_env(), timeout=CLI_TIMEOUT_SEC
        )
        out = (p.stdout or "").strip()
        if p.returncode == 0 and out:
            low = out.lower()
            if "2cf0:" in low or "nuand" in low or "bladerf" in low:
                for line in out.splitlines():
                    if any(k in line.lower() for k in ("2cf0:", "nuand", "bladerf")):
                        return True, "lsusb", line.strip()
            return False, "lsusb", "No se encontró 2cf0/nuand/bladeRF"
        return False, "lsusb", out or f"returncode={p.returncode}"
    except subprocess.TimeoutExpired:
        return False, "lsusb", f"timeout>{CLI_TIMEOUT_SEC}s"
    except Exception as e:
        return False, "lsusb", f"error: {e}"

def probe_power_state() -> dict:
    """
    Intenta primero con bladeRF-cli -p; si no puede (CLI no disponible/timeout), cae a lsusb.
    Devuelve: {"powered": bool, "method": "bladerf-cli -p"|"lsusb", "message": str}
    """
    ok, method, msg = _probe_with_cli()
    # Si el CLI respondió (ok o no) sin ser un error de instalación/timeout/exception, usa su salida.
    if ok or ("CLI no encontrado" not in msg and "timeout" not in msg and "error" not in msg):
        return {"powered": ok, "method": method, "message": msg}
    # Fallback a lsusb cuando el CLI no está disponible o falló de forma no concluyente
    ok2, m2, msg2 = _probe_with_lsusb()
    return {"powered": ok2, "method": m2, "message": msg2}

__all__ = ["probe_power_state"]

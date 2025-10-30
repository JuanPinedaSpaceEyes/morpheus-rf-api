# app/inference_router.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional, Tuple
import os
from pathlib import Path

import numpy as np
from scipy import signal
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

router = APIRouter()

# ----------------------------
# Config: rutas de artefactos
# ----------------------------
DEFAULT_WEIGHTS = "/Users/juanjosesanchezpineda/Documents/WorkSpace/morpheus-rf-api/models/best_state.pth"
MODEL_WEIGHTS = Path(os.getenv("MODEL_WEIGHTS", str(DEFAULT_WEIGHTS))).resolve()
DEFAULT_SCALER = Path(__file__).resolve().parent.parent / "scaler.save"
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER))).resolve()

TARGET_H = 1024
TARGET_W = 1024
IN_CHANNELS = 2
NUM_CLASSES = 2
EPSILON = 1e-12

# ----------------------------
# Modelo (ConvNeXt-Tiny 2-canales)
# ----------------------------
class ConvNeXtTinyFineTuner(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 2, num_classes: int = 2, init_stem: bool = False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)

        # head -> 1 logit para binario
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 1)

        # adaptar stem a 2 canales
        old_stem = self.backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )
        if init_stem:
            with torch.no_grad():
                W = old_stem.weight
                W_mean = W.mean(dim=1, keepdim=True)
                new_stem.weight.copy_(W_mean.repeat(1, in_channels, 1, 1))
                if old_stem.bias is not None:
                    new_stem.bias.copy_(old_stem.bias)
        self.backbone.features[0][0] = new_stem

        # congelar salvo stem + head
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.features[0][0].parameters():
            p.requires_grad = True
        for p in self.backbone.classifier[2].parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: [B,2,H,W] -> [B,1]
        return self.backbone(x).squeeze(-1).squeeze(-1)

# ----------------------------
# Carga artefactos al importar
# ----------------------------
if not MODEL_WEIGHTS.exists():
    raise RuntimeError(f"No encontré pesos del modelo en {MODEL_WEIGHTS}. "
                       f"Configura MODEL_WEIGHTS o coloca models/best_state.pth.")

_model = ConvNeXtTinyFineTuner(pretrained=True, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, init_stem=False)
# compat con distintas versiones de torch
try:
    state = torch.load(str(MODEL_WEIGHTS), weights_only=True, map_location=torch.device("cpu"))
except TypeError:
    state = torch.load(str(MODEL_WEIGHTS), map_location=torch.device("cpu"))
_model.load_state_dict(state, strict=False)
_model.eval()

_scaler = None
if SCALER_PATH.exists():
    _scaler = joblib.load(str(SCALER_PATH))

# ----------------------------
# Esquemas (Pydantic v2)
# ----------------------------
class IQData(BaseModel):
    # O envías I y Q por separado...
    I: Optional[List[float]] = None
    Q: Optional[List[float]] = None
    # ...o una lista de pares [[I,Q], [I,Q], ...]
    IQ: Optional[List[Tuple[float, float]]] = None

    @model_validator(mode="after")
    def validate_shapes(self) -> "IQData":
        if self.IQ is not None:
            if (self.I is not None) or (self.Q is not None):
                raise ValueError("Usa EITHER 'IQ' o 'I'+'Q', no ambos.")
            if len(self.IQ) == 0:
                raise ValueError("'IQ' no puede estar vacío.")
        else:
            if self.I is None or self.Q is None:
                raise ValueError("Si no usas 'IQ', debes enviar 'I' y 'Q'.")
            if len(self.I) != len(self.Q):
                raise ValueError("'I' y 'Q' deben tener la MISMA longitud.")
        return self

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.IQ is not None:
            arr = np.asarray(self.IQ, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("'IQ' debe ser lista de pares [I,Q].")
            return arr[:, 0], arr[:, 1]
        else:
            I = np.asarray(self.I, dtype=np.float32)
            Q = np.asarray(self.Q, dtype=np.float32)
            return I, Q

class InputMeta(BaseModel):
    Scaled: bool = Field(False, description="Si False, se aplica scaler (si existe). Si True, se asume ya escalado.")
    Threshold: float = Field(0.5, ge=0.0, le=1.0, description="Umbral de clasificación sobre la probabilidad.")
    SampleRate: Optional[float] = Field(40e6, description="Frecuencia de muestreo para spectrogram (Hz).")
    Nperseg: Optional[int] = Field(2048, description="Tamaño de ventana para spectrogram().")
    Noverlap: Optional[int] = Field(1024, description="Solapamiento para spectrogram(). Debe ser < Nperseg.")

class PipelineInput(BaseModel):
    Dtype: Literal["Float", "float", "float32", "float64"] = "Float"
    Data: IQData
    Meta: InputMeta

class PipelineOutput(BaseModel):
    Output: dict

# ----------------------------
# Utils
# ----------------------------
def compute_spectrograms(I: np.ndarray, Q: np.ndarray, fs: float, nperseg: int, noverlap: int):
    if not (0 <= noverlap < nperseg):
        raise HTTPException(400, f"Meta.Noverlap debe cumplir 0 <= noverlap < nperseg (got {noverlap} vs {nperseg}).")
    fI, tI, SxxI = signal.spectrogram(I, fs=fs, nperseg=nperseg, noverlap=noverlap)
    fQ, tQ, SxxQ = signal.spectrogram(Q, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # log10 power-safe
    SxxI = np.log10(np.maximum(SxxI, EPSILON)).astype(np.float32)
    SxxQ = np.log10(np.maximum(SxxQ, EPSILON)).astype(np.float32)
    return SxxI, SxxQ

def to_model_tensor(SxxI: np.ndarray, SxxQ: np.ndarray, scaled: bool) -> torch.Tensor:
    feat = np.stack([SxxI, SxxQ], axis=0)  # [2,F,T]
    if not scaled:
        if _scaler is None:
            raise HTTPException(500, "No encontré scaler.save y Meta.Scaled=False. "
                                     "Configura SCALER_PATH o envía Meta.Scaled=True.")
        flat = feat.reshape(1, -1)  # [1, 2*F*T]
        flat_scaled = _scaler.transform(flat)  # numpy
        feat = flat_scaled.reshape(1, 2, SxxI.shape[0], SxxI.shape[1]).astype(np.float32)
    else:
        feat = feat.reshape(1, 2, SxxI.shape[0], SxxI.shape[1]).astype(np.float32)

    x = torch.from_numpy(feat)  # [1,2,F,T]
    x = F.interpolate(x, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
    return x

# ----------------------------
# Endpoint principal
# ----------------------------
@router.post("/infer", response_model=PipelineOutput, tags=["pipeline"])
def infer(payload: PipelineInput):
    try:
        I, Q = payload.Data.to_arrays()
        fs = float(payload.Meta.SampleRate or 40e6)
        nperseg = int(payload.Meta.Nperseg or 2048)
        noverlap = int(payload.Meta.Noverlap or (nperseg // 2))

        SxxI, SxxQ = compute_spectrograms(I, Q, fs=fs, nperseg=nperseg, noverlap=noverlap)
        x = to_model_tensor(SxxI, SxxQ, scaled=payload.Meta.Scaled)

        with torch.no_grad():
            logit = _model(x).cpu().numpy().reshape(-1)  # [1]
            prob = 1.0 / (1.0 + np.exp(-logit))

        thr = float(payload.Meta.Threshold)
        label = int(prob[0] >= thr)

        return {
            "Output": {
                "Logit": [float(logit[0])],
                "Probabilíty": [float(prob[0])],   # (mantengo la clave con tilde tal como pediste)
                "meta": { "Threshold": thr },
                "Label": label
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en inferencia: {type(e).__name__}: {e}")

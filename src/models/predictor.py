"""
Inferencia de la Fase 6 (Despliegue): carga los checkpoints de Fase 4 y los umbrales de
tamaño, y predice calidad y tamaño de una imagen nueva (archivo subido o foto de cámara).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data.dataset_loader import FEATURE_NAMES, compute_feature_vector, prepare_rgb_array

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_DIR = REPO_ROOT / "experiments" / "checkpoints"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# Modelo ganador confirmado en Fase 5 (mayor F1-macro en fotos reales) + alternativa comparable.
MODELS: Dict[str, str] = {
    "MLP (ganador Fase 5)": "fase4_mlp.joblib",
    "Random Forest": "fase4_random_forest.joblib",
}
QUALITY_LABELS_ES: Dict[str, str] = {"Good": "Buena", "Mixed": "Regular", "Bad": "Mala"}
SIZE_LABELS_ES: Dict[str, str] = {"Small": "Pequeño", "Medium": "Mediano", "Large": "Grande"}
FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
FRUITS_ES: Dict[str, str] = {
    "Apple": "Manzana", "Banana": "Banano", "Guava": "Guayaba",
    "Lime": "Limón", "Orange": "Naranja", "Pomegranate": "Granada",
}


@dataclass
class Prediction:
    quality: str
    probabilities: Dict[str, float]
    d_norm: float
    size: str
    thresholds: Dict[str, float]
    thresholds_source: str
    features: Dict[str, float]
    mask: np.ndarray          # máscara 128x128 usada para las características
    image: Image.Image        # imagen orientada (EXIF) mostrada al usuario
    mask_is_fallback: bool    # True si la segmentación falló y se usó el círculo central


@lru_cache(maxsize=None)
def load_model(checkpoint: str):
    """Carga un Pipeline (StandardScaler + clasificador) y valida el contrato de entrada."""
    model = joblib.load(CHECKPOINTS_DIR / checkpoint)
    if list(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError(f"{checkpoint} espera otro orden de características")
    return model


@lru_cache(maxsize=None)
def load_size_thresholds() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Umbrales q33/q66 de d_norm por fruta:
    - 'kaggle': terciles aprendidos en Train de Kaggle (fotos de estudio).
    - 'local': terciles recalibrados en Fase 5 con las fotos propias (cámara real).
    """
    kaggle = json.loads((RESULTS_DIR / "size_thresholds_train.json").read_text(encoding="utf-8"))
    fase5 = json.loads((RESULTS_DIR / "fase5_evaluacion_metricas.json").read_text(encoding="utf-8"))
    return {"kaggle": kaggle, "local": fase5["size_evaluation"]["umbrales_locales"]}


def size_category(d_norm: float, q33: float, q66: float) -> str:
    """Regla por terciles: Small si d < q33, Medium si q33 <= d < q66, Large si d >= q66."""
    if d_norm < q33:
        return "Small"
    if d_norm < q66:
        return "Medium"
    return "Large"


def resolve_thresholds(fruit: str, source: str) -> tuple[Dict[str, float], str]:
    """Devuelve los umbrales de la fuente pedida; si la fruta no tiene umbral local, usa Kaggle."""
    table = load_size_thresholds()
    if source == "local" and fruit in table["local"]:
        return table["local"][fruit], "local"
    return table["kaggle"][fruit], "kaggle"


def predict(img: Image.Image, checkpoint: str, fruit: str, threshold_source: str = "local") -> Prediction:
    """Pipeline completo: orientación EXIF -> 128x128 -> segmentación -> 31 features -> modelo."""
    img = ImageOps.exif_transpose(img).convert("RGB")   # igual que prepare_custom_photos.py
    rgb_arr = prepare_rgb_array(img)
    feats, mask = compute_feature_vector(rgb_arr)

    model = load_model(checkpoint)
    X = pd.DataFrame([feats], columns=FEATURE_NAMES)
    proba = model.predict_proba(X)[0]
    probabilities = {cls: float(p) for cls, p in zip(model.classes_, proba)}
    quality = max(probabilities, key=probabilities.get)

    thresholds, used_source = resolve_thresholds(fruit, threshold_source)
    d_norm = feats["d_norm"]
    return Prediction(
        quality=quality,
        probabilities=probabilities,
        d_norm=d_norm,
        size=size_category(d_norm, thresholds["q33"], thresholds["q66"]),
        thresholds=thresholds,
        thresholds_source=used_source,
        features=feats,
        mask=mask,
        image=img,
        mask_is_fallback=_is_center_fallback(mask),
    )


def _is_center_fallback(mask: np.ndarray) -> bool:
    """Detecta si la máscara es exactamente el círculo de respaldo de segment_foreground_fast."""
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    circle = ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) <= (min(h, w) * 0.35) ** 2
    return bool(np.array_equal(mask, circle))


def predict_path(image_path: str | Path, checkpoint: Optional[str] = None,
                 fruit: str = "Apple", threshold_source: str = "local") -> Prediction:
    """Atajo para scripts y pruebas: predice desde una ruta en disco."""
    with Image.open(image_path) as img:
        img.load()
        return predict(img, checkpoint or MODELS["MLP (ganador Fase 5)"], fruit, threshold_source)

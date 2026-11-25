"""
Interfaz gráfica del sistema (Fase 6 de CRISP-DM: Despliegue).

Ejecutar desde la raíz del repositorio:
    streamlit run src/main.py

Flujo: imagen (archivo o cámara) -> orientación EXIF -> 128x128 -> segmentación por saliencia
+ Otsu -> 31 características -> Pipeline de Fase 4 (StandardScaler + MLP o Random Forest)
-> calidad + probabilidades; d_norm + umbrales por fruta -> tamaño.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/main.py` pone src/ en sys.path; se agrega la raíz para importar `src.*`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.models.predictor import (
    FRUITS, FRUITS_ES, MODELS, QUALITY_LABELS_ES, SIZE_LABELS_ES, load_model, predict,
)
from src.utils.visualization import mask_overlay

try:  # fotos HEIC de iPhone
    from pillow_heif import register_heif_opener
    register_heif_opener()
    UPLOAD_TYPES = ["jpg", "jpeg", "png", "heic", "heif"]
except ImportError:
    UPLOAD_TYPES = ["jpg", "jpeg", "png"]

QUALITY_COLORS = {"Good": "#2ca02c", "Mixed": "#ff7f0e", "Bad": "#d62728"}
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Clasificador de calidad de frutas", page_icon="🍎", layout="wide")

# ----------------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Configuración")
    model_name = st.selectbox("Modelo", list(MODELS), index=0,
                              help="MLP: ganador de la Fase 5. Random Forest: alternativa para comparar.")
    fruit = st.selectbox("Tipo de fruta", FRUITS, format_func=lambda f: f"{FRUITS_ES[f]} ({f})",
                         help="Define qué umbrales de d_norm se usan para el tamaño.")
    source_label = st.radio(
        "Umbrales de tamaño",
        ["Cámara real (recalibrados en Fase 5)", "Kaggle (fotos de estudio)"],
        help="Las fotos de celular encuadran la fruta distinto a Kaggle; los umbrales locales "
             "corrigen ese corrimiento de d_norm.",
    )
    threshold_source = "local" if source_label.startswith("Cámara") else "kaggle"
    st.divider()
    st.caption("Fondo simple y uniforme, una sola fruta centrada, buena luz y sin reflejos fuertes. "
               "Ver docs/protocolo_recoleccion_fotos.md.")

# ----------------------------------------------------------------------------- entrada
st.title("Clasificación de calidad y tamaño de frutas")
st.write("Sube una foto o tómala con la cámara. El sistema predice la **calidad** "
         "(Buena / Regular / Mala) y estima el **tamaño** a partir del diámetro normalizado $d_{norm}$.")

tab_upload, tab_camera = st.tabs(["📁 Subir imagen", "📷 Cámara en vivo"])
with tab_upload:
    uploaded = st.file_uploader("Imagen de una fruta", type=UPLOAD_TYPES)
with tab_camera:
    captured = st.camera_input("Ubica la fruta frente a la cámara y toma la foto")

source_file = captured or uploaded
if source_file is None:
    st.info("Esperando una imagen…")
    st.stop()

try:
    image = Image.open(source_file)
    image.load()
except Exception as exc:  # archivo corrupto o formato no soportado
    st.error(f"No se pudo leer la imagen: {exc}")
    st.stop()

load_model(MODELS[model_name])  # valida el checkpoint antes de predecir
result = predict(image, MODELS[model_name], fruit, threshold_source)

# ----------------------------------------------------------------------------- resultados
col_q, col_s = st.columns(2)
top_p = result.probabilities[result.quality]
with col_q:
    st.subheader("Calidad")
    color = QUALITY_COLORS[result.quality]
    st.markdown(f"<h2 style='color:{color};margin:0'>{QUALITY_LABELS_ES[result.quality]} "
                f"({result.quality})</h2>", unsafe_allow_html=True)
    st.caption(f"Confianza del modelo: {top_p:.0%} · {model_name}")
    for cls, p in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
        st.progress(p, text=f"{QUALITY_LABELS_ES[cls]} ({cls}): {p:.1%}")

with col_s:
    st.subheader("Tamaño")
    st.markdown(f"<h2 style='margin:0'>{SIZE_LABELS_ES[result.size]} ({result.size})</h2>",
                unsafe_allow_html=True)
    st.metric("Diámetro normalizado d_norm", f"{result.d_norm:.3f}")
    origin = "recalibrados con fotos propias" if result.thresholds_source == "local" else "de Kaggle"
    st.caption(f"Umbrales {origin} para {FRUITS_ES[fruit]}: pequeño < {result.thresholds['q33']:.3f} ≤ "
               f"mediano < {result.thresholds['q66']:.3f} ≤ grande")
    if threshold_source == "local" and result.thresholds_source == "kaggle":
        st.caption("No hay fotos propias de esta fruta; se usan los umbrales de Kaggle.")

if top_p < LOW_CONFIDENCE:
    st.warning(f"Confianza baja ({top_p:.0%}). Toma la foto con fondo uniforme y mejor luz, o revísala a mano.")
if result.mask_is_fallback:
    st.warning("No se pudo separar la fruta del fondo; se usó una región central. "
               "El tamaño y la calidad no son confiables para esta foto.")

st.subheader("Segmentación")
col_img, col_mask = st.columns(2)
with col_img:
    st.image(result.image, caption="Imagen original", width="stretch")
with col_mask:
    st.image(mask_overlay(result.image, result.mask),
             caption=f"Máscara y contorno (área relativa = {result.features['area_ratio']:.2f})",
             width="stretch")

with st.expander("Ver las 31 características extraídas"):
    st.dataframe(pd.DataFrame([result.features]).T.rename(columns={0: "valor"}), width="stretch")

st.caption("Aviso: en la Fase 5 el MLP obtuvo F1-macro 0.97 en el test de Kaggle pero 0.32 en fotos "
           "propias (cambio de dominio). Use la predicción como apoyo, no como decisión final.")

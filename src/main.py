"""
Punto de entrada de la interfaz gráfica (Fase 6 de CRISP-DM: Despliegue).
Este módulo alojará la aplicación interactiva (Streamlit / GUI) que carga los modelos
entrenados en los cuadernos de `notebooks/` y predice en tiempo real la Calidad y el Tamaño.
"""

from pathlib import Path
from src.data.dataset_loader import extract_image_features


def predict_single_image(image_path: str | Path) -> dict:
    """
    Extrae las características morfométricas (d_norm, tamaño relativo) y de color/textura
    para una imagen cargada desde la interfaz gráfica o capturada por cámara.
    """
    feats = extract_image_features(image_path)
    d_norm = feats["d_norm"]
    size_label = "Small" if d_norm < 0.52 else ("Medium" if d_norm < 0.68 else "Large")
    return {
        "d_norm": d_norm,
        "area_ratio": feats["area_ratio"],
        "estimated_size": size_label,
        "features": feats,
    }


if __name__ == "__main__":
    print("Módulo de interfaz listo. Ejecute los cuadernos en `notebooks/` para EDA y entrenamiento.")

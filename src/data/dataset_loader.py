"""
Módulo de carga e indexación unificada del dataset de frutas (Kaggle + Recolección Propia)
y cálculo matemático de descriptores geométricos (tamaño relativo d_norm) y colorimétricos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


# Mapeo canónico para unificar nombres entre Good/Bad (Lime) y Mixed (Lemon)
CANONICAL_FRUIT_MAP: Dict[str, str] = {
    "apple": "Apple",
    "banana": "Banana",
    "guava": "Guava",
    "lime": "Lime",
    "lemon": "Lime",
    "orange": "Orange",
    "pomegranate": "Pomegranate",
}

QUALITY_FOLDERS: Dict[str, str] = {
    "Good Quality_Fruits": "Good",
    "Bad Quality_Fruits": "Bad",
    "Mixed Qualit_Fruits": "Mixed",
}

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_workspace_root() -> Path:
    """Retorna la ruta raíz donde residen las carpetas del dataset y el repositorio."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root.parent


def normalize_fruit_name(raw_folder_name: str) -> str:
    """
    Normaliza el nombre de la subcarpeta (ej. 'Apple_Good', 'Lime_Bad', 'Lemon')
    al nombre canónico de la fruta ('Apple', 'Banana', 'Guava', 'Lime', 'Orange', 'Pomegranate').
    """
    base = raw_folder_name.split("_")[0].strip().lower()
    return CANONICAL_FRUIT_MAP.get(base, raw_folder_name)


def build_dataset_inventory(workspace_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Escanea las carpetas de Kaggle ('Good Quality_Fruits', 'Bad Quality_Fruits',
    'Mixed Qualit_Fruits') y construye un DataFrame completo con cada imagen.
    """
    if workspace_root is None:
        workspace_root = get_workspace_root()

    records: List[Dict[str, object]] = []

    for folder_name, quality_label in QUALITY_FOLDERS.items():
        folder_path = workspace_root / folder_name
        if not folder_path.exists():
            continue

        for subfolder in sorted(folder_path.iterdir()):
            if not subfolder.is_dir():
                continue
            fruit_name = normalize_fruit_name(subfolder.name)
            for img_path in sorted(subfolder.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    records.append(
                        {
                            "filepath": str(img_path),
                            "filename": img_path.name,
                            "source_folder": folder_name,
                            "subfolder": subfolder.name,
                            "fruit": fruit_name,
                            "quality": quality_label,
                            "class_combined": f"{fruit_name}_{quality_label}",
                            "file_size_kb": round(img_path.stat().st_size / 1024.0, 2),
                        }
                    )

    df = pd.DataFrame(records)
    return df


def otsu_threshold(gray_values: np.ndarray) -> float:
    """
    Calcula el umbral óptimo de Otsu maximizando la varianza inter-clase:
        sigma_B^2(t) = w_0(t) * w_1(t) * (mu_0(t) - mu_1(t))^2
    """
    hist, bin_edges = np.histogram(gray_values.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    prob = hist / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    w0 = np.cumsum(prob)
    w1 = 1.0 - w0
    mu = np.cumsum(prob * bin_centers)
    mu_total = mu[-1]

    # Evitar división por cero
    valid = (w0 > 1e-6) & (w1 > 1e-6)
    sigma_b_sq = np.zeros_like(w0)
    sigma_b_sq[valid] = ((mu_total * w0[valid] - mu[valid]) ** 2) / (w0[valid] * w1[valid])

    best_idx = int(np.argmax(sigma_b_sq))
    return float(bin_centers[best_idx])


def rgb_to_hsv_array(rgb_norm: np.ndarray) -> np.ndarray:
    """
    Convierte un tensor RGB normalizado en [0, 1] de dimensión (H, W, 3)
    al espacio HSV en [0, 1].
    """
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc

    s = np.where(maxc > 1e-6, deltac / np.maximum(maxc, 1e-6), 0.0)
    h = np.zeros_like(maxc)

    mask = deltac > 1e-6
    rc = np.where(mask, (maxc - r) / np.maximum(deltac, 1e-6), 0.0)
    gc = np.where(mask, (maxc - g) / np.maximum(deltac, 1e-6), 0.0)
    bc = np.where(mask, (maxc - b) / np.maximum(deltac, 1e-6), 0.0)

    cond_r = mask & (r == maxc)
    cond_g = mask & (g == maxc) & ~cond_r
    cond_b = mask & ~cond_r & ~cond_g

    h[cond_r] = bc[cond_r] - gc[cond_r]
    h[cond_g] = 2.0 + rc[cond_g] - bc[cond_g]
    h[cond_b] = 4.0 + gc[cond_b] - rc[cond_b]
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def segment_fruit_foreground(rgb_norm: np.ndarray) -> np.ndarray:
    """
    Segmenta el ejemplar de fruta respecto al fondo combinando:
    1. Contraste cromático y de luminancia respecto al borde perimetral de la imagen.
    2. Saturación en espacio HSV.
    3. Umbralización automática de Otsu y limpieza morfológica (relleno de huecos
       y selección de componente conexa principal).
    Retorna máscara booleana M de dimensión (H, W).
    """
    h, w, _ = rgb_norm.shape
    hsv = rgb_to_hsv_array(rgb_norm)

    # Estimar el color mediano del fondo usando un marco perimetral de 6 píxeles
    border_px = max(2, min(h, w) // 25)
    top = rgb_norm[:border_px, :, :].reshape(-1, 3)
    bottom = rgb_norm[-border_px:, :, :].reshape(-1, 3)
    left = rgb_norm[:, :border_px, :].reshape(-1, 3)
    right = rgb_norm[:, -border_px:, :].reshape(-1, 3)
    border_samples = np.concatenate([top, bottom, left, right], axis=0)
    bg_rgb = np.median(border_samples, axis=0)

    # Distancia euclidiana en RGB respecto al fondo + componente de saturación S
    color_dist = np.linalg.norm(rgb_norm - bg_rgb[None, None, :], axis=-1) / math.sqrt(3.0)
    saturation = hsv[..., 1]
    salience = np.clip(0.70 * color_dist + 0.30 * saturation, 0.0, 1.0)

    thresh = otsu_threshold(salience)
    mask = salience > max(thresh, 0.12)

    # Operaciones morfológicas: cierre, relleno de agujeros y componente principal
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)

    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        mask = labeled == largest_label

    # Fallback seguro si la máscara queda vacía
    if mask.sum() < 0.02 * (h * w):
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r = min(h, w) * 0.35
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)

    return mask


def extract_image_features(image_path: str | Path) -> Dict[str, float]:
    """
    Extrae descriptores matemáticos geométricos (tamaño relativo d_norm, área normalizada,
    relación de aspecto de bounding box) y estadísticos de color/textura (RGB, HSV, Laplaciano).
    """
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        rgb_arr = np.asarray(img_rgb, dtype=np.float32) / 255.0

    return extract_rgb_features(rgb_arr)


def extract_rgb_features(rgb_arr: np.ndarray) -> Dict[str, float]:
    """Extrae descriptores de una imagen RGB normalizada en [0, 1]."""
    height, width, _ = rgb_arr.shape
    mask = segment_fruit_foreground(rgb_arr)
    hsv_arr = rgb_to_hsv_array(rgb_arr)

    total_pixels = float(height * width)
    fruit_area_px = float(mask.sum())
    area_ratio = fruit_area_px / total_pixels

    # Diámetro equivalente en píxeles normalizados: d_norm = sqrt(4 * A / (pi * H * W))
    d_norm = math.sqrt((4.0 * fruit_area_px) / (math.pi * total_pixels))

    # Bounding box de la fruta segmentada
    ys, xs = np.where(mask)
    if len(ys) > 0:
        bbox_h = float(ys.max() - ys.min() + 1) / float(height)
        bbox_w = float(xs.max() - xs.min() + 1) / float(width)
        bbox_aspect = bbox_w / max(bbox_h, 1e-6)
    else:
        bbox_h, bbox_w, bbox_aspect = 0.5, 0.5, 1.0

    # Estadísticos de color únicamente sobre los píxeles de la fruta M(u,v) = 1
    fg_rgb = rgb_arr[mask]
    fg_hsv = hsv_arr[mask]

    # Textura / nitidez mediante varianza del operador Laplaciano discreto
    gray = 0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]
    laplacian = ndimage.laplace(gray)
    laplacian_var = float(np.var(laplacian[mask])) if mask.any() else float(np.var(laplacian))

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": round(float(width) / float(height), 4),
        "area_ratio": round(area_ratio, 4),
        "d_norm": round(d_norm, 4),
        "bbox_width_norm": round(bbox_w, 4),
        "bbox_height_norm": round(bbox_h, 4),
        "bbox_aspect_ratio": round(bbox_aspect, 4),
        "mean_R": round(float(fg_rgb[:, 0].mean()), 4),
        "mean_G": round(float(fg_rgb[:, 1].mean()), 4),
        "mean_B": round(float(fg_rgb[:, 2].mean()), 4),
        "std_R": round(float(fg_rgb[:, 0].std()), 4),
        "std_G": round(float(fg_rgb[:, 1].std()), 4),
        "std_B": round(float(fg_rgb[:, 2].std()), 4),
        "mean_H": round(float(fg_hsv[:, 0].mean()), 4),
        "mean_S": round(float(fg_hsv[:, 1].mean()), 4),
        "mean_V": round(float(fg_hsv[:, 2].mean()), 4),
        "std_S": round(float(fg_hsv[:, 1].std()), 4),
        "std_V": round(float(fg_hsv[:, 2].std()), 4),
        "laplacian_var": round(laplacian_var, 6),
    }


def assign_relative_size_labels(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna la categoría de tamaño relativo ('Small', 'Medium', 'Large')
    usando los terciles empíricos (q_33, q_66) del diámetro normalizado d_norm
    dentro de cada tipo de fruta.
    """
    df = df_features.copy()

    def _categorize_group(series: pd.Series) -> pd.Series:
        q1 = series.quantile(1.0 / 3.0)
        q2 = series.quantile(2.0 / 3.0)
        return pd.Series(
            np.where(series < q1, "Small", np.where(series < q2, "Medium", "Large")),
            index=series.index,
        )

    df["size_category"] = df.groupby("fruit", group_keys=False)["d_norm"].apply(_categorize_group)
    return df


# ---------------------------------------------------------------------------
# Fase 6 (Despliegue): extractor de las 31 características de los modelos.
# Réplica exacta de `segment_foreground_fast` y `extract_full_feature_vector`
# del cuaderno 02_Preparacion_Datos_y_Extraccion_Caracteristicas.ipynb, de modo
# que una imagen nueva produzca el mismo vector que vieron los modelos de Fase 4.
# ---------------------------------------------------------------------------

TARGET_SIZE: Tuple[int, int] = (128, 128)

# Orden exacto esperado por los checkpoints (feature_names en fase5_evaluacion_metricas.json).
FEATURE_NAMES: List[str] = [
    "area_ratio", "d_norm", "bbox_width_norm", "bbox_height_norm", "bbox_aspect_ratio",
    "mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B",
    "mean_H", "mean_S", "mean_V", "std_H", "std_S", "std_V",
    "laplacian_var", "laplacian_abs_mean", "sobel_mean", "sobel_std",
    "dark_pixel_ratio", "necrotic_pixel_ratio",
] + [f"hue_bin_{i}" for i in range(8)]


def segment_foreground_fast(rgb_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta la fruta sobre fondo uniforme (cuaderno 02).

    1. Estima el color de fondo b como la mediana de una franja de borde de ancho min(h, w)/25.
    2. Saliencia s(x) = 0.70 * ||I(x) - b||_2 / sqrt(3) + 0.30 * S(x), con S la saturación HSV.
    3. Umbral de Otsu sobre s (piso 0.12), cierre morfológico 3x3 y relleno de huecos.
    4. Conserva la componente conexa de mayor área; si cubre < 2 % de la imagen,
       usa un círculo central de radio 0.35 * min(h, w) como respaldo.

    Retorna (máscara booleana h x w, imagen HSV h x w x 3).
    """
    h, w, _ = rgb_norm.shape
    hsv = rgb_to_hsv_array(rgb_norm)
    b_px = max(2, min(h, w) // 25)
    border = np.concatenate([
        rgb_norm[:b_px, :, :].reshape(-1, 3),
        rgb_norm[-b_px:, :, :].reshape(-1, 3),
        rgb_norm[:, :b_px, :].reshape(-1, 3),
        rgb_norm[:, -b_px:, :].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(border, axis=0)
    color_dist = np.linalg.norm(rgb_norm - bg_rgb[None, None, :], axis=-1) / math.sqrt(3.0)
    salience = np.clip(0.70 * color_dist + 0.30 * hsv[..., 1], 0.0, 1.0)
    thresh = otsu_threshold(salience)
    mask = salience > max(thresh, 0.12)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    labeled, num_feat = ndimage.label(mask)
    if num_feat > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_feat + 1))
        mask = labeled == (int(np.argmax(sizes)) + 1)
    if mask.sum() < 0.02 * (h * w):
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) <= (min(h, w) * 0.35) ** 2
    return mask, hsv


def prepare_rgb_array(img: Image.Image) -> np.ndarray:
    """Convierte a RGB, redimensiona a TARGET_SIZE (bilineal) y normaliza a [0, 1]."""
    img_resized = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(img_resized, dtype=np.float32) / 255.0


def compute_feature_vector(rgb_arr: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calcula las 31 características sobre una imagen ya preparada (128x128, [0, 1]).

    Geometría: área relativa A = |M| / (h w) y diámetro equivalente normalizado
    d_norm = sqrt(4 |M| / (pi h w)), es decir, el diámetro del círculo de igual área
    dividido por el lado de la imagen. Color: media y desviación de RGB y HSV sobre la
    fruta, más un histograma de matiz de 8 bins. Textura: varianza y media absoluta del
    Laplaciano y media/desviación de la magnitud del gradiente de Sobel en luminancia.
    Defectos: proporción de píxeles oscuros (V < 0.25) y necróticos (V < 0.35 y S < 0.40).

    Retorna (diccionario de características, máscara de segmentación).
    """
    h, w, _ = rgb_arr.shape
    mask, hsv_arr = segment_foreground_fast(rgb_arr)
    total_px = float(h * w)
    fruit_px = float(mask.sum())
    area_ratio = fruit_px / total_px
    d_norm = math.sqrt((4.0 * fruit_px) / (math.pi * total_px))

    ys, xs = np.where(mask)
    bbox_h = float(ys.max() - ys.min() + 1) / float(h)
    bbox_w = float(xs.max() - xs.min() + 1) / float(w)
    bbox_aspect = bbox_w / max(bbox_h, 1e-6)

    fg_rgb = rgb_arr[mask]
    fg_hsv = hsv_arr[mask]

    # Histograma normalizado de matiz H (8 bins en [0, 1])
    h_hist, _ = np.histogram(fg_hsv[:, 0], bins=8, range=(0.0, 1.0), density=False)
    h_hist = h_hist.astype(np.float32) / max(float(h_hist.sum()), 1.0)

    # Textura: Laplaciano y gradiente de Sobel en luminancia
    gray = 0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2]
    lap = ndimage.laplace(gray)[mask]
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    grad_mag = np.hypot(gx, gy)[mask]

    # Proporción de píxeles oscuros / necróticos
    dark_ratio = float((fg_hsv[:, 2] < 0.25).mean())
    necrotic_ratio = float(((fg_hsv[:, 2] < 0.35) & (fg_hsv[:, 1] < 0.40)).mean())

    feats = {
        "area_ratio": round(area_ratio, 5),
        "d_norm": round(d_norm, 5),
        "bbox_width_norm": round(bbox_w, 5),
        "bbox_height_norm": round(bbox_h, 5),
        "bbox_aspect_ratio": round(bbox_aspect, 5),
        "mean_R": round(float(fg_rgb[:, 0].mean()), 5),
        "mean_G": round(float(fg_rgb[:, 1].mean()), 5),
        "mean_B": round(float(fg_rgb[:, 2].mean()), 5),
        "std_R": round(float(fg_rgb[:, 0].std()), 5),
        "std_G": round(float(fg_rgb[:, 1].std()), 5),
        "std_B": round(float(fg_rgb[:, 2].std()), 5),
        "mean_H": round(float(fg_hsv[:, 0].mean()), 5),
        "mean_S": round(float(fg_hsv[:, 1].mean()), 5),
        "mean_V": round(float(fg_hsv[:, 2].mean()), 5),
        "std_H": round(float(fg_hsv[:, 0].std()), 5),
        "std_S": round(float(fg_hsv[:, 1].std()), 5),
        "std_V": round(float(fg_hsv[:, 2].std()), 5),
        "laplacian_var": round(float(np.var(lap)), 6),
        "laplacian_abs_mean": round(float(np.abs(lap).mean()), 6),
        "sobel_mean": round(float(grad_mag.mean()), 6),
        "sobel_std": round(float(grad_mag.std()), 6),
        "dark_pixel_ratio": round(dark_ratio, 5),
        "necrotic_pixel_ratio": round(necrotic_ratio, 5),
    }
    for b_idx in range(8):
        feats[f"hue_bin_{b_idx}"] = round(float(h_hist[b_idx]), 5)
    return {name: feats[name] for name in FEATURE_NAMES}, mask


def extract_full_feature_vector(image_path: str | Path) -> Dict[str, float]:
    """Vector de 31 características de una imagen en disco (misma firma que el cuaderno 02)."""
    with Image.open(image_path) as img:
        rgb_arr = prepare_rgb_array(img)
    feats, _ = compute_feature_vector(rgb_arr)
    return feats

"""Utilidades de visualización para la interfaz: superposición de máscara y contorno."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def mask_overlay(img: Image.Image, mask: np.ndarray, max_side: int = 640) -> Image.Image:
    """
    Escala la máscara 128x128 al tamaño de la imagen, oscurece el fondo y dibuja el contorno
    de la fruta en verde. La máscara se agranda con vecino más cercano para no inventar bordes.
    """
    view = img.copy()
    view.thumbnail((max_side, max_side))
    w, h = view.size
    big = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 0

    arr = np.asarray(view, dtype=np.float32)
    arr[~big] *= 0.25
    thickness = max(2, min(w, h) // 160)
    edge = big & ~ndimage.binary_erosion(big, iterations=thickness)
    arr[edge] = (0, 255, 136)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

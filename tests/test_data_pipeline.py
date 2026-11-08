"""Pruebas unitarias para el módulo de carga de datos y extracción geométrica (PI3)."""

import unittest
import numpy as np

from src.data.dataset_loader import (
    normalize_fruit_name,
    otsu_threshold,
    rgb_to_hsv_array,
    segment_fruit_foreground,
)


class TestDataPipeline(unittest.TestCase):
    def test_normalize_fruit_name(self) -> None:
        self.assertEqual(normalize_fruit_name("Apple_Good"), "Apple")
        self.assertEqual(normalize_fruit_name("Lime_Bad"), "Lime")
        self.assertEqual(normalize_fruit_name("Lemon"), "Lime")
        self.assertEqual(normalize_fruit_name("Pomegranate_Good"), "Pomegranate")

    def test_rgb_to_hsv_range(self) -> None:
        synthetic_rgb = np.random.default_rng(42).uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)
        hsv = rgb_to_hsv_array(synthetic_rgb)
        self.assertEqual(hsv.shape, (32, 32, 3))
        self.assertTrue(np.all(hsv >= 0.0) and np.all(hsv <= 1.0))

    def test_segmentation_synthetic_circle(self) -> None:
        # Crear imagen sintética 100x100 con fondo blanco y fruta roja circular de radio 25
        h, w = 100, 100
        img = np.ones((h, w, 3), dtype=np.float32) * 0.95
        yy, xx = np.ogrid[:h, :w]
        circle = ((yy - 50) ** 2 + (xx - 50) ** 2) <= 25**2
        img[circle] = np.array([0.85, 0.15, 0.10], dtype=np.float32)

        mask = segment_fruit_foreground(img)
        iou = np.logical_and(mask, circle).sum() / np.logical_or(mask, circle).sum()
        self.assertGreater(iou, 0.85)
        self.assertGreater(otsu_threshold(img[..., 0]), 0.1)


if __name__ == "__main__":
    unittest.main()

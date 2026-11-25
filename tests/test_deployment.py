"""Pruebas de la Fase 6: extractor de 31 características e inferencia de la interfaz."""

import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset_loader import FEATURE_NAMES, compute_feature_vector, prepare_rgb_array
from src.models.predictor import MODELS, load_model, predict, size_category

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "data" / "custom_group_photos"


def synthetic_fruit() -> Image.Image:
    """Fondo gris claro con un círculo rojo de radio 0.25 * lado (d_norm esperado ~0.5)."""
    h = w = 256
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= (0.25 * h) ** 2] = (200, 30, 25)
    return Image.fromarray(img)


class TestDeployment(unittest.TestCase):
    def test_feature_order_matches_checkpoints(self) -> None:
        meta = json.loads((ROOT / "experiments/results/fase5_evaluacion_metricas.json").read_text(encoding="utf-8"))
        self.assertEqual(FEATURE_NAMES, meta["feature_names"])
        for checkpoint in MODELS.values():
            self.assertEqual(list(load_model(checkpoint).feature_names_in_), FEATURE_NAMES)

    def test_d_norm_of_synthetic_circle(self) -> None:
        feats, mask = compute_feature_vector(prepare_rgb_array(synthetic_fruit()))
        self.assertEqual(list(feats), FEATURE_NAMES)
        self.assertEqual(mask.shape, (128, 128))
        self.assertAlmostEqual(feats["d_norm"], 0.5, delta=0.03)   # diámetro / lado = 0.5

    def test_size_category_thresholds(self) -> None:
        self.assertEqual(size_category(0.30, 0.4, 0.5), "Small")
        self.assertEqual(size_category(0.40, 0.4, 0.5), "Medium")
        self.assertEqual(size_category(0.50, 0.4, 0.5), "Large")

    def test_predict_returns_valid_distribution(self) -> None:
        result = predict(synthetic_fruit(), MODELS["MLP (ganador Fase 5)"], "Apple")
        self.assertIn(result.quality, {"Good", "Mixed", "Bad"})
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
        self.assertIn(result.size, {"Small", "Medium", "Large"})

    @unittest.skipUnless((PHOTOS / "apple_bad_025.jpg").is_file(), "fotos del grupo no disponibles")
    def test_group_photo_regression(self) -> None:
        # Caso verificado manualmente: manzana con manchas -> Bad con alta confianza.
        with Image.open(PHOTOS / "apple_bad_025.jpg") as img:
            result = predict(img, MODELS["MLP (ganador Fase 5)"], "Apple")
        self.assertEqual(result.quality, "Bad")
        self.assertGreater(result.probabilities["Bad"], 0.9)


if __name__ == "__main__":
    unittest.main()

"""Smoke tests that exercise model, catalog, and inference end-to-end.

Run:
    python -m pytest tests/ -v
    # or, with no pytest installed:
    python tests/test_smoke.py
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import catalog
from src.config import MODEL_PATH, NUM_CLASSES
from src.inference import Predictor
from src.model import ProductCNN
from src.seed_catalog import seed


def test_model_forward_shape():
    model = ProductCNN()
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_catalog_seed_roundtrip(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setattr(catalog, "DB_PATH", db)
    catalog.init_db(db)
    catalog.upsert_product("TEST-0001", 0, "Test Tee", "Tops", 9.99, "test", db_path=db)
    row = catalog.get_by_class_id(0, db_path=db)
    assert row is not None
    assert row["sku"] == "TEST-0001"
    assert row["price"] == 9.99


def test_predictor_runs(tmp_path):
    """Save a randomly initialised model and run inference against it."""
    model = ProductCNN()
    weights = tmp_path / "tmp_model.pth"
    torch.save(model.state_dict(), weights)

    predictor = Predictor(weights_path=weights, device="cpu")
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    result = predictor.predict_pil(img)
    assert "class_id" in result and 0 <= result["class_id"] < NUM_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probs"]) == NUM_CLASSES


if __name__ == "__main__":
    # Minimal runner without pytest
    print("test_model_forward_shape...", end=" ")
    test_model_forward_shape()
    print("ok")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        print("test_predictor_runs...", end=" ")
        test_predictor_runs(d)
        print("ok")

    print("All smoke tests passed.")

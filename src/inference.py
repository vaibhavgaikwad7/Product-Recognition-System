"""Single-image inference."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import CLASS_NAMES, INPUT_SIZE, MODEL_PATH
from .model import ProductCNN, load_model

_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


class Predictor:
    """Wraps a ProductCNN with preprocessing and a clean predict() interface."""

    def __init__(self, weights_path=MODEL_PATH, device: str = "cpu"):
        self.device = device
        self.model: ProductCNN = load_model(weights_path, device=device)

    def _preprocess_pil(self, img: Image.Image) -> torch.Tensor:
        return _TRANSFORM(img).unsqueeze(0).to(self.device)

    def _preprocess_numpy_bgr(self, frame: np.ndarray) -> torch.Tensor:
        """OpenCV gives BGR; convert to PIL via RGB."""
        rgb = frame[:, :, ::-1] if frame.ndim == 3 else frame
        img = Image.fromarray(rgb)
        return self._preprocess_pil(img)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> dict:
        x = self._preprocess_pil(img)
        return self._infer(x)

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> dict:
        x = self._preprocess_numpy_bgr(frame)
        return self._infer(x)

    def _infer(self, x: torch.Tensor) -> dict:
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        class_id = int(np.argmax(probs))
        return {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": float(probs[class_id]),
            "probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }


_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Lazy-initialised singleton for use in the API process."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor

"""Project-wide configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CATALOG_DIR = ROOT / "catalog"
DATA_DIR = ROOT / "data"

MODEL_PATH = MODELS_DIR / "product_classifier.pth"
DB_PATH = CATALOG_DIR / "products.db"

# Fashion-MNIST class index -> human-readable label
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 28  # Fashion-MNIST is 28x28 grayscale

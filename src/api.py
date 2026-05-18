"""FastAPI service exposing prediction and catalog endpoints.

Endpoints:
    GET  /health
    GET  /products
    GET  /products/{sku}
    POST /predict/image     (multipart file: image)
    POST /predict/video     (multipart file: video, optional ?num_frames=)

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from . import catalog
from .config import MODEL_PATH
from .inference import get_predictor
from .video_inference import predict_video


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise DB and warm up the model on startup
    catalog.init_db()
    if not Path(MODEL_PATH).exists():
        print(f"WARNING: model weights not found at {MODEL_PATH}. "
              "Run `python -m src.train` first.")
    else:
        get_predictor()  # eager load
    yield


app = FastAPI(title="Product Recognition API", version="1.0.0", lifespan=lifespan)


def _attach_catalog(class_id: int) -> dict | None:
    return catalog.get_by_class_id(class_id)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": Path(MODEL_PATH).exists()}


@app.get("/products")
def list_products_endpoint() -> dict:
    return {"products": catalog.list_products()}


@app.get("/products/{sku}")
def get_product_endpoint(sku: str) -> dict:
    product = catalog.get_by_sku(sku)
    if product is None:
        raise HTTPException(status_code=404, detail=f"SKU {sku} not found")
    return product


@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    predictor = get_predictor()
    result = predictor.predict_pil(img)
    product = _attach_catalog(result["class_id"])
    return {"prediction": result, "product": product}


@app.post("/predict/video")
async def predict_video_endpoint(
    file: UploadFile = File(...),
    num_frames: int = Query(8, ge=1, le=64),
) -> dict:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # OpenCV needs a real file path, so write the upload to a tempfile
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_video(tmp_path, num_frames=num_frames)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    product = _attach_catalog(result["aggregate"]["class_id"])
    return {**result, "product": product}

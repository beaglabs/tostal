"""Temporal activity: ML model inference."""

import io
import numpy as np
from temporalio import activity
from PIL import Image


@activity.defn
async def run_inference(params: dict) -> dict:
    task = params.get("task", params.get("model_type", "classify"))
    data = params.get("data", {})
    parameters = params.get("parameters", {})

    if task in ("classify", "facies-map", "facies"):
        return _run_facies_classification(data, parameters)
    elif task in ("krige", "kriging", "krig"):
        return _run_kriging(data, parameters)
    elif task in ("segment", "litho", "segmentation"):
        return _run_segmentation(data, parameters)
    else:
        return {"error": f"Unknown task type: {task}"}


def _run_facies_classification(data: dict, parameters: dict) -> dict:
    from app.services.classifier_model import classify_facies

    curves = data.get("curves", data.get("arrays", data))
    if isinstance(curves, dict):
        curves = curves.get("data", curves)
    curves = np.asarray(curves, dtype=np.float32)

    result = classify_facies(curves)

    return {
        "data": result["facies_ids"],
        "shape": [len(result["facies_ids"])],
        "classes": result["classes"],
        "confidence": float(np.mean(result["confidence"])) if result["confidence"] else 0.0,
        "facies": result["facies"],
    }


def _run_kriging(data: dict, parameters: dict) -> dict:
    try:
        from app.services.kriging_model import krige_interpolate
        result = krige_interpolate(data, parameters)
    except ImportError:
        from app.services.kriging_model import nearest_neighbor_interpolate
        result = nearest_neighbor_interpolate(data, parameters.get("grid", {}))

    return {
        "data": result["grid"],
        "shape": result["grid_shape"],
        "classes": [],
        "confidence": 0.0,
        "grid_x": result["grid_x"],
        "grid_y": result["grid_y"],
    }


def _run_segmentation(data: dict, parameters: dict) -> dict:
    from app.services.segmentor_model import segment_lithology

    image_data = data.get("image", data.get("data", data))
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    elif isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data.astype(np.uint8)).convert("RGB")
    elif isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = Image.new("RGB", (512, 512))

    result = segment_lithology(image)

    return {
        "data": result["segmentation"],
        "shape": result["shape"],
        "classes": result["classes"],
        "confidence": 0.0,
    }
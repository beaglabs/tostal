"""XGBoost facies classifier inference service.

Loads model from container-models store and predicts lithology/facies
class per depth point from well log curves.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb


_MODEL = None
_CLASSES = None
_CURVE_NAMES = None

FACIES_CLASSES = [
    "Sandstone", "Shale", "Marl", "Limestone", "Dolostone",
    "Siltstone", "Chalk", "Volcanic", "Coal", "Conglomerate",
    "Anhydrite", "Salt",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "training" / "models" / "xgboost_facies.json"


def _load_model(model_path: Optional[str] = None):
    global _MODEL, _CLASSES, _CURVE_NAMES
    if _MODEL is not None:
        return

    path = model_path or os.environ.get("XGBOOST_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_xgboost_facies.py first.")

    _MODEL = xgb.XGBClassifier()
    _MODEL.load_model(path)

    meta_path = Path(path).parent / "xgboost_facies_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            _CLASSES = meta.get("classes", FACIES_CLASSES)
            _CURVE_NAMES = meta.get("curve_names", ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"])
    else:
        _CLASSES = FACIES_CLASSES
        _CURVE_NAMES = ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]


def classify_facies(
    curves: np.ndarray,
    model_path: Optional[str] = None,
) -> dict:
    """Classify facies from well log curves.

    Args:
        curves: numpy array of shape (n_curves, n_depth) or (n_depth, n_curves)
                Typically 6 curves: GR, RT, RHOB, NPHI, DT, CALI.

    Returns:
        dict with:
            - facies: list of class names per depth
            - facies_ids: list of integer class ids per depth
            - confidence: list of prediction confidence per depth
            - classes: list of all possible class names
    """
    _load_model(model_path)

    curves = np.asarray(curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.shape[0] != len(_CURVE_NAMES) and curves.shape[1] == len(_CURVE_NAMES):
        curves = curves.T

    n_curves, n_depth = curves.shape
    if n_curves < len(_CURVE_NAMES):
        padded = np.zeros((len(_CURVE_NAMES), n_depth), dtype=np.float32)
        padded[:n_curves] = curves
        curves = padded

    curves = (curves - np.nanmean(curves, axis=1, keepdims=True)) / (np.nanstd(curves, axis=1, keepdims=True) + 1e-8)
    curves = np.nan_to_num(curves, 0)

    X = curves.T
    pred_ids = _MODEL.predict(X).astype(int)
    proba = _MODEL.predict_proba(X)

    confidence = []
    facies_names = []
    for i, pid in enumerate(pred_ids):
        confidence.append(float(proba[i][pid]))
        facies_names.append(_CLASSES[pid] if pid < len(_CLASSES) else "unknown")

    return {
        "facies": facies_names,
        "facies_ids": pred_ids.tolist(),
        "confidence": confidence,
        "classes": _CLASSES,
    }
"""XGBoost facies classifier inference service with spatial features.

Loads model from container-models store and predicts lithology/facies
class per depth point from well log curves. Computes spatial features
(normalized depth, rolling stats, neighbors) at inference time.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

_MODEL = None
_CLASSES = None
_FEATURE_NAMES = None
_WINDOW = 5

FACIES_CLASSES = [
    "Sandstone", "Sandstone/Shale", "Shale", "Marl", "Dolomite",
    "Limestone", "Chalk", "Halite", "Anhydrite", "Tuff",
    "Coal", "Basement",
]

CURVE_NAMES = ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "training" / "models" / "xgboost_facies.json"
)


def _build_spatial_features(curves, n_depth, window=5):
    """Mirrors data.normalize_force2020.build_spatial_features for API use."""
    n_curves = curves.shape[0]
    half = window // 2
    features = []
    for d in range(n_depth):
        f = []
        for c in range(n_curves):
            f.append(float(curves[c, d]))
        f.append(d / max(n_depth - 1, 1))
        lo, hi = max(0, d - half), min(n_depth, d + half + 1)
        for c in range(n_curves):
            f.append(float(np.nanmean(curves[c, lo:hi])))
        for c in range(n_curves):
            f.append(float(np.nanstd(curves[c, lo:hi])))
        for c in range(n_curves):
            f.append(float(curves[c, max(0, d - 1)]))
        for c in range(n_curves):
            f.append(float(curves[c, min(n_depth - 1, d + 1)]))
        features.append(f)
    return np.array(features, dtype=np.float32)


def _load_model(model_path: Optional[str] = None):
    global _MODEL, _CLASSES, _WINDOW
    if _MODEL is not None:
        return

    path = model_path or os.environ.get("XGBOOST_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_xgboost_facies.py first."
        )

    _MODEL = xgb.XGBClassifier()
    _MODEL.load_model(path)

    meta_path = Path(path).parent / "xgboost_facies_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            _CLASSES = meta.get("classes", FACIES_CLASSES)
            _WINDOW = meta.get("window", 5)
    else:
        _CLASSES = FACIES_CLASSES
        _WINDOW = 5


def classify_facies(
    curves: np.ndarray,
    model_path: Optional[str] = None,
) -> dict:
    """Classify facies from well log curves with spatial context.

    Args:
        curves: numpy array of shape (n_curves, n_depth) or (n_depth, n_curves).
                Typically 6 curves: GR, RT, RHOB, NPHI, DT, CALI.

    Returns:
        dict with facies, facies_ids, confidence, classes.
    """
    _load_model(model_path)

    curves = np.asarray(curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.shape[0] != len(CURVE_NAMES) and curves.shape[1] == len(CURVE_NAMES):
        curves = curves.T

    n_curves, n_depth = curves.shape
    if n_curves < len(CURVE_NAMES):
        padded = np.zeros((len(CURVE_NAMES), n_depth), dtype=np.float32)
        padded[:n_curves] = curves
        curves = padded
        n_curves = len(CURVE_NAMES)

    curves = (curves - np.nanmean(curves, axis=1, keepdims=True)) / (
        np.nanstd(curves, axis=1, keepdims=True) + 1e-8
    )
    curves = np.nan_to_num(curves, 0)

    X = _build_spatial_features(curves, n_depth, window=_WINDOW)

    pred_ids = _MODEL.predict(X).astype(int)
    proba = _MODEL.predict_proba(X)

    confidence, facies_names = [], []
    for i, pid in enumerate(pred_ids):
        confidence.append(float(proba[i][pid]))
        facies_names.append(_CLASSES[pid] if pid < len(_CLASSES) else "unknown")

    return {
        "facies": facies_names,
        "facies_ids": pred_ids.tolist(),
        "confidence": confidence,
        "classes": _CLASSES,
    }
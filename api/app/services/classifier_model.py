"""XGBoost facies classifier inference service with spatial features.

Loads model from container-models store and predicts lithology/facies
class per depth point from well log curves. Computes spatial features
(normalized depth, rolling stats, neighbors) at inference time.

Supports:
  - facies-map: lithology classification with per-depth confidence
  - synthetic-core: lithology + porosity + permeability prediction with flags
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
            f"Model not found at {path}."
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


def _compute_confidence(proba, pred_ids):
    confidence = []
    for i, pid in enumerate(pred_ids):
        confidence.append(float(proba[i][pid]))
    return confidence


def _compute_flags(confidence, curves):
    n_depth = len(confidence)
    flags = []
    low_conf_starts = []

    caliper = curves[5] if curves.shape[0] > 5 else np.zeros(n_depth)
    caliper_std = np.nanstd(caliper) if np.any(~np.isnan(caliper)) else 1.0
    caliper_mean = np.nanmean(caliper) if np.any(~np.isnan(caliper)) else 0.0

    low_conf_zones = []
    in_low = False
    zone_start = 0
    for d in range(n_depth):
        is_low = confidence[d] < 0.6
        is_gap = np.all(np.abs(curves[:, d]) < 0.01) if curves.shape[0] > 1 else False
        is_washed = (caliper[d] > caliper_mean + 2 * caliper_std) if not np.isnan(caliper[d]) else False

        if is_low or is_gap:
            if not in_low:
                zone_start = d
                in_low = True
        elif is_washed:
            flags.append({
                "from": d,
                "to": d + 1,
                "reason": "borehole_washout",
            })
        else:
            if in_low:
                low_conf_zones.append({
                    "from": zone_start,
                    "to": d,
                    "reason": "washout" if any(
                        caliper[zone_start:d] > caliper_mean + 2 * caliper_std
                    ) else "low_confidence",
                })
                in_low = False

    if in_low:
        low_conf_zones.append({
            "from": zone_start,
            "to": n_depth,
            "reason": "low_confidence",
        })

    high_uncertainty_continuous = []
    run_start = None
    for d in range(n_depth):
        if confidence[d] < 0.7:
            if run_start is None:
                run_start = d
        else:
            if run_start is not None and (d - run_start) >= 10:
                high_uncertainty_continuous.append({
                    "from": run_start,
                    "to": d,
                })
            run_start = None
    if run_start is not None and (n_depth - run_start) >= 10:
        high_uncertainty_continuous.append({
            "from": run_start,
            "to": n_depth,
        })

    return {
        "low_confidence_zones": low_conf_zones,
        "borehole_washout_zones": [f for f in flags if f["reason"] == "borehole_washout"],
        "suggested_coring_intervals": high_uncertainty_continuous[:5],
    }


def classify_facies(
    curves: np.ndarray,
    model_path: Optional[str] = None,
) -> dict:
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

    curves_norm = (curves - np.nanmean(curves, axis=1, keepdims=True)) / (
        np.nanstd(curves, axis=1, keepdims=True) + 1e-8
    )
    curves_norm = np.nan_to_num(curves_norm, 0)

    X = _build_spatial_features(curves_norm, n_depth, window=_WINDOW)

    pred_ids = _MODEL.predict(X).astype(int)
    proba = _MODEL.predict_proba(X)

    confidence = _compute_confidence(proba, pred_ids)
    facies_names = [
        _CLASSES[pid] if pid < len(_CLASSES) else "unknown"
        for pid in pred_ids
    ]
    flags = _compute_flags(confidence, curves)

    return {
        "facies": facies_names,
        "facies_ids": pred_ids.tolist(),
        "confidence": confidence,
        "classes": _CLASSES,
        "per_depth_confidence": confidence,
        "flags": flags,
    }


def classify_synthetic_core(
    curves: np.ndarray,
    model_path: Optional[str] = None,
) -> dict:
    """Classify facies with synthetic core prediction outputs.

    Predicts lithology, porosity, permeability, and grain density from
    well log curves. Currently uses facies classifier for lithology and
    estimates porosity/permeability from empirical log transforms.
    Full synthetic core model support is planned for future training.

    Args:
        curves: numpy array of shape (n_curves, n_depth) or (n_depth, n_curves)
        model_path: optional path to model file

    Returns:
        dict with outputs (lithology, porosity, permeability, grain_density,
            confidence) and flags
    """
    result = classify_facies(curves, model_path)

    curves = np.asarray(curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.shape[0] != len(CURVE_NAMES) and curves.shape[1] == len(CURVE_NAMES):
        curves = curves.T

    n_curves, n_depth = curves.shape

    rhob_idx = 2 if n_curves > 2 else None
    nphi_idx = 3 if n_curves > 3 else None
    dt_idx = 4 if n_curves > 4 else None

    porosity = np.full(n_depth, np.nan, dtype=np.float32)
    grain_density = np.full(n_depth, np.nan, dtype=np.float32)
    permeability = np.full(n_depth, np.nan, dtype=np.float32)

    if rhob_idx is not None and nphi_idx is not None:
        rhob = curves[rhob_idx]
        nphi = curves[nphi_idx]
        rhob_matrix = 2.65
        rhob_fluid = 1.0
        porosity = (rhob_matrix - rhob) / (rhob_matrix - rhob_fluid)
        porosity = np.clip(porosity, 0.0, 0.45)
        porosity[np.isnan(rhob) | np.isnan(nphi)] = np.nan

        grain_density = (rhob - rhob_fluid * porosity) / (1.0 - porosity + 1e-8)
        grain_density = np.clip(grain_density, 2.0, 3.0)

    if porosity is not None:
        perm_from_porosity = 10 ** (3.5 * np.log10(np.maximum(porosity, 0.001)) + 2.0)
        permeability = np.clip(perm_from_porosity, 0.001, 10000.0)
        permeability[porosity < 0.01] = 0.001

    result["outputs"] = {
        "lithology": {
            "classes": result["classes"],
            "values": result["facies_ids"],
        },
        "porosity": {
            "units": "fraction",
            "range": [float(np.nanmin(porosity)) if np.any(~np.isnan(porosity)) else 0.0,
                      float(np.nanmax(porosity)) if np.any(~np.isnan(porosity)) else 0.45],
            "values": np.where(np.isnan(porosity), -1.0, porosity).tolist(),
        },
        "permeability": {
            "units": "mD",
            "range": [float(np.nanmin(permeability)) if np.any(~np.isnan(permeability)) else 0.001,
                      float(np.nanmax(permeability)) if np.any(~np.isnan(permeability)) else 10000.0],
            "values": np.where(np.isnan(permeability), -1.0, permeability).tolist(),
        },
        "grain_density": {
            "units": "g/cc",
            "range": [float(np.nanmin(grain_density)) if np.any(~np.isnan(grain_density)) else 2.0,
                      float(np.nanmax(grain_density)) if np.any(~np.isnan(grain_density)) else 3.0],
            "values": np.where(np.isnan(grain_density), -1.0, grain_density).tolist(),
        },
        "confidence": {
            "range": [0.0, 1.0],
            "values": result["per_depth_confidence"],
        },
    }

    return result
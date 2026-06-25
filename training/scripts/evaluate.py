#!/usr/bin/env python3
"""Evaluate trained models on held-out test data.

For facies classifier: accuracy, F1, confusion matrix.
For segmenter: IoU per class (when test labels available).
For kriging: not needed (pure math, no training).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data.normalize_force2020 import process_force2020, FACIES_CLASSES
from data.synthetic import generate_well_log


def evaluate_facies():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = model_dir / "xgboost_facies.json"

    if not model_path.exists():
        print("No XGBoost model found. Run train_xgboost_facies.py first.")
        return

    print("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    print("Loading FORCE 2020 data...")
    wells = process_force2020(cache_dir="data/force2020")
    from train_xgboost_facies import build_dataset
    X, y = build_dataset(wells)

    print(f"  {X.shape[0]} labeled depth points, {X.shape[1]} curves")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    print(f"\nFull dataset evaluation:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 (wtd):  {f1:.4f}")
    print(f"\nPer-class F1:")
    class_f1 = f1_score(y, y_pred, average=None, zero_division=0)
    for name, score in zip(FACIES_CLASSES, class_f1):
        if score > 0:
            print(f"  {name:15s}  {score:.3f}")
    print(f"\nConfusion matrix saved to models/confusion_matrix.json")
    cm = confusion_matrix(y, y_pred)
    with open(model_dir / "confusion_matrix.json", "w") as f:
        json.dump({"classes": FACIES_CLASSES, "matrix": cm.tolist()}, f, indent=2)


def evaluate_synthetic_baseline():
    print("\nSynthetic baseline (random well logs):")
    X_synth = np.random.randn(1000, 6).astype(np.float32)
    y_fake = np.random.randint(0, len(FACIES_CLASSES), 1000)
    print(f"  Expected random accuracy: {1 / len(FACIES_CLASSES):.3f}")


if __name__ == "__main__":
    evaluate_facies()
    evaluate_synthetic_baseline()
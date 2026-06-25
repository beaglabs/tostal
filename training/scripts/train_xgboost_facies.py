#!/usr/bin/env python3
"""Train XGBoost facies classifier on FORCE 2020 dataset.

Produces: models/xgboost_facies.json
Runtime: ~1 minute on CPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data.normalize_force2020 import process_force2020, FACIES_CLASSES


def build_dataset(wells, n_depth=512, n_curves=6):
    X_list, y_list = [], []
    for name, data in wells.items():
        curves = data["well_log"].numpy()
        facies = data.get("facies")
        if facies is None:
            continue
        facies = facies.numpy()
        if curves.shape != (n_curves, n_depth):
            continue
        for d in range(n_depth):
            label = facies[d]
            if label < 0:
                continue
            X_list.append(curves[:, d])
            y_list.append(label)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def main():
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FORCE 2020 data...")
    wells = process_force2020(cache_dir="data/force2020")
    X, y = build_dataset(wells)
    print(f"  {X.shape[0]} labeled depth points, {X.shape[1]} curves")
    print(f"  Classes: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(FACIES_CLASSES),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nTest accuracy:  {acc:.4f}")
    print(f"Test F1 (wtd):  {f1:.4f}")
    print(f"\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=FACIES_CLASSES, zero_division=0))

    model_path = out_dir / "xgboost_facies.json"
    model.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")

    meta_path = out_dir / "xgboost_facies_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "classes": FACIES_CLASSES,
            "n_curves": X.shape[1],
            "curve_names": ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"],
            "accuracy": float(acc),
            "f1_weighted": float(f1),
        }, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
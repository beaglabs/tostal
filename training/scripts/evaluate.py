"""Evaluate trained models on held-out test data.

Facies classifier (XGBoost + spatial features): accuracy, F1, confusion matrix.
DINOv2 classifier: top-1 / top-3 accuracy on DCID subset.
Taranaki classifier: cross-basin evaluation.
Kriging: not needed (pure math, no training).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import json
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from data.normalize_force2020 import process_force2020, FACIES_CLASSES, build_spatial_features

CURVE_NAMES = ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]


def _build_dataset(wells, n_curves=6, n_depth=512):
    """Build X, y arrays from wells dict."""
    X_list, y_list = [], []
    for name, data in wells.items():
        curves = data["well_log"].numpy()
        facies = data.get("facies")
        if facies is None:
            continue
        facies = facies.numpy()
        if curves.shape != (n_curves, n_depth):
            continue
        X_spatial = build_spatial_features(curves, n_depth)
        for d in range(n_depth):
            label = facies[d]
            if label < 0:
                continue
            X_list.append(X_spatial[d])
            y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def _print_metrics(y_true, y_pred, title=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{title}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 (wtd):  {f1:.4f}")
    print(f"\n  Per-class F1:")
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for name, score in zip(FACIES_CLASSES, class_f1):
        if score > 0:
            print(f"    {name:15s}  {score:.3f}")
    return acc, f1


def evaluate_facies():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = model_dir / "xgboost_facies.json"

    if not model_path.exists():
        print("No XGBoost model found.")
        return

    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    meta_path = model_dir / "xgboost_facies_meta.json"
    window = 5
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            window = meta.get("window", 5)

    print("Loading FORCE 2020 data...")
    wells = process_force2020(cache_dir="data/force2020")
    X, y = _build_dataset(wells)
    print(f"  {X.shape[0]} labeled depth points, {X.shape[1]} features")

    y_pred = model.predict(X)
    _print_metrics(y, y_pred, "FORCE 2020 (in-domain)")

    cm = confusion_matrix(y, y_pred)
    with open(model_dir / "confusion_matrix.json", "w") as f:
        json.dump({"classes": FACIES_CLASSES, "matrix": cm.tolist()}, f, indent=2)
    print(f"  Confusion matrix saved to models/confusion_matrix.json")


def evaluate_taranaki_cross_basin():
    """Evaluate on Taranaki data if available (cross-basin test)."""
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = model_dir / "xgboost_facies.json"
    if not model_path.exists():
        return

    try:
        from data.normalize_taranaki import process_taranaki
        wells = process_taranaki(cache_dir="data/taranaki")
    except Exception as e:
        print(f"\nSkipping Taranaki evaluation: {e}")
        return

    if not wells:
        return

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    X, y = _build_dataset(wells)
    if len(X) == 0:
        print(f"\nTaranaki: no labeled depth points")
        return

    print(f"\nTaranaki cross-basin evaluation:")
    print(f"  {X.shape[0]} labeled depth points, {X.shape[1]} features")

    y_pred = model.predict(X)
    _print_metrics(y, y_pred, "Taranaki Basin (cross-basin)")


def evaluate_dino():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = model_dir / "dino_classifier.pt"

    if not model_path.exists():
        print("\nNo DINOv2 classifier found.")
        return

    print("\nLoading DINOv2 classifier...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    num_classes = checkpoint["num_classes"]
    image_size = checkpoint.get("image_size", 448)

    from train_dino_classifier import ClassifierHead, _pil_from_item, extract_labels
    try:
        from datasets import load_dataset
    except ImportError:
        print("  pip install datasets required")
        return

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()

    head = ClassifierHead(in_dim=384, num_classes=num_classes)
    head.load_state_dict(checkpoint["head_state"])
    head.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = load_dataset("168sir/drill-core-image-dataset", split="train")
    n_subset = min(500, len(ds))
    indices = np.random.RandomState(99).choice(len(ds), n_subset, replace=False)
    y_true = extract_labels(ds, indices, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    head = head.to(device)

    correct = 0
    top3_correct = 0
    with torch.no_grad():
        for idx in indices:
            img = _pil_from_item(ds[idx])
            tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(tensor)
            logits = head(feat)
            preds = logits.argmax(dim=1).cpu().item()
            top3 = logits.topk(min(3, num_classes), dim=1).indices.cpu()
            label = y_true[indices.tolist().index(idx)]
            if preds == label:
                correct += 1
            if label in top3[0]:
                top3_correct += 1

    print(f"  DINOv2 top-1 accuracy: {correct / n_subset:.4f} ({n_subset} samples)")
    print(f"  DINOv2 top-3 accuracy: {top3_correct / n_subset:.4f}")


def evaluate_synthetic_baseline():
    print("\nSynthetic baseline (random):")
    expected = 1 / len(FACIES_CLASSES)
    print(f"  Expected random accuracy: {expected:.3f}")


if __name__ == "__main__":
    evaluate_facies()
    evaluate_taranaki_cross_basin()
    evaluate_dino()
    evaluate_synthetic_baseline()
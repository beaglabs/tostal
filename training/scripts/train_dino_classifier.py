#!/usr/bin/env python3
"""Train DINOv2 lithology classifier on DCID drill core images.

Uses DINOv2 vit_s14 (22M params) frozen backbone + lightweight classifier head.
Trains on a 1000-image subset for speed. Full dataset optional.

Produces: models/dino_classifier.pt
Runtime: ~2 min on GPU (T4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

LITHOLOGY_CLASSES = [
    "background", "sandstone", "shale", "limestone", "dolostone",
    "siltstone", "conglomerate", "volcanic", "coal", "anhydrite",
    "chert", "marl",
]
IMAGE_SIZE = 448
SUBSET_SIZE = 1000
BATCH_SIZE = 16
EPOCHS = 10


def _pil_from_item(item):
    raw = item["image"]
    if isinstance(raw, Image.Image):
        img = raw
    elif isinstance(raw, bytes):
        img = Image.open(io.BytesIO(raw))
    else:
        img = Image.open(io.BytesIO(raw)) if hasattr(raw, "read") else raw
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def discover_labels(dataset):
    features = dataset.features if hasattr(dataset, "features") else {}
    if "label" in features:
        names = features["label"].names if hasattr(features["label"], "names") else None
        if names:
            return {i: name for i, name in enumerate(names)}, len(names)

    sample = dataset[0]
    if isinstance(sample, dict) and "label" in sample:
        return None, None

    return None, None


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=384, hidden=256, num_classes=12, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def extract_labels(dataset, indices, num_classes):
    labels = []
    for idx in indices:
        item = dataset[idx]
        if "label" in item and item["label"] is not None:
            lbl = item["label"]
            labels.append(int(lbl) if not isinstance(lbl, int) else lbl)
        else:
            labels.append(0)
    return torch.tensor(labels, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets required")
        return

    print(f"Device: {device}")
    print("Loading DINOv2 small backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading DCID dataset...")
    ds = load_dataset("168sir/drill-core-image-dataset", split="train")
    label_map, num_classes = discover_labels(ds)

    if num_classes is None:
        num_classes = len(LITHOLOGY_CLASSES)
        print(f"  No labels found — using {num_classes} proxy classes (weak supervision)")
    else:
        print(f"  Found {num_classes} classes: {list(label_map.values())[:5]}...")
        if num_classes > 50:
            num_classes = len(LITHOLOGY_CLASSES)
            print(f"  Too many classes, capping at {num_classes}")

    n_total = len(ds)
    n_subset = min(SUBSET_SIZE, n_total)
    indices = np.random.RandomState(42).choice(n_total, n_subset, replace=False)
    n_train = int(n_subset * 0.8)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
    print(f"  {len(train_idx)} train / {len(val_idx)} val images (subset of {n_total})")

    y_all = extract_labels(ds, indices, num_classes)

    print("Extracting DINOv2 features...")
    features = torch.zeros(n_subset, 384, dtype=torch.float32)
    backbone = backbone.to(device)
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Features")):
            img = _pil_from_item(ds[idx])
            tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(tensor)
            features[i] = feat.cpu()

    X_train = features[:n_train]
    X_val = features[n_train:]
    y_train = y_all[:n_train]
    y_val = y_all[n_train:]

    print("Training classifier head...")
    head = ClassifierHead(in_dim=384, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        head.train()
        total_loss = 0
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            xb = X_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == y_val).float().mean().item()
        best_acc = max(best_acc, val_acc)
        print(f"  Epoch {epoch + 1:2d}/{EPOCHS}  loss={total_loss / (n_train // BATCH_SIZE):.4f}  val_acc={val_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}")

    model_path = out_dir / "dino_classifier.pt"
    torch.save({
        "head_state": head.state_dict(),
        "backbone": "dinov2_vits14",
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "feature_dim": 384,
    }, model_path)
    print(f"Model saved to {model_path}")

    meta_path = out_dir / "dino_classifier_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "classes": LITHOLOGY_CLASSES[:num_classes],
            "architecture": "dinov2_vits14 + classifier_head",
            "image_size": IMAGE_SIZE,
            "val_accuracy": float(best_acc),
        }, f, indent=2)


if __name__ == "__main__":
    main()
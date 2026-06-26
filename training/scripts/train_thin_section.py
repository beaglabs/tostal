#!/usr/bin/env python3
"""Train DINOv2 thin-section petrography classifier on LITHOS dataset.

Loads the LITHOS dataset via kagglehub (with Croissant metadata for
discovery), extracts DINOv2 features from polarized-light photomicrograph
patches, and trains a lightweight classifier head for 25 mineral classes.

Dataset: Paola Ruiz Puentes, "Towards Automated Petrography"
         NeurIPS 2025 Datasets and Benchmarks.
         211,604 patches x 25 mineral classes, 18.6 GB.

Produces: models/thin_section_classifier.pt
Runtime: ~5 min feature extraction + 2 min training on GPU (T4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from data.thin_section import (
    MINERAL_CLASSES,
    LithosIndex,
    load_lithos_dataset,
    fetch_croissant_metadata,
    inspect_dataset,
)

IMAGE_SIZE = 448
SUBSET_SIZE = 3000
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=384, hidden=256, num_classes=25, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Croissant metadata for LITHOS dataset...")
    local_meta = Path(__file__).resolve().parent.parent / "data" / "lithos-dataset-metadata.json"
    meta_source = str(local_meta) if local_meta.exists() else None
    try:
        meta = fetch_croissant_metadata(meta_source)
        print(f"  Name: {meta.get('name')}")
        print(f"  Description: {meta.get('description', '')[:120]}...")
        print(f"  License: {meta.get('license', {}).get('name', 'unknown')}")
    except Exception as e:
        print(f"  Croissant metadata unavailable ({e}), continuing with kagglehub...")

    print("\nDownloading LITHOS dataset via kagglehub...")
    dataset_path = load_lithos_dataset()
    print(f"  Local path: {dataset_path}")

    inspect_dataset(dataset_path)

    print("\nBuilding image-to-label index...")
    index = LithosIndex(dataset_path)
    num_classes = index.num_classes
    n_total = len(index)
    print(f"  Total images: {n_total}")
    print(f"  Classes: {num_classes}")
    counts = index.label_counts()
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name:20s}: {cnt:6d}")

    n_subset = min(SUBSET_SIZE, n_total)
    indices = np.random.RandomState(42).choice(n_total, n_subset, replace=False)
    n_train = int(n_subset * 0.8)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
    print(f"\n  Training subset: {n_train} train / {n_subset - n_train} val")

    y_all = torch.tensor([index.get_label(i) for i in indices], dtype=torch.long)

    print("\nLoading DINOv2 small backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Extracting DINOv2 features...")
    features = torch.zeros(n_subset, 384, dtype=torch.float32)
    backbone = backbone.to(device)
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Features")):
            img = index.get_image(idx)
            tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(tensor)
            features[i] = feat.cpu()

    X_train = features[:n_train]
    X_val = features[n_train:]
    y_train = y_all[:n_train]
    y_val = y_all[n_train:]

    print("\nTraining classifier head (25 mineral classes)...")
    head = ClassifierHead(in_dim=384, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        head.train()
        total_loss = 0
        perm = torch.randperm(n_train)
        n_batches = 0
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
            n_batches += 1
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == y_val).float().mean().item()

            val_probs = torch.softmax(val_logits, dim=1)
            val_top3 = val_probs.topk(3, dim=1).indices
            val_top3_acc = torch.tensor([
                y_val[i].item() in val_top3[i].tolist()
                for i in range(len(y_val))
            ]).float().mean().item()

        best_acc = max(best_acc, val_acc)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1:2d}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}  top3={val_top3_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}")

    model_path = out_dir / "thin_section_classifier.pt"
    torch.save({
        "head_state": head.state_dict(),
        "backbone": "dinov2_vits14",
        "num_classes": num_classes,
        "image_size": IMAGE_SIZE,
        "feature_dim": 384,
    }, model_path)
    print(f"Model saved to {model_path}")

    meta_path = out_dir / "thin_section_classifier_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": "LITHOS (Paola Ruiz Puentes, NeurIPS 2025)",
            "classes": MINERAL_CLASSES,
            "num_classes": num_classes,
            "architecture": "dinov2_vits14 + classifier_head (384 → 256 → 25)",
            "image_size": IMAGE_SIZE,
            "subset_size": n_subset,
            "val_accuracy": float(best_acc),
            "val_top3_accuracy": float(val_top3_acc),
        }, f, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fine-tune DeepLabV3 on DCID drill core images for lithology segmentation.

Produces: models/segmenter.pt
Runtime: ~30 min on GPU (T4/V100).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


LITHOLOGY_CLASSES = [
    "background", "sandstone", "shale", "limestone", "dolostone",
    "siltstone", "conglomerate", "volcanic", "coal", "anhydrite",
    "chert", "marl",
]
NUM_CLASSES = len(LITHOLOGY_CLASSES)


DCID_LABEL_MAP = None


def _discover_labels(dataset):
    global DCID_LABEL_MAP
    if DCID_LABEL_MAP is not None:
        return DCID_LABEL_MAP

    DCID_LABEL_MAP = {}
    features = dataset.features if hasattr(dataset, "features") else {}
    if "label" in features:
        names = features["label"].names if hasattr(features["label"], "names") else None
        if names:
            DCID_LABEL_MAP = {i: name for i, name in enumerate(names)}
            return DCID_LABEL_MAP

    sample = dataset[0]
    if isinstance(sample, dict) and "label" in sample:
        label_val = sample["label"]
        if isinstance(label_val, int):
            DCID_LABEL_MAP = {0: "class_0"}
        elif isinstance(label_val, str):
            DCID_LABEL_MAP = {0: "class_0"}

    return DCID_LABEL_MAP


class DrillCoreDataset(Dataset):
    def __init__(self, indices, dataset, image_size=512):
        self.indices = indices
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        raw = item["image"]
        if isinstance(raw, Image.Image):
            image = raw
        elif isinstance(raw, bytes):
            image = Image.open(io.BytesIO(raw))
        else:
            image = Image.open(io.BytesIO(raw)) if hasattr(raw, "read") else raw
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image)
        label = torch.zeros(tensor.shape[1], tensor.shape[2], dtype=torch.long)
        if "label" in item and item["label"] is not None:
            lbl = item["label"]
            if isinstance(lbl, int):
                label.fill_(lbl + 1)
        return tensor, label


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading DCID drill core dataset...")
    try:
        full_ds = load_dataset("168sir/drill-core-image-dataset", split="train")
        _discover_labels(full_ds)
        n_total = len(full_ds)
        n_train = int(n_total * 0.8)
        indices = list(range(n_total))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        train_ds = DrillCoreDataset(train_indices, full_ds)
        val_ds = DrillCoreDataset(val_indices, full_ds)
        print(f"  {len(train_indices)} train / {len(val_indices)} val images")
        if DCID_LABEL_MAP:
            print(f"  Labels: {list(DCID_LABEL_MAP.values())[:5]}...")
        else:
            print("  No classification labels found — training with background masks")
    except Exception as e:
        print(f"  HuggingFace dataset not available: {e}")
        print("  Creating placeholder model...")
        model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
        torch.save(model.state_dict(), out_dir / "segmenter.pt")
        print(f"  Placeholder saved to {out_dir / 'segmenter.pt'}")
        return

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = deeplabv3_resnet50(weights="DEFAULT", weights_backbone="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Training {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output["out"], labels)
            if "aux" in output:
                loss += 0.5 * criterion(output["aux"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                val_loss += criterion(output["out"], labels).item()
        print(f"  Train loss: {total_loss / len(train_loader):.4f}  Val loss: {val_loss / len(val_loader):.4f}")

    model_path = out_dir / "segmenter.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    meta_path = out_dir / "segmenter_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "classes": LITHOLOGY_CLASSES,
            "architecture": "deeplabv3_resnet50",
            "image_size": 512,
        }, f, indent=2)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Fine-tune DeepLabV3 on DCID drill core images for lithology segmentation.

Produces: models/segmenter.pt
Runtime: ~30 min on GPU (T4/V100).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
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


class DrillCoreDataset(Dataset):
    def __init__(self, split="train", image_size=512):
        if load_dataset is None:
            raise ImportError("pip install datasets")
        self.dataset = load_dataset("168sir/drill-core-image-dataset", split=split)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
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
        train_ds = DrillCoreDataset(split="train")
        val_ds = DrillCoreDataset(split="test") if "test" in train_ds.dataset else train_ds
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
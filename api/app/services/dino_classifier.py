"""DINOv2 lithology classifier inference service.

Loads DINOv2 vit_s14 backbone + classifier head from container-models store
and classifies drill core / outcrop photos into lithology categories.
"""
import io
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

_MODEL = None
_BACKBONE = None
_HEAD = None
_CLASSES = None
_IMAGE_SIZE = 448

LITHOLOGY_CLASSES = [
    "background", "sandstone", "shale", "limestone", "dolostone",
    "siltstone", "conglomerate", "volcanic", "coal", "anhydrite",
    "chert", "marl",
]

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "training" / "models" / "dino_classifier.pt"
)


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


def _load_model(model_path: Optional[str] = None):
    global _BACKBONE, _HEAD, _CLASSES, _IMAGE_SIZE
    if _BACKBONE is not None:
        return

    path = model_path or os.environ.get("DINO_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_dino_classifier.py first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _IMAGE_SIZE = checkpoint.get("image_size", 448)
    num_classes = checkpoint["num_classes"]

    _BACKBONE = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    _BACKBONE.eval()

    _HEAD = ClassifierHead(in_dim=384, num_classes=num_classes)
    _HEAD.load_state_dict(checkpoint["head_state"])
    _HEAD.eval()

    meta_path = Path(path).parent / "dino_classifier_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            _CLASSES = meta.get("classes", LITHOLOGY_CLASSES[:num_classes])
    else:
        _CLASSES = LITHOLOGY_CLASSES[:num_classes]

    if torch.cuda.is_available():
        _BACKBONE = _BACKBONE.cuda()
        _HEAD = _HEAD.cuda()


def _preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def classify_lithology(
    image: Union[Image.Image, bytes],
    model_path: Optional[str] = None,
    top_k: int = 3,
) -> dict:
    """Classify lithology from a drill core or outcrop photo.

    Args:
        image: PIL Image or raw image bytes.
        model_path: optional path to model checkpoint.
        top_k: number of top predictions to return.

    Returns:
        dict with predicted_class, confidence, top_k predictions.
    """
    _load_model(model_path)

    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image or bytes")

    device = next(_BACKBONE.parameters()).device
    tensor = _preprocess(image).to(device)

    with torch.no_grad():
        features = _BACKBONE(tensor)
        logits = _HEAD(features)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {
            "class": _CLASSES[i] if i < len(_CLASSES) else f"class_{i}",
            "confidence": float(probs[i]),
        }
        for i in top_indices
    ]

    return {
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_k": top_predictions,
        "classes": _CLASSES,
    }
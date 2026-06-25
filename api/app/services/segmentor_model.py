"""DeepLabV3 lithology segmentation inference service.

Loads model from container-models store and predicts pixel-level
lithology class from core/outcrop photos.
"""
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image


_MODEL = None
_CLASSES = None
_IMAGE_SIZE = 512

LITHOLOGY_CLASSES = [
    "background", "sandstone", "shale", "limestone", "dolostone",
    "siltstone", "conglomerate", "volcanic", "coal", "anhydrite",
    "chert", "marl",
]

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "training" / "models" / "segmenter.pt"


def _load_model(model_path: Optional[str] = None):
    global _MODEL, _CLASSES
    if _MODEL is not None:
        return

    path = model_path or os.environ.get("SEGMENTER_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    num_classes = len(LITHOLOGY_CLASSES)

    _MODEL = deeplabv3_resnet50(weights=None, num_classes=num_classes)

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        _MODEL.load_state_dict(state_dict, strict=False)

    _MODEL.eval()

    meta_path = Path(path).parent / "segmenter_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            _CLASSES = meta.get("classes", LITHOLOGY_CLASSES)
    else:
        _CLASSES = LITHOLOGY_CLASSES

    if torch.cuda.is_available():
        _MODEL = _MODEL.cuda()


def _preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def segment_lithology(
    image: Image.Image,
    model_path: Optional[str] = None,
) -> dict:
    """Segment lithology from a core/outcrop photo.

    Args:
        image: PIL Image of a drill core or outcrop photo.
        model_path: optional path to model weights.

    Returns:
        dict with:
            - segmentation: 2D list of class IDs (pixel-level)
            - classes: list of class names
            - shape: [height, width] of output mask
    """
    _load_model(model_path)

    device = next(_MODEL.parameters()).device
    tensor = _preprocess(image).to(device)

    with torch.no_grad():
        output = _MODEL(tensor)
        pred = output["out"].argmax(dim=1).squeeze(0).cpu().numpy()

    return {
        "segmentation": pred.tolist(),
        "classes": _CLASSES,
        "shape": list(pred.shape),
    }
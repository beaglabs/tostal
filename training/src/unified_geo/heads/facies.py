import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class FaciesHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_dim = config.num_heads * config.dh
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, config.facies_num_classes),
        )

    def forward(self, slot_v):
        B, H, M, Dh = slot_v.shape
        pooled = slot_v.mean(dim=2)
        flat = pooled.reshape(B, H * Dh)
        return self.classifier(flat)
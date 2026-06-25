import torch
import torch.nn as nn

from ..config import ModelConfig


class KrigingHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_dim = config.num_heads * config.dh
        self.net = nn.Sequential(
            nn.Linear(input_dim + 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, slot_v, query_points):
        B, H, M, Dh = slot_v.shape
        pooled = slot_v.mean(dim=2)
        flat = pooled.reshape(B, H * Dh)
        flat = flat.unsqueeze(1).expand(-1, query_points.shape[1], -1)
        combined = torch.cat([flat, query_points], dim=-1)
        return self.net(combined).squeeze(-1)
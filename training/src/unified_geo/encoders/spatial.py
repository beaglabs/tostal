import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, config.spatial_mlp_hidden),
            nn.GELU(),
            nn.Linear(config.spatial_mlp_hidden, config.d_model),
        )
        self.value_proj = nn.Linear(1, config.d_model)
        self.fuse = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(self, x):
        coords = x[..., :3]
        values = x[..., 3:4] if x.shape[-1] > 3 else None

        pos_enc = self.pos_mlp(coords)
        if self.value_proj is not None and values is not None:
            val_enc = self.value_proj(values)
            combined = pos_enc + val_enc
        else:
            combined = pos_enc

        return self.fuse(combined)
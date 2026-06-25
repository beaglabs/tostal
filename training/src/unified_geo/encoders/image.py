import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.image_patch_size
        self.d_model = config.d_model

        self.patch_embed = nn.Conv2d(
            config.image_channels, config.d_model,
            kernel_size=self.patch_size, stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 17, config.d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.image_vit_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.vit = nn.TransformerEncoder(encoder_layer, num_layers=config.image_vit_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), patches], dim=1)
        n = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :n, :]
        tokens = self.vit(tokens)
        return tokens
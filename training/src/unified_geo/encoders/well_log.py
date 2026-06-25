import torch
import torch.nn as nn


class WellLogEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_tokens = config.well_log_output_tokens
        self.output_dim = config.d_model

        ch = config.well_log_conv_channels
        layers = []
        in_ch = config.well_log_in_channels
        for c in ch:
            layers.extend([
                nn.Conv1d(in_ch, c, kernel_size=7, padding=3, stride=config.well_log_stride),
                nn.GroupNorm(min(8, c), c),
                nn.GELU(),
            ])
            in_ch = c
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(ch[-1], self.output_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.output_tokens, self.output_dim) * 0.02)

    def forward(self, x):
        B, C, depth = x.shape
        x = self.conv(x)
        x = x[:, :, :self.output_tokens] if x.shape[2] > self.output_tokens else x
        pad = self.output_tokens - x.shape[2]
        if pad > 0:
            x = nn.functional.pad(x, (0, pad))
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = x + self.pos_embed
        return x


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


class SpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, config.spatial_mlp_hidden),
            nn.GELU(),
            nn.Linear(config.spatial_mlp_hidden, config.d_model),
        )
        self.value_proj = nn.Linear(1, config.d_model) if config.spatial_mlp_hidden > 0 else None
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


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids):
        return self.embed(input_ids)
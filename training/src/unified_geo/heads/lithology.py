import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class LithologyHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_dim = config.num_heads * config.dh
        self.proj = nn.Linear(input_dim, 256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final = nn.Conv2d(32, config.lithology_num_classes, kernel_size=3, padding=1)

    def forward(self, slot_v):
        B, H, M, Dh = slot_v.shape
        pooled = slot_v.mean(dim=2)
        flat = pooled.reshape(B, H * Dh)
        x = F.gelu(self.proj(flat))
        x = x.view(B, 256, 1, 1).expand(-1, -1, 4, 4)
        x = F.gelu(self.deconv1(x))
        x = F.gelu(self.deconv2(x))
        x = F.gelu(self.deconv3(x))
        x = self.final(x)
        return F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
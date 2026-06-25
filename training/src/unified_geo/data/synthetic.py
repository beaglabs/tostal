import torch
import numpy as np
from typing import Iterator

from ..config import ModelConfig, TrainingConfig


class DataConfig:
    """Combines ModelConfig + TrainingConfig for data generation."""
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainingConfig):
        self.well_log_in_channels = model_cfg.well_log_in_channels
        self.well_log_depth = train_cfg.well_log_depth
        self.image_channels = model_cfg.image_channels
        self.image_size = train_cfg.image_size
        self.spatial_n_points = train_cfg.spatial_n_points
        self.text_seq_len = train_cfg.text_seq_len
        self.vocab_size = model_cfg.vocab_size
        self.well_mask_prob = train_cfg.well_mask_prob
        self.image_mask_prob = train_cfg.image_mask_prob
        self.spatial_mask_prob = train_cfg.spatial_mask_prob


def generate_well_log(batch_size, n_curves, depth, device, seed=None):
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    random_walk = torch.cumsum(
        torch.randn(batch_size, n_curves, depth, device=device, generator=g) * 0.05, dim=-1
    )
    drift = torch.linspace(0, 1, depth, device=device).view(1, 1, -1) * torch.randn(
        batch_size, n_curves, 1, device=device, generator=g
    ) * 0.1
    data = random_walk + drift
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True) + 1e-5
    data = (data - mean) / std
    return data


def generate_image(batch_size, channels, size, device, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data = np.random.randn(batch_size, channels, size, size).astype(np.float32)
    for b in range(batch_size):
        for c in range(channels):
            freq = 1 + np.random.rand() * 4
            n_bands = 3 + int(np.random.rand() * 3)
            for _ in range(n_bands):
                angle = np.random.rand() * np.pi
                pattern = np.sin(
                    freq * (np.arange(size).reshape(1, -1) * np.cos(angle) +
                            np.arange(size).reshape(-1, 1) * np.sin(angle))
                )
                data[b, c] += pattern * 0.3
    data = data / np.maximum(data.std(axis=(2, 3), keepdims=True) + 1e-5, 0.1)
    return torch.from_numpy(data).to(device)


def generate_spatial(batch_size, n_points, device, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    coords = rng.uniform(0, 1, (batch_size, n_points, 3)).astype(np.float32)
    values = (
        np.sin(coords[:, :, 0] * 6) * np.cos(coords[:, :, 1] * 4) +
        np.exp(-coords[:, :, 2] * 2) * 0.5 +
        rng.normal(0, 0.1, (batch_size, n_points))
    ).astype(np.float32)
    data = np.concatenate([coords, values[:, :, np.newaxis]], axis=-1)
    return torch.from_numpy(data).to(device)


def generate_text_batch(batch_size, seq_len, vocab_size, device, seed=None):
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None
    return torch.randint(0, min(vocab_size, 256), (batch_size, seq_len), device=device, generator=g)


def apply_well_log_mask(data, mask_prob, device):
    B, C, depth = data.shape
    masked = data.clone()
    for b in range(B):
        n_mask = max(1, int(depth * mask_prob))
        start = torch.randint(0, depth - n_mask, (1,), device=device).item()
        masked[b, :, start:start + n_mask] = 0
        mask = torch.zeros(B, C, depth, device=device)
        mask[b, :, start:start + n_mask] = 1
    return masked, mask


def apply_image_mask(data, mask_prob, device):
    B, C, H, W = data.shape
    masked = data.clone()
    mask = torch.zeros(B, C, H, W, device=device)
    patch_size = H // 2
    for b in range(B):
        if torch.rand(1, device=device).item() < mask_prob:
            i = torch.randint(0, H - patch_size, (1,), device=device).item()
            j = torch.randint(0, W - patch_size, (1,), device=device).item()
            masked[b, :, i:i + patch_size, j:j + patch_size] = 0
            mask[b, :, i:i + patch_size, j:j + patch_size] = 1
    return masked, mask


def apply_spatial_mask(data, mask_prob, device):
    B, N, F = data.shape
    masked = data.clone()
    mask = torch.zeros(B, N, device=device)
    n_mask = max(1, int(N * mask_prob))
    for b in range(B):
        indices = torch.randperm(N, device=device)[:n_mask]
        masked[b, indices, :] = 0
        mask[b, indices] = 1
    return masked, mask


def generate_mixed_batch(batch_size, data_cfg: DataConfig, device, seed=None):
    well_log = generate_well_log(
        batch_size, data_cfg.well_log_in_channels, data_cfg.well_log_depth, device, seed
    )
    image = generate_image(batch_size, data_cfg.image_channels, data_cfg.image_size, device, seed)
    spatial = generate_spatial(batch_size, data_cfg.spatial_n_points, device, seed)
    text = generate_text_batch(batch_size, data_cfg.text_seq_len, data_cfg.vocab_size, device, seed)

    masked_well, mask_well = apply_well_log_mask(well_log, data_cfg.well_mask_prob, device)
    masked_image, mask_image = apply_image_mask(image, data_cfg.image_mask_prob, device)
    masked_spatial, mask_spatial = apply_spatial_mask(spatial, data_cfg.spatial_mask_prob, device)

    return {
        "well_log": masked_well,
        "well_target": well_log,
        "well_mask": mask_well,
        "image": masked_image,
        "image_target": image,
        "image_mask": mask_image,
        "spatial": masked_spatial,
        "spatial_target": spatial,
        "spatial_mask": mask_spatial,
        "text_ids": text,
    }


def mixed_batch_iterator(batch_size, model_cfg, train_cfg, device, num_batches=None):
    data_cfg = DataConfig(model_cfg, train_cfg)
    step = 0
    while num_batches is None or step < num_batches:
        yield generate_mixed_batch(batch_size, data_cfg, device, seed=42 + step)
        step += 1
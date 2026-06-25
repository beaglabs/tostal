import torch
import numpy as np
from typing import Iterator


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
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import torch

from .normalize_force2020 import (
    FACIES_CLASSES,
    CURVE_ALIASES,
    _download,
    _match_curve,
    _normalize_well_name,
)

try:
    import lasio
except ImportError:
    lasio = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

TARANAKI_URL = "https://zenodo.org/records/3832955/files/taranaki-basin-curated-well-logs.tar.gz"


def download_taranaki(cache_dir="data/taranaki"):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    las_archive = cache / "taranaki_basin_well_logs.tar.gz"
    _download(TARANAKI_URL, las_archive, "Taranaki Basin LAS")
    return cache


def parse_taranaki_las_files(las_dir, n_depth=512, n_curves=6):
    if lasio is None:
        raise ImportError("lasio required: pip install lasio")
    las_dir = Path(las_dir)
    las_paths = sorted(las_dir.rglob("*.las")) + sorted(las_dir.rglob("*.LAS"))
    if not las_paths:
        raise FileNotFoundError(f"No LAS files found recursively in {las_dir}")

    wells = {}
    for las_path in las_paths:
        well_name = las_path.stem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            las = lasio.read(las_path)

        curve_data = np.zeros((n_curves, n_depth), dtype=np.float32)
        depths = None
        for i, key in enumerate(["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]):
            data = _match_curve(las, key)
            if data is not None:
                if depths is None:
                    if las.index is not None:
                        depths = las.index
                    else:
                        depths = data

        if depths is None:
            continue

        for i, key in enumerate(["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]):
            data = _match_curve(las, key)
            if data is not None:
                data = np.asarray(data, dtype=np.float32)
            if data is not None and len(data) == len(depths):
                valid = ~np.isnan(data)
                if valid.sum() > 10:
                    data[~valid] = np.nanmean(data[valid]) if valid.any() else 0
                    f = np.linspace(0, len(data) - 1, n_depth)
                    curve_data[i] = np.interp(f, np.arange(len(data)), data)

        valid_mask = curve_data.sum(axis=0) != 0
        if valid_mask.sum() < n_depth * 0.1:
            continue

        mean = curve_data.mean(axis=1, keepdims=True) + 1e-8
        std = curve_data.std(axis=1, keepdims=True) + 1e-8
        curve_data = (curve_data - mean) / std
        curve_data[:, ~valid_mask] = 0

        facies = _read_taranaki_lithology(las, n_depth)

        wells[well_name] = {
            "well_log": torch.from_numpy(curve_data),
            "metadata": {
                "name": well_name,
                "basin": "Taranaki Basin",
            },
        }
        if facies is not None:
            wells[well_name]["facies"] = torch.from_numpy(facies)
    return wells


def _read_taranaki_lithology(las, n_depth):
    lith_candidates = ["LITH", "LITHOLOGY", "FACIES", "FACIES_CLASS", "LABEL"]
    facies_label_map = {
        "sandstone": 0, "sand": 0,
        "sandstoneshale": 1, "siltstone": 1, "silt": 1,
        "shale": 2, "mudstone": 2, "claystone": 2,
        "marl": 3, "calcareous shale": 3,
        "dolomite": 4, "dolostone": 4,
        "limestone": 5,
        "chalk": 6,
        "halite": 7, "salt": 7,
        "anhydrite": 8,
        "tuff": 9, "volcanic": 9, "volcaniclastic": 9,
        "coal": 10, "lignite": 10,
        "basement": 11, "igneous": 11, "granite": 11,
    }

    for name in lith_candidates:
        if name in las.keys():
            raw = np.asarray(las[name].data)
            labels = np.full(n_depth, -1, dtype=np.int64)
            src_x = np.arange(len(raw))
            target_x = np.linspace(0, len(raw) - 1, n_depth)

            if raw.dtype.kind in ("U", "S", "O"):
                raw_str = np.array([str(v).strip().lower() for v in raw])
                for j in range(n_depth):
                    nearest = int(round(target_x[j]))
                    nearest = min(max(nearest, 0), len(raw_str) - 1)
                    val = raw_str[nearest]
                    if val in facies_label_map:
                        labels[j] = facies_label_map[val]
            else:
                raw = np.asarray(raw, dtype=np.float32)
                class_idx = np.full(len(raw), -1, dtype=np.int64)
                for code, idx in facies_label_map.items():
                    if isinstance(code, str):
                        continue
                raw_int = raw.astype(np.int64)
                unique_vals = np.unique(raw_int[~np.isnan(raw)])
                for val in unique_vals:
                    if val >= 0 and val < len(FACIES_CLASSES):
                        class_idx[raw_int == val] = val
                for j in range(n_depth):
                    nearest = int(round(target_x[j]))
                    nearest = min(max(nearest, 0), len(class_idx) - 1)
                    labels[j] = class_idx[nearest]

            if (labels >= 0).sum() > 0:
                return labels

    for name in lith_candidates:
        if name in las.keys():
            raw = np.asarray(las[name].data, dtype=np.float32)
            labels = np.full(n_depth, -1, dtype=np.int64)
            src_x = np.arange(len(raw))
            target_x = np.linspace(0, len(raw) - 1, n_depth)
            raw_int = raw.astype(np.int64)
            for j in range(n_depth):
                nearest = int(round(target_x[j]))
                nearest = min(max(nearest, 0), len(raw_int) - 1)
                val = raw_int[nearest]
                if 0 <= val < len(FACIES_CLASSES):
                    labels[j] = val
            if (labels >= 0).sum() > 0:
                return labels

    return None


def process_taranaki(
    cache_dir="data/taranaki",
    n_depth=512,
    n_curves=6,
    force_reprocess=False,
):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    processed_path = cache / "processed.pt"

    if processed_path.exists() and not force_reprocess:
        print(f"Loading cached Taranaki data from {processed_path}")
        return torch.load(processed_path, weights_only=False)

    cache = download_taranaki(cache_dir)
    las_archive = cache / "taranaki_basin_well_logs.tar.gz"
    las_dir = cache / "las"
    las_files = list(las_dir.rglob("*.las")) + list(las_dir.rglob("*.LAS")) if las_dir.exists() else []
    if not las_files:
        if las_dir.exists():
            shutil.rmtree(las_dir)
        print("Extracting Taranaki LAS files...")
        las_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(las_archive, "r:gz") as tf:
            tf.extractall(las_dir)

        children = list(las_dir.iterdir())
        wrapper = None
        for child in children:
            if child.is_dir() and child != las_dir:
                child_contents = list(child.iterdir())
                has_las = any(f.suffix.lower() == ".las" for f in child_contents)
                has_subdirs = any(d.is_dir() for d in child_contents)
                if has_las or has_subdirs:
                    wrapper = child
                    break
        if wrapper:
            print(f"  Flattening wrapper directory {wrapper.name}")
            for item in wrapper.iterdir():
                target = las_dir / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
            shutil.rmtree(wrapper)

    print(f"Parsing Taranaki LAS files from {las_dir}...")
    wells = parse_taranaki_las_files(las_dir, n_depth=n_depth, n_curves=n_curves)
    print(f"  Parsed {len(wells)} wells")
    labeled = sum(1 for w in wells.values() if "facies" in w)
    print(f"  Loaded lithology labels for {labeled} wells")

    torch.save(wells, processed_path)
    print(f"Saved {len(wells)} processed wells to {processed_path}")
    return wells
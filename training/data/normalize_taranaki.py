import csv
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .normalize_force2020 import FACIES_CLASSES, _download

TARANAKI_URL = (
    "https://zenodo.org/records/3832955/files/"
    "taranaki-basin-curated-well-logs.tar.gz?download=1"
)

CURVE_COLUMNS = {
    "GR": "GR",
    "RT": "RESD",
    "RHOB": "DENS",
    "NPHI": "NEUT",
    "DT": "DTC",
    "CALI": "CALI",
}

TARANAKI_FORMATION_LITHOLOGY = {
    "Mangahewa": 0,        # sandstone
    "McKee": 0,             # sandstone
    "Kaimiro": 0,            # sandstone
    "Farewell": 0,           # sandstone
    "Pakawau": 0,            # sandstone
    "North Cape": 0,         # sandstone
    "Manganui": 2,           # shale/mudstone
    "Otaraoa": 2,            # mudstone
    "Turi": 2,               # mudstone/siltstone
    "Wainui": 2,             # mudstone
    "Mangaa": 2,             # mudstone
    "Urenui": 1,             # siltstone/sandstone-shale
    "Mount Messenger": 1,    # siltstone/sandstone interbeds
    "Mohakatino": 9,         # volcaniclastic/tuff
    "Tikorangi": 5,          # limestone
    "Matapo": 5,             # limestone
    "Ariki": 1,              # siltstone
    "Rakopi": 0,             # sandstone + coal
}

TARANAKI_LABEL_MAP = {}
TARANAKI_LABEL_NAMES = {}


def download_taranaki(cache_dir="data/taranaki"):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "taranaki_basin_well_logs.tar.gz"
    _download(TARANAKI_URL, archive, "Taranaki Basin LAS")
    return cache


def _extract_archive(cache_dir):
    cache = Path(cache_dir)
    archive = cache / "taranaki_basin_well_logs.tar.gz"
    extract_dir = cache / "extracted"

    csv_path = extract_dir / "logs.csv"
    if csv_path.exists():
        return extract_dir

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(extract_dir)
    except tarfile.ReadError:
        archive.unlink()
        cache = download_taranaki(cache_dir)
        archive = cache / "taranaki_basin_well_logs.tar.gz"
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(extract_dir)

    wrapper = None
    for item in extract_dir.iterdir():
        if item.is_dir():
            csv_in_wrapper = item / "logs.csv"
            if csv_in_wrapper.exists():
                wrapper = item
                break

    if wrapper:
        for item in wrapper.iterdir():
            target = extract_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        shutil.rmtree(wrapper)

    return extract_dir


def _resolve_lithology_label(formation_name):
    if not formation_name or not formation_name.strip():
        return -1
    name = formation_name.strip()
    if name in TARANAKI_FORMATION_LITHOLOGY:
        return int(TARANAKI_FORMATION_LITHOLOGY[name])
    return -1


def _build_label_map(wells):
    """Remap sparse global class indices to contiguous local indices.

    XGBoost multi:softmax requires classes to be 0..K-1. Formation-proxy
    labels from Taranaki are sparse (e.g. [0, 1, 2, 5, 9]), so we remap to
    [0, 1, 2, 3, 4] and store the mapping.
    """
    global TARANAKI_LABEL_MAP, TARANAKI_LABEL_NAMES
    TARANAKI_LABEL_MAP = {}
    TARANAKI_LABEL_NAMES = {}

    unique_labels = set()
    for data in wells.values():
        if "facies" in data:
            arr = data["facies"].numpy()
            valid = arr[arr >= 0]
            unique_labels.update(valid.tolist())

    sorted_labels = sorted(int(l) for l in unique_labels)
    for local_idx, global_idx in enumerate(sorted_labels):
        TARANAKI_LABEL_MAP[global_idx] = local_idx
        name = FACIES_CLASSES[global_idx] if global_idx < len(FACIES_CLASSES) else f"class_{global_idx}"
        TARANAKI_LABEL_NAMES[local_idx] = name

    print(f"  Label remap: {TARANAKI_LABEL_MAP} → {dict(TARANAKI_LABEL_NAMES)}")

    for data in wells.values():
        if "facies" in data:
            facies = data["facies"].clone().numpy()
            remapped = np.full_like(facies, -1)
            for global_idx, local_idx in TARANAKI_LABEL_MAP.items():
                remapped[facies == global_idx] = local_idx
            data["facies"] = torch.from_numpy(remapped)


def parse_taranaki_csv(extract_dir, n_depth=512, n_curves=6, max_wells=None):
    csv_path = Path(extract_dir) / "logs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"logs.csv not found in {extract_dir}")

    curve_keys = ["GR", "RT", "RHOB", "NPHI", "DT", "CALI"]

    print(f"  Reading {csv_path}...")
    well_data = defaultdict(lambda: {"depths": [], "curves": {k: [] for k in curve_keys}, "formations": []})

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            well_name = row.get("WELLNAME", "").strip()
            if not well_name:
                continue
            if max_wells and len(well_data) > max_wells and well_name not in well_data:
                continue

            try:
                depth = float(row.get("DEPT", ""))
            except (ValueError, TypeError):
                continue

            w = well_data[well_name]
            w["depths"].append(depth)
            for std_key, csv_col in CURVE_COLUMNS.items():
                try:
                    val = float(row.get(csv_col, ""))
                except (ValueError, TypeError):
                    val = float("nan")
                w["curves"][std_key].append(val)
            w["formations"].append(row.get("FORMATION", ""))

    print(f"  Loaded {len(well_data)} wells from CSV")

    wells = {}
    for well_name, data in well_data.items():
        if len(data["depths"]) < 10:
            continue

        depths = np.array(data["depths"], dtype=np.float32)
        sort_idx = np.argsort(depths)
        depths = depths[sort_idx]

        curve_data = np.zeros((n_curves, n_depth), dtype=np.float32)
        curves_ok = 0
        for i, key in enumerate(curve_keys):
            raw = np.array(data["curves"][key], dtype=np.float32)[sort_idx]
            valid = ~np.isnan(raw)
            if valid.sum() < 10:
                continue
            raw[~valid] = np.nanmean(raw[valid])
            f = np.linspace(0, len(raw) - 1, n_depth)
            curve_data[i] = np.interp(f, np.arange(len(raw)), raw)
            curves_ok += 1

        if curves_ok < 3:
            continue

        valid_mask = curve_data.sum(axis=0) != 0
        if valid_mask.sum() < n_depth * 0.1:
            continue

        mean = curve_data.mean(axis=1, keepdims=True) + 1e-8
        std = curve_data.std(axis=1, keepdims=True) + 1e-8
        curve_data = (curve_data - mean) / std
        curve_data[:, ~valid_mask] = 0

        formations = [data["formations"][i] for i in sort_idx]
        facies = np.full(n_depth, -1, dtype=np.int64)
        formation_depth_map = {}
        for j in range(len(depths)):
            fm = formations[j].strip()
            if fm:
                formation_depth_map.setdefault(fm, []).append(depths[j])

        for fm, fm_depths in formation_depth_map.items():
            label = _resolve_lithology_label(fm)
            if label < 0:
                continue
            fm_min, fm_max = min(fm_depths), max(fm_depths)
            target_depths = np.linspace(depths[0], depths[-1], n_depth)
            in_range = (target_depths >= fm_min) & (target_depths <= fm_max)
            facies[in_range] = label

        wells[well_name] = {
            "well_log": torch.from_numpy(curve_data),
            "metadata": {"name": well_name, "basin": "Taranaki Basin"},
        }
        if (facies >= 0).sum() > 0:
            wells[well_name]["facies"] = torch.from_numpy(facies)

    total_labeled = sum(1 for w in wells.values() if "facies" in w)
    print(f"  Parsed {len(wells)} wells ({total_labeled} with formation-proxy labels)")

    _build_label_map(wells)
    return wells


def process_taranaki(cache_dir="data/taranaki", n_depth=512, n_curves=6, force_reprocess=False):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    processed_path = cache / "processed.pt"

    if processed_path.exists() and not force_reprocess:
        print(f"Loading cached Taranaki data from {processed_path}")
        return torch.load(processed_path, weights_only=False)

    download_taranaki(cache_dir)
    extract_dir = _extract_archive(cache_dir)
    wells = parse_taranaki_csv(extract_dir, n_depth=n_depth, n_curves=n_curves)

    torch.save(wells, processed_path)
    print(f"Saved {len(wells)} processed wells to {processed_path}")
    return wells
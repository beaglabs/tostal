"""NPD core description pipeline for synthetic core training.

Parses core description data from NPD/SODIR to extract lithology labels,
porosity, permeability, and grain density measurements for wells that have
both wireline logs and core data. Used to train the synthetic core model
that predicts core properties from log curves.
"""
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .normalize_force2020 import FACIES_CLASSES

LITHOLOGY_KEYWORDS = {
    "sandstone": 0,
    "sand": 0,
    "quartz arenite": 0,
    "arkose": 0,
    "greywacke": 0,
    "sandstoneshale": 1,
    "sandstone/shale": 1,
    "siltstone": 1,
    "silty": 1,
    "silt": 1,
    "muddy sandstone": 1,
    "shaly sand": 1,
    "shale": 2,
    "mudstone": 2,
    "claystone": 2,
    "clay": 2,
    "mud": 2,
    "marl": 3,
    "calcareous mudstone": 3,
    "calcareous shale": 3,
    "calcareous claystone": 3,
    "marly": 3,
    "dolomite": 4,
    "dolostone": 4,
    "dolomitic": 4,
    "limestone": 5,
    "micrite": 5,
    "wackestone": 5,
    "packstone": 5,
    "grainstone": 5,
    "boundstone": 5,
    "chalk": 6,
    "chalky": 6,
    "halite": 7,
    "salt": 7,
    "evaporite": 7,
    "anhydrite": 8,
    "anhydritic": 8,
    "gypsum": 8,
    "tuff": 9,
    "volcanic": 9,
    "volcaniclastic": 9,
    "basalt": 9,
    "andesite": 9,
    "rhyolite": 9,
    "igneous": 9,
    "coal": 10,
    "lignite": 10,
    "carbonaceous": 10,
    "basement": 11,
    "granite": 11,
    "gneiss": 11,
    "schist": 11,
    "crystalline": 11,
    "metamorphic": 11,
}

NUMERIC_EXTRACTORS = {
    "porosity": re.compile(
        r"porosity[:\s]*(\d+\.?\d*)\s*(?:%|p\.?u\.?|fraction)?", re.IGNORECASE
    ),
    "permeability": re.compile(
        r"permeability[:\s]*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:mD|md|millidarcy)?",
        re.IGNORECASE,
    ),
    "grain_density": re.compile(
        r"(?:grain|matrix)\s*density[:\s]*(\d+\.?\d*)\s*(?:g/cc|g/cm3|kg/m3)?",
        re.IGNORECASE,
    ),
}


def parse_lithology_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    text_lower = text.strip().lower()
    for keyword, idx in sorted(LITHOLOGY_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in text_lower:
            return idx
    return None


def parse_numeric_from_text(text: str, extractors: dict) -> dict:
    result = {}
    for key, pattern in extractors.items():
        match = pattern.search(text)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                pass
    return result


def extract_core_labels(
    core_descriptions: list,
    n_depth: int = 512,
    total_md_min: float = 0.0,
    total_md_max: float = 512.0,
) -> dict:
    """Convert core description records to depth-indexed label arrays.

    Args:
        core_descriptions: list of dicts with keys like 'core_depth_top',
            'core_depth_bottom', 'lithology_description',
            'porosity', 'permeability', 'grain_density'
        n_depth: number of depth points
        total_md_min: minimum measured depth of the well
        total_md_max: maximum measured depth of the well

    Returns:
        dict with 'lithology', 'porosity', 'permeability',
            'grain_density', 'confidence' arrays
    """
    depth_range = total_md_max - total_md_min
    if depth_range <= 0:
        depth_range = 1.0

    lithology = np.full(n_depth, -1, dtype=np.int64)
    porosity = np.full(n_depth, np.nan, dtype=np.float32)
    permeability = np.full(n_depth, np.nan, dtype=np.float32)
    grain_density = np.full(n_depth, np.nan, dtype=np.float32)
    confidence = np.zeros(n_depth, dtype=np.float32)

    for record in core_descriptions:
        top = float(record.get("core_depth_top", record.get("top_md", 0)))
        base = float(record.get("core_depth_bottom", record.get("base_md", 0)))

        top_idx = int((top - total_md_min) / depth_range * n_depth)
        base_idx = int((base - total_md_min) / depth_range * n_depth)
        top_idx = max(0, min(n_depth - 1, top_idx))
        base_idx = max(top_idx + 1, min(n_depth, base_idx))

        desc = str(record.get("lithology_description", record.get("description", "")))
        lith_idx = parse_lithology_from_text(desc)

        numerics = {}
        for key in ["porosity", "permeability", "grain_density"]:
            if key in record and record[key] is not None:
                try:
                    numerics[key] = float(record[key])
                except (ValueError, TypeError):
                    pass

        if not numerics:
            numerics = parse_numeric_from_text(desc, NUMERIC_EXTRACTORS)

        for j in range(top_idx, base_idx):
            if lith_idx is not None:
                lithology[j] = lith_idx
                confidence[j] = max(confidence[j], 0.8)
            if "porosity" in numerics:
                porosity[j] = numerics["porosity"]
                confidence[j] = max(confidence[j], 0.8)
            if "permeability" in numerics:
                permeability[j] = numerics["permeability"]
                confidence[j] = max(confidence[j], 0.7)
            if "grain_density" in numerics:
                grain_density[j] = numerics["grain_density"]
                confidence[j] = max(confidence[j], 0.9)

    return {
        "lithology": torch.from_numpy(lithology),
        "porosity": torch.from_numpy(porosity),
        "permeability": torch.from_numpy(permeability),
        "grain_density": torch.from_numpy(grain_density),
        "confidence": torch.from_numpy(confidence),
    }


def generate_core_training_labels(
    wells: dict,
    core_data: dict,
    n_depth: int = 512,
) -> int:
    """Attach core-derived labels to wells dict for synthetic core training.

    Args:
        wells: dict of well_name -> {well_log, metadata, ...}
        core_data: dict of well_name -> list of core description records
        n_depth: number of depth points

    Returns:
        number of wells with core labels added
    """
    count = 0
    for well_name, well in wells.items():
        if well_name not in core_data:
            continue

        md_values = well.get("metadata", {})
        total_md_min = float(md_values.get("depth_min", 0))
        total_md_max = float(md_values.get("depth_max", n_depth))

        labels = extract_core_labels(
            core_data[well_name],
            n_depth=n_depth,
            total_md_min=total_md_min,
            total_md_max=total_md_max,
        )

        well["core_lithology"] = labels["lithology"]
        well["core_porosity"] = labels["porosity"]
        well["core_permeability"] = labels["permeability"]
        well["core_grain_density"] = labels["grain_density"]
        well["core_confidence"] = labels["confidence"]
        count += 1

    return count
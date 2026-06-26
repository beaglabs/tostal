"""LITHOS thin-section petrography dataset loader.

Uses Croissant metadata for dataset discovery and kagglehub for
authenticated download + caching. Parses CSV annotations to build
an image-to-mineral-label index for DINOv2 classifier training.

Dataset: Paola Ruiz Puentes, "Towards Automated Petrography"
         NeurIPS 2025 Datasets and Benchmarks.
         211,604 RGB polarized-light patches, 25 mineral classes.
"""
import csv
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CROISSANT_URL = "https://www.kaggle.com/datasets/paolaruizpuentes/lithos-dataset/croissant/download"
KAGGLE_DATASET = "paolaruizpuentes/lithos-dataset"

MINERAL_CLASSES = [
    "quartz", "plagioclase", "k-feldspar", "biotite", "muscovite",
    "chlorite", "calcite", "dolomite", "olivine", "pyroxene",
    "amphibole", "garnet", "opaques", "zircon", "apatite",
    "epidote", "serpentine", "talc", "gypsum", "anhydrite",
    "halite", "clay", "glauconite", "chert", "organic",
]

_MINERAL_TO_IDX = {name: i for i, name in enumerate(MINERAL_CLASSES)}


def _resolve_mineral_class(raw: str) -> int:
    """Map a raw mineral name from the CSV to a canonical class index."""
    name = raw.strip().lower().replace("_", " ").replace("-", " ")
    name = " ".join(name.split())
    if name in _MINERAL_TO_IDX:
        return _MINERAL_TO_IDX[name]
    for canonical in MINERAL_CLASSES:
        if canonical in name or name in canonical:
            return _MINERAL_TO_IDX[canonical]
    logger.debug("Unknown mineral class: %r, mapping to quartz (0)", raw)
    return 0


def fetch_croissant_metadata(source: Optional[str] = None) -> dict:
    """Parse Croissant JSON-LD metadata for LITHOS.

    Args:
        source: URL or local file path to Croissant JSON-LD. Defaults to
                the Kaggle Croissant endpoint.

    Returns:
        Parsed metadata dict.
    """
    target = source or CROISSANT_URL

    if Path(target).exists():
        with open(target, "r", encoding="utf-8") as fh:
            return json.load(fh)

    try:
        import mlcroissant as mlc
    except ImportError:
        raise ImportError("pip install mlcroissant required for remote Croissant URLs")

    ds = mlc.Dataset(target)
    return ds.metadata.to_json()


def load_lithos_dataset(cache_dir: Optional[str] = None) -> Path:
    """Download the LITHOS dataset via kagglehub and return the extracted path.

    kagglehub handles authentication, download resumption, and caching.
    Returns the local filesystem path to the extracted dataset directory.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("pip install kagglehub required")

    path_str = kagglehub.dataset_download(KAGGLE_DATASET, path=cache_dir)
    return Path(path_str)


class LithosIndex:
    """Index mapping images to mineral labels for the LITHOS dataset."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.records: list[dict] = []
        self._num_classes = len(MINERAL_CLASSES)
        self._build_index()

    def _find_csv(self) -> Path:
        """Locate the annotation CSV in the dataset directory."""
        for candidate in sorted(self.dataset_path.rglob("*.csv")):
            with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline(128).lower()
                if any(kw in header for kw in ("mineral", "class", "label", "lithology")):
                    return candidate
        csvs = sorted(self.dataset_path.rglob("*.csv"))
        if csvs:
            return csvs[0]
        raise FileNotFoundError(f"No CSV found in {self.dataset_path}")

    def _find_image_root(self) -> Path:
        """Find the root directory containing PNG images."""
        png_dirs = set()
        for png in self.dataset_path.rglob("*.png"):
            png_dirs.add(png.parent)
        if len(png_dirs) == 1:
            return next(iter(png_dirs))
        for d in sorted(png_dirs):
            if "patch" in str(d).lower() or "image" in str(d).lower():
                return d
        if png_dirs:
            return sorted(png_dirs)[0]
        return self.dataset_path

    def _build_index(self):
        csv_path = self._find_csv()
        self.image_root = self._find_image_root()
        logger.info("Parsing annotations: %s", csv_path)
        logger.info("Image root: %s", self.image_root)

        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                fh.seek(0)
                reader = csv.DictReader(fh)

            col_file = self._find_column(reader.fieldnames, ("filename", "file", "image", "path", "name"))
            col_label = self._find_column(reader.fieldnames, ("mineral", "class", "label", "lithology", "type", "category"))

            if col_file is None or col_label is None:
                raise ValueError(
                    f"Could not identify file/label columns in CSV. "
                    f"Fields: {reader.fieldnames}"
                )

            seen = set()
            for row in reader:
                fname = row.get(col_file, "").strip()
                label_str = row.get(col_label, "").strip()
                if not fname or not label_str:
                    continue
                if fname in seen:
                    continue
                seen.add(fname)
                label_idx = _resolve_mineral_class(label_str)
                self.records.append({"file": fname, "label": label_idx})

        logger.info("Indexed %d images across %d classes", len(self), self._num_classes)

    @staticmethod
    def _find_column(fieldnames, candidates):
        if fieldnames is None:
            return None
        lower = [f.lower() for f in fieldnames]
        for cand in candidates:
            if cand in lower:
                return fieldnames[lower.index(cand)]
        return fieldnames[0]

    def __len__(self):
        return len(self.records)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_names(self):
        return list(MINERAL_CLASSES)

    def get_image(self, idx: int) -> Image.Image:
        """Load a single image by index."""
        rec = self.records[idx]
        img_path = self._resolve_image_path(rec["file"])
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def get_label(self, idx: int) -> int:
        return self.records[idx]["label"]

    def _resolve_image_path(self, fname: str) -> Path:
        direct = self.image_root / fname
        if direct.exists():
            return direct
        for png in self.image_root.rglob(fname):
            return png
        for png in self.dataset_path.rglob(fname):
            return png
        for png in self.dataset_path.rglob("*.png"):
            if png.name == fname or png.stem == Path(fname).stem:
                return png
        raise FileNotFoundError(f"Image not found: {fname}")

    def labels_array(self) -> np.ndarray:
        return np.array([r["label"] for r in self.records], dtype=np.int64)

    def label_counts(self) -> dict:
        arr = self.labels_array()
        return {MINERAL_CLASSES[i]: int((arr == i).sum()) for i in range(self._num_classes)}


def inspect_dataset(dataset_path: Path):
    """Print a summary of the LITHOS dataset contents."""
    print(f"Dataset path: {dataset_path}")
    print(f"\nContents:")
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            count = len(list(item.rglob("*")))
            print(f"  {item.name}/  ({count} entries)")
        else:
            print(f"  {item.name}  ({item.stat().st_size / 1e6:.1f} MB)")

    csvs = sorted(dataset_path.rglob("*.csv"))
    if csvs:
        print(f"\nCSV files ({len(csvs)}):")
        for c in csvs:
            with open(c, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline().strip()
                lines = sum(1 for _ in fh)
            print(f"  {c.relative_to(dataset_path)}  header={header[:100]}  rows={lines}")

    pngs = list(dataset_path.rglob("*.png"))
    print(f"\nPNG images: {len(pngs)}")
    if pngs:
        print(f"  Sample: {pngs[0].relative_to(dataset_path)}")
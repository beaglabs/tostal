from dataclasses import dataclass
from pathlib import Path

EXTENSION_MAP = {
    (".sgy", ".segy"): ("geology", "segy"),
    (".las",): ("geology", "las"),
    (".dlis",): ("geology", "dlis"),
    (".h5", ".hdf5"): ("materials", "hdf5"),
    (".nc", ".nc4"): ("climate", "netcdf"),
    (".grib", ".grb"): ("climate", "grib"),
    (".tif", ".tiff"): ("geology", "raster"),
    (".jpg", ".jpeg", ".png"): ("bioimaging", "raster"),
    (".zarr",): ("auto", ""),
    (".json",): ("metadata", ""),
}


@dataclass
class ClassifiedFile:
    filename: str
    file_format: str
    domain: str
    subdirectory: str


def classify_file(filename: str) -> ClassifiedFile:
    ext = Path(filename).suffix.lower()

    for extensions, (domain, subdirectory) in EXTENSION_MAP.items():
        if ext in extensions:
            if domain == "auto":
                return ClassifiedFile(
                    filename=filename,
                    file_format=ext.lstrip("."),
                    domain=ext.lstrip(".") if ext != ".zarr" else "auto",
                    subdirectory=filename.replace(ext, ""),
                )
            if domain == "metadata":
                return ClassifiedFile(
                    filename=filename,
                    file_format="json",
                    domain="metadata",
                    subdirectory=subdirectory,
                )
            return ClassifiedFile(
                filename=filename,
                file_format=ext.lstrip("."),
                domain=domain,
                subdirectory=subdirectory,
            )

    return ClassifiedFile(
        filename=filename,
        file_format=ext.lstrip(".") if ext else "unknown",
        domain="other",
        subdirectory="",
    )


def build_icechunk_path(classified: ClassifiedFile) -> str:
    if classified.domain == "other":
        return classified.filename
    if classified.domain == "metadata":
        return classified.filename
    if classified.domain == "auto":
        return classified.subdirectory
    return f"{classified.domain}/{classified.subdirectory}/{classified.filename}"
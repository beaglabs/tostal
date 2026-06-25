from typing import Optional

import xarray as xr
import zarr
from azure.storage.blob import BlobServiceClient

from app.config import get_settings
from app.models.db import Customer

settings = get_settings()


def _get_blob_service() -> Optional[BlobServiceClient]:
    if settings.azure_storage_connection_string:
        return BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
    return None


def get_store_path(customer: Customer, relative_path: str) -> str:
    return f"{customer.azure_container_name}/{relative_path}"


async def write_array_to_icechunk(
    customer: Customer,
    relative_path: str,
    data: xr.DataArray,
    chunk_size: Optional[list[int]] = None,
) -> dict:
    """Write an xarray DataArray to Icechunk/Zarr and return metadata."""
    store_path = get_store_path(customer, relative_path)

    blob_service = _get_blob_service()
    if blob_service:
        store = zarr.storage.AzureBlobStore(
            account_name=settings.azure_storage_account,
            account_key=settings.azure_storage_key,
            container=customer.azure_container_name,
            prefix=relative_path,
        )
    else:
        store = zarr.storage.MemoryStore()

    chunks = chunk_size or [min(128, s) for s in data.shape]
    data.to_dataset(name="data").to_zarr(store, mode="w", encoding={"data": {"chunks": chunks}})

    return {
        "icechunk_uri": f"icechunk://azure://{settings.azure_storage_account}/{store_path}",
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "chunk_size": chunks,
        "size_bytes": data.nbytes,
    }


async def read_array_from_icechunk(
    customer: Customer, relative_path: str
) -> Optional[xr.DataArray]:
    """Read an xarray DataArray from Icechunk/Zarr."""
    blob_service = _get_blob_service()
    if blob_service:
        store = zarr.storage.AzureBlobStore(
            account_name=settings.azure_storage_account,
            account_key=settings.azure_storage_key,
            container=customer.azure_container_name,
            prefix=relative_path,
        )
    else:
        store = zarr.storage.MemoryStore()

    try:
        ds = xr.open_zarr(store)
        return ds.get("data")
    except Exception:
        return None


async def write_notebook_state(
    customer: Customer, notebook_id: str, state_json: str
) -> str:
    """Write notebook state JSON to Icechunk. Returns the URI."""
    relative_path = f"notebooks/{notebook_id}/state.json"
    blob_service = _get_blob_service()

    if blob_service:
        blob_client = blob_service.get_blob_client(
            container=customer.azure_container_name, blob=relative_path
        )
        blob_client.upload_blob(state_json, overwrite=True)

    store_path = f"{customer.azure_container_name}/{relative_path}"
    return f"icechunk://azure://{settings.azure_storage_account}/{store_path}"


async def read_notebook_state(customer: Customer, notebook_id: str) -> Optional[str]:
    """Read notebook state JSON from Icechunk."""
    relative_path = f"notebooks/{notebook_id}/state.json"
    blob_service = _get_blob_service()

    if blob_service:
        blob_client = blob_service.get_blob_client(
            container=customer.azure_container_name, blob=relative_path
        )
        if blob_client.exists():
            return blob_client.download_blob().readall().decode("utf-8")
    return None
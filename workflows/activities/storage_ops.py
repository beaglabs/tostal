"""Temporal activity: Icechunk read/write operations."""

from temporalio import activity


@activity.defn
async def read_from_icechunk(params: dict) -> dict:
    customer_id = params["customer_id"]
    file_ids = params["file_ids"]

    # Stub: reads arrays from customer's Icechunk store
    return {"customer_id": customer_id, "file_ids": file_ids, "arrays": []}


@activity.defn
async def write_to_icechunk(params: dict) -> dict:
    customer_id = params["customer_id"]
    result = params["result"]
    prefix = params["prefix"]

    # Stub: writes result Zarr to customer's Icechunk store
    output_uri = f"icechunk://azure://account/customer-{customer_id}/{prefix}/output.zarr"

    return {
        "uri": output_uri,
        "shape": result.get("shape", []),
        "classes": result.get("classes", []),
        "confidence": result.get("confidence", 0.0),
    }


@activity.defn
async def convert_file(params: dict) -> dict:
    filename = params["filename"]
    file_format = params["format"]

    # Stub: converts uploaded file to Zarr format using Xarray
    return {"filename": filename, "format": file_format, "status": "converted"}
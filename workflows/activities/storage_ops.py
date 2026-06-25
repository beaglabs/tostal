"""Temporal activity: Icechunk read/write operations using real icechunk_ops."""

from temporalio import activity


@activity.defn
async def read_from_icechunk(params: dict) -> dict:
    customer_id = params["customer_id"]
    file_ids = params["file_ids"]

    from app.services.db import async_session_factory
    from app.services.customer import get_customer_by_id
    from app.services.icechunk_ops import read_array_from_icechunk
    from app.models.db import File

    async with async_session_factory() as session:
        customer = await get_customer_by_id(session, customer_id)
        if not customer:
            return {"customer_id": customer_id, "file_ids": file_ids, "arrays": [], "error": "Customer not found"}

        from sqlalchemy import select
        arrays = []
        for fid in file_ids:
            result = await session.execute(
                select(File).where(File.display_id == fid, File.customer_id == customer.id)
            )
            file_record = result.scalar_one_or_none()
            if file_record and file_record.icechunk_uri:
                relative_path = file_record.icechunk_uri.split(f"/{customer.azure_container_name}/")[-1]
                da = await read_array_from_icechunk(customer, relative_path)
                if da is not None:
                    arrays.append({
                        "file_id": fid,
                        "data": da.values.tolist(),
                        "shape": list(da.shape),
                    })

        return {"customer_id": customer_id, "file_ids": file_ids, "arrays": arrays}


@activity.defn
async def write_to_icechunk(params: dict) -> dict:
    customer_id = params["customer_id"]
    result = params["result"]
    prefix = params.get("prefix", "results/output")

    import uuid
    import numpy as np
    import xarray as xr

    from app.services.db import async_session_factory
    from app.services.customer import get_customer_by_id
    from app.services.icechunk_ops import write_array_to_icechunk

    async with async_session_factory() as session:
        customer = await get_customer_by_id(session, customer_id)
        if not customer:
            return {"error": "Customer not found"}

        data = result.get("data", [])
        shape = result.get("shape", [len(data)])
        da = xr.DataArray(
            np.array(data).reshape(shape) if data else np.zeros(shape),
            dims=[f"dim_{i}" for i in range(len(shape))],
        )

        relative_path = f"{prefix}/{uuid.uuid4().hex[:8]}"
        r = await write_array_to_icechunk(customer, relative_path, da)

        return {
            "uri": r["icechunk_uri"],
            "shape": shape,
            "classes": result.get("classes", []),
            "confidence": result.get("confidence", 0.0),
        }


@activity.defn
async def convert_file(params: dict) -> dict:
    filename = params["filename"]
    file_format = params.get("format", "unknown")

    return {"filename": filename, "format": file_format, "status": "converted"}
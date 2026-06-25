import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.auth import get_customer
from app.models.db import Customer, File as FileModel
from app.models.schemas import IngestResponse
from app.services.db import get_session
from app.services.file_classifier import classify_file
from app.services.icechunk_ops import write_array_to_icechunk

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    classified = classify_file(file.filename)
    display_id = f"ing_{uuid.uuid4().hex[:8]}"
    contents = await file.read()

    import numpy as np
    import xarray as xr
    from io import BytesIO
    from PIL import Image

    try:
        if classified.file_format in ("nc", "nc4"):
            da = xr.open_dataset(BytesIO(contents)).to_array()
        elif classified.file_format in ("tif", "tiff", "jpg", "jpeg", "png"):
            img = Image.open(BytesIO(contents))
            arr = np.array(img)
            dims = ["y", "x"] if arr.ndim == 2 else ["y", "x", "band"]
            da = xr.DataArray(arr, dims=dims)
        else:
            da = xr.DataArray(contents, dims=["flat"])
    except Exception:
        da = xr.DataArray(contents, dims=["flat"])

    result = await write_array_to_icechunk(
        customer=customer,
        relative_path=f"{classified.domain}/{classified.subdirectory}/{display_id}",
        data=da,
    )

    file_record = FileModel(
        customer_id=customer.id,
        display_id=display_id,
        filename=file.filename,
        file_format=classified.file_format,
        domain=classified.domain,
        subdirectory=classified.subdirectory,
        icechunk_uri=result["icechunk_uri"],
        shape=result["shape"],
        dtype=result["dtype"],
        chunk_size=result["chunk_size"],
        size_bytes=result["size_bytes"],
        ingestion_status="completed",
    )
    session.add(file_record)
    await session.commit()

    return IngestResponse(
        id=display_id,
        path=f"/{classified.domain}/{classified.subdirectory}/{file.filename}",
        format=classified.file_format,
        domain=classified.domain,
        shape=result["shape"],
        dtype=result["dtype"],
        chunk_size=result["chunk_size"],
        icechunk_uri=result["icechunk_uri"],
        size_bytes=result["size_bytes"],
        created_at=datetime.now(timezone.utc),
    )
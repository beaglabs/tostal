from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.auth import get_customer
from app.models.db import Customer, Job
from app.services.db import get_session
from app.services.icechunk_ops import read_array_from_icechunk

router = APIRouter(tags=["render"])


@router.get("/render/{job_id}")
async def render_result(
    job_id: str,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Job).where(Job.display_id == job_id, Job.customer_id == customer.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.result_icechunk_uri:
        raise HTTPException(status_code=404, detail="No result available")

    relative_path = job.result_icechunk_uri.split(f"/{customer.azure_container_name}/")[-1]
    da = await read_array_from_icechunk(customer, relative_path)
    if da is None:
        raise HTTPException(status_code=404, detail="Result data not found")


    metadata = {
        "job_id": job.display_id,
        "shape": list(da.shape),
        "dtype": str(da.dtype),
        "dims": list(da.dims) if hasattr(da, "dims") else [],
    }

    return {
        "metadata": metadata,
        "data_sample": da.values[:10].tolist() if da.ndim == 1 else da.values[:10, :10].tolist(),
    }
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.auth import get_customer
from app.models.db import Customer, Job
from app.models.schemas import KrigingRequest, KrigingResponse
from app.services.db import get_session

router = APIRouter(tags=["krigging"])


@router.post("/krigging", response_model=KrigingResponse)
async def start_kriging(
    request: KrigingRequest,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    display_id = f"krg_{uuid.uuid4().hex[:8]}"
    job = Job(
        customer_id=customer.id,
        display_id=display_id,
        job_type="krige",
        task="kriging",
        status="pending",
        input_file_ids=request.observations.file_ids,
        parameters=request.model_dump(),
    )
    session.add(job)
    await session.commit()

    return KrigingResponse(
        job_id=display_id,
        status="pending",
    )


@router.get("/krigging/{job_id}/result", response_model=KrigingResponse)
async def get_kriging_result(
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

    return KrigingResponse(
        job_id=job.display_id,
        status=job.status,
        result_path=job.result_icechunk_uri,
        result_shape=job.result_shape,
    )
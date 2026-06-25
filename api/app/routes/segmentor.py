import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.auth import get_customer
from app.models.db import Customer, Job
from app.models.schemas import SegmentorRequest, SegmentorResponse
from app.services.db import get_session

router = APIRouter(tags=["segmentor"])


@router.post("/segmentor", response_model=SegmentorResponse)
async def start_segmentation(
    request: SegmentorRequest,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    display_id = f"seg_{uuid.uuid4().hex[:8]}"
    job = Job(
        customer_id=customer.id,
        display_id=display_id,
        job_type="segment",
        task=request.task,
        status="pending",
        input_file_ids=request.file_ids,
        parameters=request.parameters,
    )
    session.add(job)
    await session.commit()

    return SegmentorResponse(
        job_id=display_id,
        status="pending",
    )


@router.get("/segmentor/{job_id}/result", response_model=SegmentorResponse)
async def get_segmentation_result(
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

    return SegmentorResponse(
        job_id=job.display_id,
        status=job.status,
        result_path=job.result_icechunk_uri,
        result_shape=job.result_shape,
        classes=job.classes,
        rendering_url=f"/v1/render/{job.display_id}" if job.result_icechunk_uri else None,
    )
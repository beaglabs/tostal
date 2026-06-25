import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio.client import Client

from app.config import get_settings
from app.middleware.auth import get_customer
from app.models.db import Customer, Job
from app.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResultResponse,
)
from app.services.db import get_session

router = APIRouter(tags=["classifiers"])
settings = get_settings()


async def _get_temporal_client():
    return await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )


@router.post("/classifiers", response_model=ClassifyResponse)
async def start_classification(
    request: ClassifyRequest,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    display_id = f"cls_{uuid.uuid4().hex[:8]}"
    job = Job(
        customer_id=customer.id,
        display_id=display_id,
        job_type="classify",
        task=request.task,
        status="pending",
        input_file_ids=request.file_ids,
        parameters=request.parameters,
    )
    session.add(job)
    await session.commit()

    try:
        client = await _get_temporal_client()
        handle = await client.start_workflow(
            "ClassifyWorkflow",
            {
                "customer_id": str(customer.id),
                "file_ids": request.file_ids,
                "task": request.task,
                "parameters": request.parameters,
            },
            id=f"temporal-{display_id}",
            task_queue="tostal-task-queue",
        )
        job.temporal_workflow_id = handle.id
        job.status = "processing"
        await session.commit()
    except Exception as e:
        job.status = "failed"
        job.error_message = f"Temporal dispatch failed: {str(e)}"
        await session.commit()

    return ClassifyResponse(
        job_id=display_id,
        status=job.status,
        estimated_completion=None,
    )


@router.get("/classifiers/{job_id}/result", response_model=ClassifyResultResponse)
async def get_classification_result(
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

    return ClassifyResultResponse(
        job_id=job.display_id,
        status=job.status,
        result_path=job.result_icechunk_uri,
        result_shape=job.result_shape,
        classes=job.classes,
        confidence=job.confidence,
        rendering_url=f"/v1/render/{job.display_id}" if job.result_icechunk_uri else None,
    )
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.middleware.auth import get_customer
from app.models.db import Customer, Notebook
from app.models.schemas import (
    NotebookCreate,
    NotebookResponse,
    NotebookState,
    NotebookUpdate,
)
from app.services.db import get_session
from app.services.icechunk_ops import read_notebook_state, write_notebook_state

router = APIRouter(tags=["notebooks"])


@router.get("/notebooks", response_model=list[NotebookResponse])
async def list_notebooks(
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook)
        .where(Notebook.customer_id == customer.id)
        .order_by(Notebook.updated_at.desc())
    )
    notebooks = result.scalars().all()
    return [
        NotebookResponse(
            id=str(nb.id),
            display_id=nb.display_id,
            name=nb.name,
            description=nb.description,
            icechunk_state_uri=nb.icechunk_state_uri,
            cell_count=nb.cell_count,
            status=nb.status,
            created_at=nb.created_at,
            updated_at=nb.updated_at,
        )
        for nb in notebooks
    ]


@router.post("/notebooks", response_model=NotebookResponse, status_code=201)
async def create_notebook(
    request: NotebookCreate,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    display_id = f"nb_{uuid.uuid4().hex[:8]}"

    state = NotebookState(
        notebook_id=display_id,
        name=request.name,
        description=request.description,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    icechunk_uri = await write_notebook_state(
        customer=customer,
        notebook_id=display_id,
        state_json=json.dumps(state.model_dump()),
    )

    notebook = Notebook(
        customer_id=customer.id,
        display_id=display_id,
        name=request.name,
        description=request.description,
        icechunk_state_uri=icechunk_uri,
    )
    session.add(notebook)
    await session.commit()
    await session.refresh(notebook)

    return NotebookResponse(
        id=str(notebook.id),
        display_id=notebook.display_id,
        name=notebook.name,
        description=notebook.description,
        icechunk_state_uri=notebook.icechunk_state_uri,
        cell_count=notebook.cell_count,
        status=notebook.status,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@router.get("/notebooks/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(
    notebook_id: str,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook).where(
            Notebook.display_id == notebook_id, Notebook.customer_id == customer.id
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return NotebookResponse(
        id=str(notebook.id),
        display_id=notebook.display_id,
        name=notebook.name,
        description=notebook.description,
        icechunk_state_uri=notebook.icechunk_state_uri,
        cell_count=notebook.cell_count,
        status=notebook.status,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@router.put("/notebooks/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(
    notebook_id: str,
    request: NotebookUpdate,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook).where(
            Notebook.display_id == notebook_id, Notebook.customer_id == customer.id
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    if request.name is not None:
        notebook.name = request.name
    if request.description is not None:
        notebook.description = request.description
    if request.cell_count is not None:
        notebook.cell_count = request.cell_count
    if request.status is not None:
        notebook.status = request.status

    await session.commit()
    await session.refresh(notebook)

    return NotebookResponse(
        id=str(notebook.id),
        display_id=notebook.display_id,
        name=notebook.name,
        description=notebook.description,
        icechunk_state_uri=notebook.icechunk_state_uri,
        cell_count=notebook.cell_count,
        status=notebook.status,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
    )


@router.delete("/notebooks/{notebook_id}", status_code=204)
async def delete_notebook(
    notebook_id: str,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook).where(
            Notebook.display_id == notebook_id, Notebook.customer_id == customer.id
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    await session.delete(notebook)
    await session.commit()


@router.get("/notebooks/{notebook_id}/state")
async def get_notebook_state(
    notebook_id: str,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook).where(
            Notebook.display_id == notebook_id, Notebook.customer_id == customer.id
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    state_json = await read_notebook_state(customer, notebook_id)
    if not state_json:
        return {"notebook_id": notebook_id, "name": notebook.name, "cells": [], "cell_order": []}

    return json.loads(state_json)


@router.put("/notebooks/{notebook_id}/state")
async def save_notebook_state(
    notebook_id: str,
    state: dict,
    customer: Customer = Depends(get_customer),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Notebook).where(
            Notebook.display_id == notebook_id, Notebook.customer_id == customer.id
        )
    )
    notebook = result.scalar_one_or_none()
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")

    uri = await write_notebook_state(
        customer=customer,
        notebook_id=notebook_id,
        state_json=json.dumps(state),
    )
    notebook.icechunk_state_uri = uri
    notebook.cell_count = len(state.get("cells", []))
    await session.commit()

    return {"status": "saved", "notebook_id": notebook_id}
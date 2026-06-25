import uuid
from datetime import datetime
from typing import Optional

from azure.storage.blob import BlobServiceClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db import Customer

settings = get_settings()


def _container_name() -> str:
    short_id = str(uuid.uuid4())[:12]
    return f"container-customer-{short_id}"


def _icechunk_store_uri(container_name: str) -> str:
    return f"icechunk://azure://{settings.azure_storage_account}/{container_name}"


def _get_blob_service() -> Optional[BlobServiceClient]:
    if settings.azure_storage_connection_string:
        return BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
    return None


async def provision_customer(
    session: AsyncSession,
    email: str,
    name: Optional[str] = None,
    stripe_customer_id: Optional[str] = None,
) -> Customer:
    container_name = _container_name()
    store_uri = _icechunk_store_uri(container_name)

    blob_service = _get_blob_service()
    if blob_service:
        blob_service.create_container(container_name)

    customer = Customer(
        email=email,
        name=name,
        stripe_customer_id=stripe_customer_id,
        azure_container_name=container_name,
        icechunk_store_uri=store_uri,
        storage_quota_bytes=settings.default_storage_quota_bytes,
    )
    session.add(customer)
    await session.commit()
    await session.refresh(customer)
    return customer


async def get_customer_by_id(session: AsyncSession, customer_id: str) -> Optional[Customer]:
    result = await session.execute(select(Customer).where(Customer.id == customer_id))
    return result.scalar_one_or_none()


async def get_customer_by_stripe_id(
    session: AsyncSession, stripe_customer_id: str
) -> Optional[Customer]:
    result = await session.execute(
        select(Customer).where(Customer.stripe_customer_id == stripe_customer_id)
    )
    return result.scalar_one_or_none()


async def update_subscription_status(
    session: AsyncSession,
    customer: Customer,
    status: str,
    current_period_end: Optional[datetime] = None,
) -> Customer:
    customer.subscription_status = status
    if current_period_end:
        customer.current_period_end = current_period_end
    await session.commit()
    await session.refresh(customer)
    return customer
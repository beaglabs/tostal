from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.db import Customer
from app.services.db import get_session

settings = get_settings()
security = HTTPBearer(auto_error=False)


def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_customer_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> str:
    if settings.environment == "development" and not credentials:
        dev_customer_id = request.headers.get("X-Dev-Customer-Id")
        if dev_customer_id:
            return dev_customer_id
        raise HTTPException(status_code=401, detail="Authentication required")

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_jwt(credentials.credentials)
    customer_id = payload.get("customer_id")
    if not customer_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing customer_id")
    return str(customer_id)


async def get_customer(
    customer_id: str = Depends(get_current_customer_id),
    session: AsyncSession = Depends(get_session),
) -> Customer:
    result = await session.execute(select(Customer).where(Customer.id == customer_id))
    customer = result.scalar_one_or_none()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer
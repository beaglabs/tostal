import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_ingest_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/ingest")
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_classifiers_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/classifiers", json={})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_krigging_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/krigging", json={})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_segmentor_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/segmentor", json={})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_notebooks_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/notebooks")
        assert response.status_code == 401
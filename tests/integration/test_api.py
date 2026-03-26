"""Integration tests for the FastAPI app (requires Qdrant running)."""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")


@pytest.mark.asyncio
async def test_list_collections(client):
    resp = await client.get("/v1/collections")
    assert resp.status_code == 200
    assert "collections" in resp.json()


@pytest.mark.asyncio
async def test_query_empty(client):
    resp = await client.post("/v1/query", json={"query": ""})
    assert resp.status_code == 400

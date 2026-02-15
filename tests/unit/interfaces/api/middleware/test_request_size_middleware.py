"""Tests for request size middleware."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from src.interfaces.api.middleware.request_size_middleware import RequestSizeMiddleware


def _echo(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@pytest.fixture
def app() -> Starlette:
    a = Starlette(routes=[Route("/upload", _echo, methods=["POST"])])
    a.add_middleware(RequestSizeMiddleware, max_bytes=100)
    return a


@pytest.mark.asyncio
async def test_small_request_passes(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/upload", content=b"small")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_large_request_rejected(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/upload", content=b"x" * 200)
    assert resp.status_code == 413

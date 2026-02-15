"""Tests for security headers middleware."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from src.interfaces.api.middleware.security_headers_middleware import SecurityHeadersMiddleware


def _homepage(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@pytest.fixture
def app() -> Starlette:
    a = Starlette(routes=[Route("/", _homepage)])
    a.add_middleware(SecurityHeadersMiddleware)
    return a


@pytest.mark.asyncio
async def test_security_headers_present(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "max-age=" in resp.headers["Strict-Transport-Security"]
    assert resp.headers["Content-Security-Policy"] == "default-src 'self'"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["Permissions-Policy"] == "camera=(), microphone=()"

"""Tests for JWT auth middleware."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.infrastructure.config.settings import get_settings
from src.infrastructure.security.jwt_handler import JWTHandler
from src.interfaces.api.middleware.jwt_auth_middleware import JWTAuthMiddleware

_handler = JWTHandler(secret_key="test-secret")


def _protected(request: Request) -> JSONResponse:
    return JSONResponse({"user": request.state.user.get("sub", "unknown")})


def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Starlette:
    # Override environment to enable JWT auth for unit tests
    monkeypatch.setenv("ENVIRONMENT", "unit-testing")
    monkeypatch.setenv("JWT__ENABLED", "true")
    get_settings.cache_clear()

    a = Starlette(routes=[Route("/protected", _protected), Route("/health", _health)])
    a.add_middleware(JWTAuthMiddleware, jwt_handler=_handler)
    return a


@pytest.mark.asyncio
async def test_skip_public_paths(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/protected")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_returns_403(app: Starlette) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/protected", headers={"Authorization": "Bearer bad.token"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_valid_token_passes(app: Starlette) -> None:
    token = _handler.create_access_token({"sub": "testuser"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["user"] == "testuser"


@pytest.mark.asyncio
async def test_refresh_token_rejected(app: Starlette) -> None:
    token = _handler.create_refresh_token({"sub": "testuser"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403

"""Tests for auth routes."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.infrastructure.security.jwt_handler import JWTHandler


def _make_app():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI

    from src.interfaces.api.routes.auth_routes import router as auth_router

    app = FastAPI()
    app.include_router(auth_router)
    return app


@pytest.fixture
def app():  # type: ignore[no-untyped-def]
    return _make_app()


@pytest.mark.asyncio
async def test_token_no_api_keys_returns_501(app) -> None:  # type: ignore[no-untyped-def]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/auth/token",
            json={"client_id": "test", "client_secret": "secret"},
        )
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_token_invalid_credentials_returns_401(app) -> None:  # type: ignore[no-untyped-def]
    with patch("src.interfaces.api.routes.auth_routes.get_settings") as mock_settings:
        mock_settings.return_value.API_KEYS = ["valid-key"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/auth/token",
                json={"client_id": "test", "client_secret": "wrong"},
            )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_token_valid_credentials_returns_tokens(app) -> None:  # type: ignore[no-untyped-def]
    with patch("src.interfaces.api.routes.auth_routes.get_settings") as mock_settings:
        mock_settings.return_value.API_KEYS = ["valid-key"]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/auth/token",
                json={"client_id": "test", "client_secret": "valid-key"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_refresh_with_access_token_fails(app) -> None:  # type: ignore[no-untyped-def]
    handler = JWTHandler()
    access = handler.create_access_token({"sub": "user"})
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": access},
        )
    assert resp.status_code == 401

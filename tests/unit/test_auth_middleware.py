"""Tests for API key authentication middleware."""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.interfaces.api.middleware.auth_middleware import AuthMiddleware


async def _homepage(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("healthy")


def _make_app(api_keys: list[str], monkeypatch: pytest.MonkeyPatch) -> Starlette:
    # Override environment to enable auth for unit tests
    monkeypatch.setenv("ENVIRONMENT", "unit-testing")
    from src.infrastructure.config.settings import get_settings

    get_settings.cache_clear()

    app = Starlette(
        routes=[
            Route("/", _homepage),
            Route("/health", _health),
        ],
    )
    app.add_middleware(AuthMiddleware, api_keys=api_keys)
    return app


def test_auth_passthrough_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_make_app(api_keys=[], monkeypatch=monkeypatch))
    response = client.get("/")
    assert response.status_code == 200


def test_auth_skips_health(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_make_app(api_keys=["secret"], monkeypatch=monkeypatch))
    response = client.get("/health")
    assert response.status_code == 200


def test_auth_missing_key_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_make_app(api_keys=["secret"], monkeypatch=monkeypatch))
    response = client.get("/")
    assert response.status_code == 401


def test_auth_invalid_key_returns_403(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_make_app(api_keys=["secret"], monkeypatch=monkeypatch))
    response = client.get("/", headers={"X-API-Key": "wrong"})
    assert response.status_code == 403


def test_auth_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_make_app(api_keys=["secret"], monkeypatch=monkeypatch))
    response = client.get("/", headers={"X-API-Key": "secret"})
    assert response.status_code == 200

"""Tests for rate limit middleware."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.interfaces.api.middleware.rate_limit_middleware import RateLimitMiddleware


async def _homepage(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _make_app(rpm: int = 60) -> Starlette:
    app = Starlette(routes=[Route("/", _homepage)])
    app.add_middleware(RateLimitMiddleware, rpm=rpm)
    return app


def test_rate_limit_allows_normal_traffic() -> None:
    client = TestClient(_make_app(rpm=60))
    response = client.get("/")
    assert response.status_code == 200


def test_rate_limit_blocks_excess_traffic() -> None:
    client = TestClient(_make_app(rpm=2))
    # Exhaust the bucket: initial capacity is 2
    client.get("/")
    client.get("/")
    response = client.get("/")
    assert response.status_code == 429
    assert "Retry-After" in response.headers

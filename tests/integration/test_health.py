# tests/integration/test_health.py
"""Integration tests for health and readiness endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.interfaces.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)

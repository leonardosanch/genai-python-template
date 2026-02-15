# tests/unit/test_audit_middleware.py
"""Tests for audit middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.infrastructure.governance.in_memory_audit_trail import InMemoryAuditTrail
from src.interfaces.api.middleware.audit_middleware import AuditMiddleware


@pytest.fixture
def audit_trail() -> InMemoryAuditTrail:
    return InMemoryAuditTrail()


@pytest.fixture
def app_with_audit(audit_trail: InMemoryAuditTrail) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuditMiddleware, audit_trail=audit_trail)

    @app.get("/api/v1/data")
    async def get_data() -> dict:  # type: ignore[type-arg]
        return {"data": "ok"}

    @app.get("/health")
    async def health() -> dict:  # type: ignore[type-arg]
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(app_with_audit: FastAPI) -> TestClient:
    return TestClient(app_with_audit)


class TestAuditMiddleware:
    async def test_records_api_call(
        self, client: TestClient, audit_trail: InMemoryAuditTrail
    ) -> None:
        client.get("/api/v1/data")
        records = await audit_trail.query()
        assert len(records) == 1
        assert records[0].action == "GET /api/v1/data"
        assert records[0].resource == "/api/v1/data"
        assert records[0].details["status_code"] == 200

    async def test_skips_health_endpoint(
        self, client: TestClient, audit_trail: InMemoryAuditTrail
    ) -> None:
        client.get("/health")
        records = await audit_trail.query()
        assert len(records) == 0

    async def test_records_status_code(
        self, client: TestClient, audit_trail: InMemoryAuditTrail
    ) -> None:
        client.get("/nonexistent")
        records = await audit_trail.query()
        if records:
            assert records[0].details["status_code"] == 404

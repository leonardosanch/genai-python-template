# tests/unit/test_tenant_context.py
"""Tests for tenant context and middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.value_objects.tenant_context import TenantContext, TenantTier
from src.interfaces.api.middleware.tenant_middleware import (
    TenantMiddleware,
    get_current_tenant,
)


class TestTenantContext:
    def test_create_valid(self) -> None:
        ctx = TenantContext(tenant_id="t1", tenant_name="Acme", tier=TenantTier.PREMIUM)
        assert ctx.tenant_id == "t1"
        assert ctx.tier == TenantTier.PREMIUM

    def test_empty_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TenantContext(tenant_id="")

    def test_whitespace_tenant_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            TenantContext(tenant_id="   ")

    def test_frozen(self) -> None:
        ctx = TenantContext(tenant_id="t1")
        with pytest.raises(AttributeError):
            ctx.tenant_id = "t2"  # type: ignore[misc]

    def test_default_tier(self) -> None:
        ctx = TenantContext(tenant_id="t1")
        assert ctx.tier == TenantTier.STANDARD


class TestTenantMiddleware:
    @pytest.fixture
    def app_with_tenant(self) -> FastAPI:
        app = FastAPI()
        app.add_middleware(TenantMiddleware, require_tenant=False)

        @app.get("/api/v1/data")
        async def get_data() -> dict:  # type: ignore[type-arg]
            tenant = get_current_tenant()
            return {
                "tenant_id": tenant.tenant_id if tenant else None,
                "tier": tenant.tier.value if tenant else None,
            }

        @app.get("/health")
        async def health() -> dict:  # type: ignore[type-arg]
            return {"status": "ok"}

        return app

    def test_with_tenant_header(self, app_with_tenant: FastAPI) -> None:
        client = TestClient(app_with_tenant)
        response = client.get(
            "/api/v1/data",
            headers={"x-tenant-id": "acme", "x-tenant-tier": "premium"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "acme"
        assert data["tier"] == "premium"

    def test_without_tenant_header_optional(self, app_with_tenant: FastAPI) -> None:
        client = TestClient(app_with_tenant)
        response = client.get("/api/v1/data")
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] is None

    def test_health_skipped(self, app_with_tenant: FastAPI) -> None:
        client = TestClient(app_with_tenant)
        response = client.get("/health")
        assert response.status_code == 200

    def test_required_tenant_missing(self) -> None:
        app = FastAPI()
        app.add_middleware(TenantMiddleware, require_tenant=True)

        @app.get("/api/v1/data")
        async def get_data() -> dict:  # type: ignore[type-arg]
            return {"ok": True}

        client = TestClient(app)
        response = client.get("/api/v1/data")
        assert response.status_code == 400
        assert "X-Tenant-ID" in response.json()["detail"]

    def test_invalid_tier_defaults_to_standard(self, app_with_tenant: FastAPI) -> None:
        client = TestClient(app_with_tenant)
        response = client.get(
            "/api/v1/data",
            headers={"x-tenant-id": "t1", "x-tenant-tier": "invalid_tier"},
        )
        data = response.json()
        assert data["tier"] == "standard"

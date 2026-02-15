"""Tests for RBAC dependency — role-based route protection."""

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from src.domain.value_objects.role import Role
from src.interfaces.api.dependencies.rbac import require_roles


def _build_app() -> FastAPI:
    """Tiny app with role-protected routes for testing."""
    app = FastAPI()

    @app.get("/admin", dependencies=[Depends(require_roles(Role.ADMIN))])
    async def admin_only() -> dict[str, str]:
        return {"ok": "admin"}

    @app.get(
        "/operate",
        dependencies=[Depends(require_roles(Role.OPERATOR, Role.ADMIN))],
    )
    async def operator_or_admin() -> dict[str, str]:
        return {"ok": "operate"}

    @app.get("/public")
    async def public() -> dict[str, str]:
        return {"ok": "public"}

    return app


@pytest.fixture()
def app() -> FastAPI:
    return _build_app()


class TestRequireRoles:
    """Tests for the require_roles dependency."""

    @pytest.mark.asyncio()
    async def test_no_user_returns_401(self, app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/admin")
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]

    @pytest.mark.asyncio()
    async def test_user_without_role_returns_403(self, app: FastAPI) -> None:
        # Simulate JWT middleware injecting user with viewer role
        @app.middleware("http")
        async def inject_user(request, call_next):  # noqa: ANN001, ANN202
            request.state.user = {"sub": "u1", "roles": ["viewer"]}
            return await call_next(request)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/admin")
        assert resp.status_code == 403
        assert "Insufficient permissions" in resp.json()["detail"]

    @pytest.mark.asyncio()
    async def test_user_with_correct_role_passes(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def inject_user(request, call_next):  # noqa: ANN001, ANN202
            request.state.user = {"sub": "u1", "roles": ["admin"]}
            return await call_next(request)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/admin")
        assert resp.status_code == 200
        assert resp.json() == {"ok": "admin"}

    @pytest.mark.asyncio()
    async def test_multiple_allowed_roles(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def inject_user(request, call_next):  # noqa: ANN001, ANN202
            request.state.user = {"sub": "u1", "roles": ["operator"]}
            return await call_next(request)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/operate")
        assert resp.status_code == 200
        assert resp.json() == {"ok": "operate"}

    @pytest.mark.asyncio()
    async def test_empty_roles_returns_403(self, app: FastAPI) -> None:
        @app.middleware("http")
        async def inject_user(request, call_next):  # noqa: ANN001, ANN202
            request.state.user = {"sub": "u1", "roles": []}
            return await call_next(request)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/admin")
        assert resp.status_code == 403

    @pytest.mark.asyncio()
    async def test_public_route_no_rbac(self, app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/public")
        assert resp.status_code == 200

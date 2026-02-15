# src/interfaces/api/middleware/tenant_middleware.py
"""Middleware for extracting and propagating tenant context."""

from contextvars import ContextVar

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.domain.value_objects.tenant_context import TenantContext, TenantTier

logger = structlog.get_logger(__name__)

_tenant_ctx: ContextVar[TenantContext | None] = ContextVar("tenant_ctx", default=None)


def get_current_tenant() -> TenantContext | None:
    """Get the current tenant context from contextvars."""
    return _tenant_ctx.get()


class TenantMiddleware(BaseHTTPMiddleware):
    """Extracts tenant context from X-Tenant-ID header.

    Sets the tenant context in a ContextVar for downstream access.
    Returns 400 if the header is missing on tenant-required paths.
    """

    # Paths that don't require tenant context
    SKIP_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json", "/redoc"})

    def __init__(self, app: object, require_tenant: bool = False) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._require_tenant = require_tenant

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        tenant_id = request.headers.get("x-tenant-id")
        tenant_name = request.headers.get("x-tenant-name", "")
        tier_str = request.headers.get("x-tenant-tier", "standard")

        if not tenant_id:
            if self._require_tenant:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "X-Tenant-ID header is required."},
                )
            return await call_next(request)

        try:
            tier = TenantTier(tier_str.lower())
        except ValueError:
            tier = TenantTier.STANDARD

        ctx = TenantContext(tenant_id=tenant_id, tenant_name=tenant_name, tier=tier)
        token = _tenant_ctx.set(ctx)

        try:
            response = await call_next(request)
        finally:
            _tenant_ctx.reset(token)

        return response

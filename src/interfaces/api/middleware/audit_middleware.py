# src/interfaces/api/middleware/audit_middleware.py
"""Middleware for recording API request audit trails."""

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.domain.entities.audit_record import AuditRecord
from src.domain.ports.audit_trail_port import AuditTrailPort

logger = structlog.get_logger(__name__)

# Paths excluded from audit logging
_SKIP_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json", "/redoc"})


class AuditMiddleware(BaseHTTPMiddleware):
    """Records an audit entry for each API request.

    Captures method, path, status code, and duration.
    """

    def __init__(self, app: object, audit_trail: AuditTrailPort) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._audit_trail = audit_trail

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        actor = "anonymous"
        if hasattr(request.state, "user_id"):
            actor = request.state.user_id

        entry = AuditRecord(
            action=f"{request.method} {request.url.path}",
            actor=actor,
            resource=request.url.path,
            details={
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "method": request.method,
                "client": request.client.host if request.client else "unknown",
            },
        )

        try:
            await self._audit_trail.record(entry)
        except Exception:
            logger.warning("audit_middleware_record_failed", path=request.url.path)

        return response

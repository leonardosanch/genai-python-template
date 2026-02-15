# src/interfaces/api/middleware/logging_middleware.py
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and adding a correlation ID."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process the request and log its details."""
        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        with structlog.contextvars.bound_contextvars(
            correlation_id=correlation_id,
            request_id=request.headers.get("x-request-id"),
            user_agent=request.headers.get("user-agent"),
        ):
            response = await call_next(request)

            process_time = time.time() - start_time
            response.headers["X-Correlation-ID"] = correlation_id

            logger.info(
                "api_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(process_time * 1000),
            )
            return response

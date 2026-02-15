"""API key authentication middleware.

Validates X-API-Key header against configured keys.
Empty API_KEYS list disables authentication (passthrough).
"""

import json
from http import HTTPStatus

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.infrastructure.config import get_settings

_SKIP_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""

    def __init__(self, app: object, api_keys: list[str]) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._api_keys = set(api_keys)
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth for health/docs endpoints or in testing
        if request.url.path in _SKIP_PATHS or self._settings.ENVIRONMENT == "testing":
            return await call_next(request)

        # Passthrough if no keys configured
        if not self._api_keys:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return Response(
                content=json.dumps({"error": "Missing API key"}),
                status_code=HTTPStatus.UNAUTHORIZED,
                media_type="application/json",
            )

        if api_key not in self._api_keys:
            return Response(
                content=json.dumps({"error": "Invalid API key"}),
                status_code=HTTPStatus.FORBIDDEN,
                media_type="application/json",
            )

        return await call_next(request)

"""JWT Bearer authentication middleware."""

import json
from http import HTTPStatus

from jose import JWTError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.infrastructure.config import get_settings
from src.infrastructure.security.jwt_handler import JWTHandler

_SKIP_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/api/v1/auth/token"}


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Validates Authorization: Bearer <JWT> header.

    Injects decoded claims into ``request.state.user``.
    Skips public paths (health, docs, auth).
    """

    def __init__(self, app: object, jwt_handler: JWTHandler | None = None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._jwt = jwt_handler or JWTHandler()
        self._settings = get_settings()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if (
            request.url.path in _SKIP_PATHS
            or not self._settings.jwt.ENABLED
            or self._settings.ENVIRONMENT == "testing"
        ):
            return await call_next(request)

        # Skip WebSocket upgrades (handled at route level)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(
                content=json.dumps({"error": "Missing or invalid Authorization header"}),
                status_code=HTTPStatus.UNAUTHORIZED,
                media_type="application/json",
            )

        token = auth_header.removeprefix("Bearer ").strip()
        try:
            claims = self._jwt.decode_token(token)
        except JWTError:
            return Response(
                content=json.dumps({"error": "Invalid or expired token"}),
                status_code=HTTPStatus.FORBIDDEN,
                media_type="application/json",
            )

        if claims.get("type") != "access":
            return Response(
                content=json.dumps({"error": "Invalid token type"}),
                status_code=HTTPStatus.FORBIDDEN,
                media_type="application/json",
            )

        request.state.user = claims
        return await call_next(request)

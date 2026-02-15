"""Middleware that rejects request bodies exceeding a configurable size limit."""

import json
from http import HTTPStatus

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Rejects requests with Content-Length exceeding ``max_bytes``."""

    def __init__(self, app: object, max_bytes: int = 1_048_576) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._max_bytes:
            return Response(
                content=json.dumps(
                    {
                        "error": "Payload too large",
                        "max_bytes": self._max_bytes,
                    }
                ),
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                media_type="application/json",
            )
        return await call_next(request)

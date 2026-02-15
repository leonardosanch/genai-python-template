# src/interfaces/api/middleware/input_sanitization_middleware.py
"""Middleware to detect prompt injection attempts in request bodies."""

import json
import re

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Paths excluded from scanning
_SKIP_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json", "/redoc"})

# Prompt injection patterns (case-insensitive)
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\|?\s*system\s*\|?>", re.IGNORECASE),
    re.compile(r"\brole\s*:\s*system\b", re.IGNORECASE),
    re.compile(r"```\s*system\b", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?a?\s*(new|different)\s+ai", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous)", re.IGNORECASE),
    re.compile(r"override\s+(your\s+)?(instructions|rules|guidelines)", re.IGNORECASE),
]


def _scan_value(value: str) -> str | None:
    """Check a string value against injection patterns. Returns matched pattern or None."""
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(0)
    return None


def _scan_obj(obj: object) -> str | None:
    """Recursively scan a JSON-parsed object for injection patterns."""
    if isinstance(obj, str):
        return _scan_value(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            result = _scan_obj(v)
            if result:
                return result
    if isinstance(obj, list):
        for item in obj:
            result = _scan_obj(item)
            if result:
                return result
    return None


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Scans JSON request bodies for prompt injection patterns.

    Only processes POST, PUT, PATCH methods with JSON content-type.
    Returns 400 if injection is detected.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _SKIP_PATHS:
            return await call_next(request)

        if request.method not in {"POST", "PUT", "PATCH"}:
            return await call_next(request)

        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return await call_next(request)

        body = await request.body()
        if not body:
            return await call_next(request)

        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return await call_next(request)

        matched = _scan_obj(parsed)
        if matched:
            logger.warning(
                "prompt_injection_detected",
                path=request.url.path,
                matched_pattern=matched,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=400,
                content={"detail": "Request rejected: potentially unsafe input detected."},
            )

        return await call_next(request)

# src/interfaces/api/middleware/__init__.py
"""API middleware and exception handlers."""

from .audit_middleware import AuditMiddleware
from .auth_middleware import AuthMiddleware
from .error_handler import (
    domain_error_exception_handler,
    http_exception_handler,
    pydantic_validation_exception_handler,
    request_validation_exception_handler,
    unhandled_exception_handler,
)
from .input_sanitization_middleware import InputSanitizationMiddleware
from .jwt_auth_middleware import JWTAuthMiddleware
from .logging_middleware import LoggingMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .request_size_middleware import RequestSizeMiddleware
from .security_headers_middleware import SecurityHeadersMiddleware

__all__ = [
    "AuditMiddleware",
    "AuthMiddleware",
    "InputSanitizationMiddleware",
    "JWTAuthMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "RequestSizeMiddleware",
    "SecurityHeadersMiddleware",
    "domain_error_exception_handler",
    "http_exception_handler",
    "pydantic_validation_exception_handler",
    "request_validation_exception_handler",
    "unhandled_exception_handler",
]

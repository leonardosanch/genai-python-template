# src/interfaces/api/middleware/error_handler.py
import json
import traceback
from http import HTTPStatus

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import (
    ValidationError as PydanticValidationError,  # conflicts with Domain ValidationError
)

from src.domain.exceptions import (
    DomainError,
    HallucinationError,
    LLMError,
    LLMRateLimitError,
    RetrievalError,
    TokenBudgetExceededError,
    ValidationError,
)

logger = structlog.get_logger()


async def domain_error_exception_handler(request: Request, exc: DomainError) -> Response:
    """Handles domain-specific exceptions, translating them into appropriate HTTP responses."""
    status_code: HTTPStatus
    detail: str = exc.args[0] if exc.args else "An unexpected domain error occurred."

    if isinstance(exc, ValidationError):
        status_code = HTTPStatus.BAD_REQUEST
    elif isinstance(exc, HallucinationError):
        status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    elif isinstance(exc, (LLMRateLimitError, TokenBudgetExceededError)):
        status_code = HTTPStatus.TOO_MANY_REQUESTS
    elif isinstance(exc, (LLMError, RetrievalError)):
        status_code = HTTPStatus.BAD_GATEWAY
    else:
        status_code = HTTPStatus.INTERNAL_SERVER_ERROR

    # Log the full exception traceback for internal debugging
    logger.error(
        "domain_exception_caught",
        error=detail,
        status_code=status_code.value,
        exc_info=exc,
        correlation_id=structlog.contextvars.get_contextvars().get("correlation_id"),
    )

    response_content = {"error": detail}
    if correlation_id := structlog.contextvars.get_contextvars().get("correlation_id"):
        response_content["correlation_id"] = correlation_id

    return Response(
        content=json.dumps(response_content),
        status_code=status_code.value,
        media_type="application/json",
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> Response:
    """Handles FastAPI's HTTPException, adding correlation ID."""
    response_content = {"error": exc.detail}
    if correlation_id := structlog.contextvars.get_contextvars().get("correlation_id"):
        response_content["correlation_id"] = correlation_id

    return Response(
        content=json.dumps(response_content),
        status_code=exc.status_code,
        media_type="application/json",
    )


async def pydantic_validation_exception_handler(
    request: Request, exc: PydanticValidationError
) -> Response:
    """Handles Pydantic validation errors (e.g., from FastAPI request bodies)."""
    detail = exc.errors()
    logger.warning(
        "request_validation_error",
        errors=detail,
        correlation_id=structlog.contextvars.get_contextvars().get("correlation_id"),
    )
    response_content = {"error": "Validation error", "details": detail}
    if correlation_id := structlog.contextvars.get_contextvars().get("correlation_id"):
        response_content["correlation_id"] = correlation_id

    return Response(
        content=json.dumps(response_content),
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        media_type="application/json",
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Handles FastAPI's RequestValidationError."""
    detail = exc.errors()
    logger.warning(
        "request_validation_error",
        errors=detail,
        correlation_id=structlog.contextvars.get_contextvars().get("correlation_id"),
    )
    response_content: dict[str, object] = {"error": "Validation error", "details": detail}
    if correlation_id := structlog.contextvars.get_contextvars().get("correlation_id"):
        response_content["correlation_id"] = correlation_id

    return Response(
        content=json.dumps(response_content),
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        media_type="application/json",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Handles all unhandled exceptions, translating them into a generic HTTP 500 response."""
    correlation_id = structlog.contextvars.get_contextvars().get("correlation_id")
    error_detail = f"An unexpected error occurred: {exc}"

    logger.critical(
        "unhandled_exception",
        error=error_detail,
        exc_info=exc,
        traceback=traceback.format_exc(),
        correlation_id=correlation_id,
    )

    response_content = {"error": "Internal Server Error"}
    if correlation_id:
        response_content["correlation_id"] = correlation_id

    return Response(
        content=json.dumps(response_content),
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        media_type="application/json",
    )

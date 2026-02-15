"""Structured logging setup with structlog.

Reference implementation for JSON-formatted logging.
Never log secrets, PII, or raw prompts with sensitive data.
"""

import logging

import structlog

from src.infrastructure.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure structlog for JSON structured logging."""
    log_level = getattr(logging, settings.observability.LOG_LEVEL.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
